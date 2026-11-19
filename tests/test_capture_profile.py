"""The source profile must follow the JPEG through capture and direct upload."""
import io
import json
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from twin_depot.vision import Instrument, capture_profile, handler
from twin_depot.workcell_source import LatestFrames


def reported():
    return dict(camera_key="0f"*16, native_size=[1280,720], video_size=[720,1280],
                zoom=1.0, facing="environment", resize_mode=None)


def jpeg():
    out = io.BytesIO()
    Image.new("RGB", (720,1280)).save(out, format="JPEG")
    return out.getvalue()


def test_route_and_optics_distinguish_identical_output_dimensions():
    data = dict(reported(), deviceId="never-forward-this", label="do-not-forward-this", source="forged")
    phone = capture_profile(data, "phone")
    local = capture_profile(data, "local")
    assert phone["source"] == "phone" and local["source"] == "local"
    assert "deviceId" not in phone and "label" not in phone
    assert phone != local
    assert capture_profile(dict(data, zoom=2), "phone") != phone
    assert capture_profile(reported(), "phone") == phone
    assert capture_profile(None, "phone") is None


@pytest.mark.parametrize("bad", [
    {"camera_key":"raw-device-id"}, {"native_size":[True,720]}, {"video_size":[1280,0]},
    {"zoom":float("nan")}, {"zoom":True}, {"resize_mode":"unknown"}, {"facing":[]}, {"version":2}
])
def test_invalid_metadata_rejected(bad):
    with pytest.raises(ValueError):
        capture_profile(dict(reported(), **bad), "phone")


def test_new_and_legacy_receivers_reset_profile_on_each_session():
    raw = jpeg()
    profile = capture_profile(reported(), "phone")
    for device in (LatestFrames(), Instrument(lambda image,plane: [])):
        session = device.start(profile)
        seq = device.ticket(session)
        device.receive(raw,session,seq)
        if isinstance(device, LatestFrames):
            assert device.read().jpeg == raw and device.read().profile == profile
        else:
            assert device.state(None)["profile"] == profile and device.state(None)["fresh"]
        device.start()
        assert (device.read().profile if isinstance(device, LatestFrames) else device.state(None)["profile"]) is None


@pytest.mark.parametrize(("mode", "endpoint", "expected"), [("phone", "/start", "phone"), ("desk", "/camera/start", "local")])
def test_http_source_uses_receiving_route_and_preserves_old_clients(mode, endpoint, expected):
    device = LatestFrames()
    receive = handler(mode, device, "sample", "127.0.0.1", 0, None)
    with ThreadingHTTPServer(("127.0.0.1", 0), receive) as server:
        thread = threading.Thread(target=lambda: server.serve_forever(.02), daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}{endpoint}"
            opener = build_opener(ProxyHandler({}))
            for body in (json.dumps({"profile": dict(reported(), source="arbitrary")}).encode(), b""):
                req = Request(url, data=body, headers={"X-Pair": "sample"})
                with opener.open(req, timeout=2) as response:
                    assert json.load(response)["session"] == device.read().session
                if body:
                    assert device.read().profile["source"] == expected
                else:
                    assert device.read().profile is None
        finally:
            server.shutdown()
            thread.join(2)
