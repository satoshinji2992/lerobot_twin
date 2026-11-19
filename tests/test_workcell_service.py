"""Exercise the workcell CLI and HTTP contract on loopback with an isolated config."""

import base64
import io
import json
import math
import os
import re
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

import pytest
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
CAPTURE = ROOT / "data" / "vision" / "auto-check.jpg"
WEIGHTS = ROOT / "data" / "vision" / "FastSAM-s.pt"
REGISTRATION = {
    "top_center": [100, 10, 40],
    "x_point": [150, 10, 40],
    "y_point": [100, 60, 40],
    "check": {"reference": [10, 10, 29], "base": [110, 20, 40]},
}


class Client:
    def __init__(self, port, token, config):
        self.origin = f"http://127.0.0.1:{port}"
        self.token = token
        self.config = config
        self.http = build_opener(ProxyHandler({}))

    def call(self, path, data=None, token=None, headers=None):
        headers = {"X-Pair": self.token if token is None else token, **(headers or {})}
        if isinstance(data, dict):
            data = json.dumps(data).encode()
            headers["Content-Type"] = "application/json"
        req = Request(self.origin + path, data=data, headers=headers)
        with self.http.open(req, timeout=8) as response:
            if response.headers.get_content_type() == "application/json":
                return json.load(response)
            return response.read().decode("utf-8")

    def camera(self):
        url = urlsplit(self.call("/state")["connection"]["local_url"])
        assert url.path == "/camera" and re.fullmatch(r"\d{8}", url.fragment)
        return url.fragment

    def start(self, pin, profile=None):
        body = {} if profile is None else {"profile": profile}
        return self.call("/camera/start", body, pin)["session"]

    def upload(self, pin, session, jpeg):
        headers = {"X-Session": session}
        seq = self.call("/camera/ticket", b"", pin, headers)["seq"]
        headers.update(
            {
                "X-Seq": str(seq),
                "X-Media-Time": str(time.monotonic()),
                "X-Next-Ticket": "1",
                "Content-Type": "image/jpeg",
            }
        )
        headers["X-Workcell-Status"] = "1"
        response = self.call("/camera/frame", jpeg, pin, headers)
        assert response["next_seq"] > seq
        assert response["calibration"] in {"missing","saved","matched","lens_unverified","mismatch"}
        return seq

    def wait_state(self, predicate, timeout=8):
        until = time.monotonic() + timeout
        while time.monotonic() < until:
            state = self.call("/state")
            if predicate(state):
                return state
            time.sleep(0.08)
        pytest.fail(f"Timed out waiting for state; last status was {state.get('status')!r}")


@pytest.fixture(scope="module")
def service(tmp_path_factory):
    with launch_workcell(tmp_path_factory.mktemp("workcell-http"), {"cell_note": "retained"}) as client:
        yield client


@contextmanager
def launch_workcell(tmp, initial_config):
    # Use the CLI's preferred interpreter so the test is hermetic with respect
    # to whatever interpreter is running pytest. Do not launch setup/downloads.
    py = next(
        (
            path
            for env in [".workcell-venv", ".vision-venv"]
            for path in [ROOT / env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")]
            if path.is_file()
        ),
        Path(sys.executable),
    )
    config = tmp / "workcell.json"
    config.write_text(json.dumps(initial_config), encoding="utf-8")
    with socket.socket() as socket_:
        socket_.bind(("127.0.0.1", 0))
        port = socket_.getsockname()[1]
    log = tmp / "service.log"
    command = [
        str(py),
        str(ROOT / "twin.py"),
        "workcell",
        "--local-camera",
        "--frame",
        "workcell",
        "--port",
        str(port),
        "--config",
        str(config),
        "--no-browser",
        "--device",
        "cpu",
    ]
    if not WEIGHTS.is_file():
        # Page routing, persistence and the capture protocol are testable without
        # downloading weights. The captured-image integration case is skipped.
        command[:3] = [
            str(py),
            "-c",
            "from twin_depot import robot_view as view; "
            "view.build_predictor=lambda *args: (lambda image: [], 'cpu'); view.main()",
        ]
    with log.open("wb") as output:
        proc = subprocess.Popen(
            command,
            cwd=ROOT,
            env={**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1",
                 "TWIN_WORKCELL_ENDPOINT_FILE": str(tmp / "active-workcell.json")},
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=os.name != "nt",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            until = time.monotonic() + 45
            while time.monotonic() < until:
                output_text = log.read_text(encoding="utf-8", errors="replace")
                found = re.search(rf"http://127\.0\.0\.1:{port}/?#([A-Za-z0-9_-]+)", output_text)
                if found:
                    client = Client(port, found[1], config)
                    if client.call("/state").get("connection"):
                        break
                if proc.poll() is not None:
                    pytest.fail(f"Workcell exited with {proc.returncode}: {output_text[-1600:]}")
                time.sleep(0.1)
            else:
                pytest.fail("Workcell did not publish its loopback observation URL")
            yield client
        finally:
            if proc.poll() is None:
                # Windows venv launchers can fork another Python interpreter.
                # Stop only the process tree that belongs to this fixture.
                if os.name == "nt":
                    subprocess.run(
                        ["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, timeout=10
                    )
                else:
                    os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=10)


def test_pages_calibration_and_registration(service):
    for path in ("/", "/board", "/registration", "/camera"):
        assert "<html" in service.call(path).lower()
    with pytest.raises(HTTPError) as denied:
        service.call("/state", token="invalid")
    assert denied.value.code == 403
    assert service.camera()
    assert service.call("/board/status", {})["count"] == 0
    preview = service.call("/registration/preview", REGISTRATION)
    assert preview.get("result", preview)["can_apply"]
    assert "base_from_reference" not in json.loads(service.config.read_text("utf-8"))
    service.call("/registration/apply", REGISTRATION)
    saved = json.loads(service.config.read_text("utf-8"))
    assert saved["cell_note"] == "retained"
    assert [row[3] for row in saved["base_from_reference"][:3]] == pytest.approx([100, 10, 11])


def blank_jpeg(size=(640, 480)):
    out = io.BytesIO()
    Image.new("RGB", size, (70, 70, 70)).save(out, format="JPEG", quality=85)
    return out.getvalue()


def test_reconnect_invalidates_pending_uploads(service):
    pin = service.camera()
    old = service.start(pin)
    ticket = service.call("/camera/ticket", b"", pin, {"X-Session": old})["seq"]
    session = service.start(pin)
    assert session != old
    with pytest.raises(HTTPError) as rejected:
        service.call("/camera/frame", blank_jpeg(), pin, {"X-Session": old, "X-Seq": str(ticket)})
    assert 400 <= rejected.value.code < 500
    service.upload(pin, session, blank_jpeg())
    state = service.wait_state(lambda s: s.get("session") == session and s.get("fresh"))
    assert not state.get("rows") and not state.get("camera_pose")


def test_camera_ack_distinguishes_duplicate_and_recovery(service):
    pin = service.camera()
    session = service.start(pin)
    headers = {"X-Session": session, "X-Next-Ticket": "1"}
    seq = service.call("/camera/ticket", b"", pin, headers)["seq"]
    jpeg = blank_jpeg()
    for stamp, expected in [(0, True), (0, False), (0.04, True)]:
        headers.update({"X-Seq": str(seq), "X-Media-Time": str(stamp)})
        reply = service.call("/camera/frame", jpeg, pin, headers)
        assert reply["ok"] is expected
        if not expected:
            assert reply["dropped"] == "duplicate"
        seq = reply["next_seq"]
    rx = service.call("/state")["camera_rx"]
    assert (rx["received"], rx["duplicate"], rx["last_accepted_seq"]) == (3, 1, 3)


@pytest.mark.skipif(
    not (CAPTURE.is_file() and WEIGHTS.is_file()),
    reason="Local captured cube frame and FastSAM weights required for real-model replay",
)
def test_real_capture_occlusion_timeout_and_reconnection(service):
    pin = service.camera()
    service.call("/registration/apply", REGISTRATION)
    jpeg = CAPTURE.read_bytes()
    with Image.open(io.BytesIO(jpeg)) as capture:
        blank = blank_jpeg(capture.size)

    def acquire(session, registered=False):
        until = time.monotonic() + 30
        while time.monotonic() < until:
            service.upload(pin, session, jpeg)
            time.sleep(0.16)
            s = service.call("/state")
            if (
                s.get("session") == session
                and s.get("registered") is registered
                and {"red", "green"} <= {r["color"] for r in s.get("rows", [])}
            ):
                assert s.get("fresh") and s.get("camera_pose")
                assert s.get("sim", "").startswith("data:image/jpeg;base64,"), s.get("status")
                return s
        pytest.fail(f"Real capture was not reconstructed: {s.get('status')!r}")

    session = service.start(pin)
    measured = acquire(session)
    for row in measured["rows"]:
        assert len(row["xyz"]) == 3 and len(row["pose"]) == 4

    def xyz(frame, color):
        return next(row["xyz"] for row in frame["rows"] if row["color"] == color)

    delta = [g - r for g, r in zip(xyz(measured, "green"), xyz(measured, "red"), strict=True)]
    service.call("/registration/orient", {"quarter": 1, "epoch": measured["axis_epoch"]})
    measured = acquire(session, registered=True)
    assert xyz(measured, "red") == pytest.approx([100, 10, 25.5], abs=0.01)
    # B→C points toward -Y in the initial observed frame. After teaching it
    # as positive X, old (dx,dy,dz) becomes (-dy,dx,dz) about the red block.
    assert xyz(measured, "green") == pytest.approx([100 - delta[1], 10 + delta[0], 25.5 + delta[2]], abs=2)
    service.upload(pin, session, blank)
    hidden = service.wait_state(
        lambda s: (
            s.get("session") == session
            and s.get("fresh")
            and s["seq"] > measured["seq"]
            and not s.get("camera_pose")
        )
    )
    assert not hidden.get("rows") and not hidden.get("camera_pose")
    assert not hidden["registered"]
    with pytest.raises(HTTPError) as outdated:
        service.call("/registration/orient", {"quarter": 1, "epoch": measured["axis_epoch"]})
    assert outdated.value.code == 400
    acquire(session)
    stale = service.wait_state(lambda s: not s.get("fresh") and not s.get("rows"), timeout=6)
    assert not stale.get("camera_pose") and not stale.get("history")
    new_session = service.start(pin)
    assert new_session != session
    acquire(new_session)
    replacement = service.start(pin)
    restarted = service.call("/state")
    assert restarted["session"] == replacement
    assert not restarted.get("fresh") and not restarted.get("camera_pose") and not restarted.get("rows")


@pytest.mark.skipif(
    not (CAPTURE.is_file() and WEIGHTS.is_file()),
    reason="Local captured cube frame and FastSAM weights required for real-model replay",
)
def test_real_render_views_and_matched_camera_profile(tmp_path):
    jpeg = CAPTURE.read_bytes()
    with Image.open(io.BytesIO(jpeg)) as capture:
        w, h = capture.size
    profile = {
        "version": 1,
        "source": "local",
        "camera_key": "1" * 32,
        "native_size": [w, h],
        "video_size": [w, h],
        "zoom": 1.0,
        "facing": "environment",
        "resize_mode": "none",
    }
    # A known camera matrix exercises loading/matching; ground-truth optics
    # are not required for testing the render and calibration UI contract.
    f = math.hypot(w, h) / (2 * math.tan(math.radians(34)))
    intrinsics = {
        "image_size": [w, h],
        "K": [[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]],
        "dist": [0.0] * 5,
        "capture_profile": profile,
    }
    with launch_workcell(tmp_path, {"intrinsics": intrinsics}) as client:
        idle = client.call("/state")
        assert idle["calibration"]["state"]=="saved"
        assert idle["connection"]["transport"]=="direct"
        pin = client.camera()
        session = client.start(pin, profile)
        seq = 0
        scenes = {}

        def replay_until(session, predicate):
            until = time.monotonic() + 20
            while time.monotonic() < until:
                client.upload(pin, session, jpeg)
                time.sleep(0.15)
                snap = client.call("/state")
                if snap.get("session") == session and snap.get("fresh") and predicate(snap):
                    return snap
            pytest.fail(f"Expected snapshot never arrived: {snap.get('status')!r}")

        for view in ["overview", "detail", "side", "top"]:
            assert client.call("/view", {"view": view}) == {"ok": True, "view": view}
            snap = replay_until(
                session,
                lambda s, after=seq, expected=view: (
                    s["seq"] > after and s["view"] == expected and bool(s.get("sim"))
                ),
            )
            seq = snap["seq"]
            assert snap["calibration"]["state"] == "matched"
            assert snap["calibrated"] and snap["camera_pose"]
            assert {"red", "green"} <= {row["color"] for row in snap["rows"]}
            kind, data = snap["sim"].split(",", 1)
            assert kind == "data:image/jpeg;base64"
            with Image.open(io.BytesIO(base64.b64decode(data, validate=True))) as photo:
                assert photo.format == "JPEG"
                scenes[view] = photo.convert("RGB")
        # Decode the actual image. A changed label or a differently encoded
        # copy of the same image does not qualify as another viewing angle.
        diff = ImageChops.difference(scenes["detail"], scenes["top"]).convert("L")
        assert sum(diff.histogram()[15:]) > 0.015 * diff.width * diff.height

        for changed in ({"zoom": 2.0}, {"camera_key": "2" * 32}):
            other = client.start(pin, {**profile, **changed})
            paused = replay_until(other, lambda s: "不匹配" in s["status"])
            assert paused["calibration"]["state"] == "mismatch"
            assert not paused["calibrated"] and not paused["rows"]
            assert not paused.get("camera_pose") and not paused.get("sim")
        original = client.start(pin, profile)
        restored = replay_until(original, lambda s: bool(s.get("camera_pose") and s.get("sim")))
        assert restored["calibrated"] and restored["view"] == "top"
        assert {"red", "green"} <= {r["color"] for r in restored["rows"]}
        assert json.loads(client.config.read_text("utf-8"))["intrinsics"] == intrinsics


@pytest.mark.skipif(not CAPTURE.exists(),reason="Local cube photo for camera-only replay")
def test_fast_camera_does_not_publish_inferred_corners_as_measured(service):
    mode={"mode":"camera","pipeline":"fast"}
    assert service.call("/frame-mode",mode)==mode
    try:
        with Image.open(CAPTURE) as original:
            frame=original.copy()
        # Leave the green cube in the original optical frame; remove the red reference.
        frame.paste((37,39,38),(270,260,410,394))
        stream=io.BytesIO();frame.save(stream,"JPEG",quality=84)
        pin=service.camera();sid=service.start(pin)
        service.upload(pin,sid,stream.getvalue())
        snap=service.wait_state(lambda s:s.get("session")==sid and s.get("fresh"))
        assert snap["pipeline"]=="fast" and snap["frame_mode"]=="camera"
        assert snap["rows"]==[] and not snap.get("camera_pose") and not snap["registered"]
        assert snap["sim"] and any(e["color"]=="green" and len(e["pixels"])==8 for e in snap["completions"])
    finally:
        service.call("/frame-mode",{"mode":"workcell","pipeline":"precision"})
