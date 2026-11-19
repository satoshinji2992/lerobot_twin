"""LAN camera + workstation. The only input to the instrument is an image stream."""

import argparse
import base64
import hashlib
import io
import ipaddress
import json
import math
import os
import re
import secrets
import shutil
import socket
import tempfile
import threading
import time
import webbrowser
from collections import deque
from contextlib import ExitStack
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen

import cv2
import numpy as np
from PIL import Image

from .cube_measure import Plane, project
from .cube_tracking import CubeTracker, PREDICT_S
from .vision_tls import camera_tls

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "vision"
FRESH_S = 1.5
WEIGHT_SHA = "c9f78716a81c7aff0d608ccc73e1b82ab3aaad86005049f6a92106a0be6d0844"


def default_weights(variant="s"):
    filename = {"s": "FastSAM-s.pt", "x": "FastSAM-x.pt"}[variant]
    sha = WEIGHT_SHA if variant == "s" else "752cadc2828edb1cd4bc4f9eb587100631af06ea2108f4c9ed56df4755701e76"
    path = DATA / filename
    def verified(p):
        return p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == sha
    if verified(path):
        return path
    print(f"[vision] downloading {filename}", flush=True)
    urls = (f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{filename}",
            f"https://huggingface.co/Uminosachi/FastSAM/resolve/main/{filename}")
    for url in urls:
        with tempfile.NamedTemporaryFile(dir=DATA, delete=False) as tmp:
            partial = Path(tmp.name)
        try:
            with urlopen(url, timeout=40) as src, partial.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            if verified(partial):
                partial.replace(path)
                return path
        except OSError as exc:
            print(f"[vision] {exc}", flush=True)
        finally:
            partial.unlink(missing_ok=True)
    raise SystemExit(f"Could not download verified weights. Place {filename} in {DATA} (SHA256 {sha}).")


def jpeg_url(image):
    ok, jpg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 86])
    if not ok:
        raise ValueError("无法编码相机帧")
    return "data:image/jpeg;base64," + base64.b64encode(jpg).decode("ascii")


def capture_profile(raw, source):
    """Validate capture optics; the receiver assigns the source from its own HTTP route."""
    if raw is None:
        return None
    if not isinstance(raw, dict) or source not in ("local", "phone"):
        raise ValueError("无法识别相机采集配置")
    if "version" in raw and (type(raw["version"]) is not int or raw["version"] != 1):
        raise ValueError("无法识别相机采集配置")
    camera_key = raw.get("camera_key")
    if not isinstance(camera_key, str) or not re.fullmatch(r"[0-9a-f]{32}", camera_key):
        raise ValueError("相机配置标识无效")
    profile = dict(version=1, source=source, camera_key=camera_key)
    for key in ("native_size", "video_size"):
        size = raw.get(key)
        if not isinstance(size, list) or len(size) != 2 or not all(type(n) is int and 0 < n <= 65535 for n in size):
            raise ValueError("相机配置中的视频尺寸无效")
        profile[key] = list(size)
    zoom = raw.get("zoom")
    if zoom is not None and (type(zoom) not in (float, int) or not 0 < zoom <= 1000 or not math.isfinite(zoom)):
        raise ValueError("相机配置中的变焦倍率无效")
    profile["zoom"] = None if zoom is None else round(zoom, 4)
    for key, choices in (("facing", ("user", "environment", "left", "right")),
                         ("resize_mode", ("none", "crop-and-scale"))):
        value = raw.get(key)
        if value is not None and value not in choices:
            raise ValueError("相机配置中的镜头方向或裁切模式无效")
        profile[key] = value
    return profile


class Instrument:
    def __init__(self, predict):
        self.predict = predict
        self.guard = threading.RLock()
        self.busy = threading.Lock()
        self.start()

    def start(self, profile=None):
        with self.guard:
            self.session = secrets.token_hex(10)
            self.profile = profile
            self.frame = self.plane = self.frozen = self.held = None
            self.pending = None
            self.seq = 0
            self.tracker = CubeTracker()
            self.media_clock = None
            self.timings = deque(maxlen=60)
            return self.session

    def ticket(self, session):
        with self.guard:
            if session != self.session:
                raise ValueError("相机连接已更新，请重新开始")
            self.seq += 1
            self.pending = self.seq, time.monotonic()
            return self.seq

    def receive(self, raw, session, seq, media_s=None):
        received = time.monotonic()
        if media_s is not None and (not np.isfinite(media_s) or media_s < 0):
            raise ValueError("无效的视频时间戳")
        if not self.busy.acquire(blocking=False):
            raise ValueError("上一帧还在处理")
        try:
            with self.guard:
                if session != self.session or not self.pending or seq != self.pending[0]:
                    raise ValueError("帧已过期")
                _, stamp = self.pending
                self.pending = None
                plane = self.plane
                if media_s is not None and self.media_clock is True and media_s == self.tracker.stamp:
                    return "duplicate"
            if time.monotonic() - stamp > FRESH_S:
                return "expired"
            with Image.open(io.BytesIO(raw)) as header:
                if header.format != "JPEG" or max(header.size) > 2560 or min(header.size) < 64:
                    raise ValueError("需要有效的 JPEG 视频帧，最长边不超过 2560 像素")
                normalized = header.getexif().get(274, 1) == 1
            image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            if image is None:
                raise ValueError("相机帧无法解码")
            valid = plane is not None and plane.matches(image)
            infer_start = time.monotonic()
            cubes = plane.cubes(self.predict(image, plane)) if valid else []
            infer_ms = 1000 * (time.monotonic()-infer_start)
            # Canvas JPEGs already have the delivered orientation. Preserve their original
            # edge samples when the 3-D viewer or chessboard solver reads this legacy relay.
            preview = (
                "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")
                if normalized else jpeg_url(image)
            )
            result = dict(seq=seq, image=preview, cubes=cubes, valid=valid, predictions=[],
                          calibrated=plane is not None, points=plane.points.tolist() if plane else [],
                          matrix=plane.to_mm.tolist() if valid else None)
            with self.guard:
                if session != self.session or self.plane is not plane:
                    return
                now = time.monotonic()
                if self.media_clock != (media_s is not None) or (self.frame and stamp-self.frame[0] > FRESH_S):
                    self.tracker.reset()
                self.media_clock = media_s is not None
                if not valid or now-stamp > FRESH_S:
                    self.tracker.reset()
                else:
                    missing = self.tracker.update(cubes, media_s if media_s is not None else stamp)
                    for track in missing:
                        xy = [track["fused_x_mm"], track["fused_y_mm"]]
                        if plane.in_workspace(xy):
                            track["center_px"] = project([xy], plane.to_px)[0].tolist()
                            result["predictions"].append(track)
                    for cube in cubes:
                        xy = [cube["fused_x_mm"], cube["fused_y_mm"]]
                        cube["fused_px"] = project([xy], plane.to_px)[0].tolist()
                if valid:
                    self.timings.append((1000*(received-stamp), infer_ms, 1000*(now-stamp)))
                    result["timing_p95_ms"] = np.percentile(self.timings, 95, axis=0).round().astype(int).tolist()
                    result["timing_n"] = len(self.timings)
                self.frame = stamp, image, result
        finally:
            self.busy.release()

    def state(self, after):
        with self.guard:
            now = time.monotonic()
            held = None if not self.held else dict(self.held[1], age_s=round(now-self.held[0], 2))
            if not self.frame:
                return dict(session=self.session, seq=0, fresh=False, held=held, profile=self.profile)
            stamp, _, frame = self.frame
            result = dict(frame)
            if after == str(result["seq"]):
                result.pop("image")
            fresh = now-stamp <= FRESH_S
            if not fresh:
                result.update(cubes=[], predictions=[], matrix=None)
            else:
                result["predictions"] = [dict(p, ttl_ms=round(1000*(PREDICT_S-p["missing_s"]-(now-stamp))))
                                         for p in result["predictions"] if p["missing_s"] + now-stamp < PREDICT_S]
            return dict(result, session=self.session, fresh=fresh, age_ms=round(1000*(now-stamp)),
                        held=held, profile=self.profile)

    def freeze(self):
        with self.guard:
            if not self.frame or time.monotonic()-self.frame[0] > FRESH_S:
                raise ValueError("等待新的相机帧")
            self.frozen = (secrets.token_hex(8), self.frame[1])
            return dict(id=self.frozen[0], image=self.frame[2]["image"])

    def calibrate(self, obj):
        with self.guard:
            if not self.frozen or self.frozen[0] != obj.get("id"):
                raise ValueError("请重新冻结画面")
            self.plane = Plane(self.frozen[1], obj["points"], obj.get("width"), obj.get("height"),
                               cube_reference=obj.get("reference") == "cube")
            self.frozen = self.frame = self.held = None
            self.tracker.reset()
            self.timings.clear()

    def hold(self, obj):
        with self.guard:
            if not self.frame or obj.get("session") != self.session or time.monotonic()-self.frame[0] > FRESH_S:
                raise ValueError("测量已过期")
            stamp, _, current = self.frame
            i = int(obj["index"])
            if current["seq"] != obj.get("seq"):
                # A new frame can arrive between rendering the button and clicking it.
                if not 0 < current["seq"] - int(obj["seq"]) <= 4:
                    raise ValueError("目标画面已更新，请重试")
                matches = [j for j, c in enumerate(current["cubes"]) if
                           c.get("track_id") == obj.get("track_id", c.get("track_id")) and
                           np.hypot(c["x_mm"] - float(obj["x_mm"]), c["y_mm"] - float(obj["y_mm"])) <= 1.5]
                if len(matches) != 1:
                    raise ValueError("目标已经移动或被遮挡，请重新观测")
                i = matches[0]
            if not 0 <= i < len(current["cubes"]) or not current["cubes"][i]["stable"]:
                raise ValueError("等待同一位置连续稳定，再锁定")
            cube = current["cubes"][i]
            self.held = stamp, dict(track_id=cube["track_id"], x_mm=cube["fused_x_mm"], y_mm=cube["fused_y_mm"],
                                    raw_x_mm=cube["x_mm"], raw_y_mm=cube["y_mm"], yaw_mod90_deg=cube["yaw_mod90_deg"])


def handler(mode, device, token, ip, port, ca, phone_status=None):
    class Handler(BaseHTTPRequestHandler):
        # Safari can reuse its TLS connection across frames.
        protocol_version = "HTTP/1.1"
        disable_nagle_algorithm = True

        def log_message(self, *_):
            pass

        def send(self, content, typ="application/json", code=200):
            if typ == "application/json":
                content = json.dumps(content, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", typ)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(content)))
            if code >= 400:
                self.close_connection = True
                self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(content)

        def dispatch(self, post=False):
            path = urlsplit(self.path).path
            if mode == "setup":
                if not post and path == "/camera-root.cer":
                    return self.send(ca.read_bytes(), "application/x-x509-ca-cert")
                if not post and path == "/":
                    body = (HERE / "vision_setup.html").read_text("utf-8").replace("CAMERA_URL", f"https://{ip}:{port}/")
                    return self.send(body.encode(), "text/html; charset=utf-8")
            elif not post and (path == "/" or (mode == "desk" and path == "/camera")):
                page = "phone" if path == "/camera" else mode
                body = (HERE / f"vision_{page}.html").read_text("utf-8")
                body = body.replace("SETUP_URL", f"http://{ip}:{port-1}/")
                if page == "desk":
                    body = body.replace("WIFI_VISIBILITY", "" if ca else "hidden")
                    body = body.replace("ROOT_SHA256", hashlib.sha256(ca.read_bytes()).hexdigest() if ca else "")
                return self.send(body.encode(), "text/html; charset=utf-8")
            elif not secrets.compare_digest(self.headers.get("X-Pair", "").encode(), token.encode()):
                return self.send({"error": "配对码不正确"}, code=403)
            elif mode == "desk" and not post and path == "/state":
                return self.send(device.state(self.headers.get("X-After")))
            elif post and mode in ("desk", "phone"):
                local_camera = mode == "desk" and path.startswith("/camera/")
                if local_camera:
                    path = path.removeprefix("/camera")
                n = int(self.headers.get("Content-Length", "0"))
                if not 0 <= n <= (2_500_000 if path == "/frame" else 4096):
                    raise ValueError("数据过大")
                self.connection.settimeout(6)
                data = self.rfile.read(n)
                obj = {} if not data or path == "/frame" else json.loads(data)
                if mode == "phone" or local_camera:
                    sid = self.headers.get("X-Session")
                    if path == "/start":
                        if not isinstance(obj, dict):
                            raise ValueError("相机启动参数需要 JSON 对象")
                        profile = capture_profile(obj.get("profile"), "local" if local_camera else "phone")
                        return self.send({"session": device.start() if profile is None else device.start(profile)})
                    if path == "/ticket":
                        return self.send({"seq": device.ticket(sid)})
                    if path == "/frame":
                        media = self.headers.get("X-Media-Time")
                        dropped = device.receive(data, sid, int(self.headers.get("X-Seq", "-1")),
                                                 float(media) if media is not None else None)
                        reply = {"ok": dropped is None}
                        if dropped:
                            reply["dropped"] = dropped
                        if self.headers.get("X-Next-Ticket") == "1":
                            reply["next_seq"] = device.ticket(sid)
                        if phone_status and self.headers.get("X-Workcell-Status") == "1":
                            reply.update(phone_status(sid))
                        return self.send(reply)
                elif path == "/freeze":
                    return self.send(device.freeze())
                elif path == "/calibrate":
                    device.calibrate(obj)
                    return self.send({"ok": True})
                elif path == "/hold":
                    device.hold(obj)
                    return self.send({"ok": True})
                elif path == "/clear":
                    with device.guard:
                        device.held = None
                    return self.send({"ok": True})
            return self.send({"error": "not found"}, code=404)

        def do_GET(self):
            self.dispatch()

        def do_POST(self):
            try:
                self.dispatch(True)
            except (ValueError, TypeError, KeyError, OSError, cv2.error) as e:
                self.send({"error": str(e)}, code=400)
            except Exception as e:
                print(f"[vision] {type(e).__name__}: {e}", flush=True)
                self.send({"error": "模型处理失败，请查看电脑终端"}, code=500)
    return Handler


def main(argv=None):
    ap = argparse.ArgumentParser(description="iPhone LAN camera / 29 mm cube workbench")
    ap.add_argument("--ip", help="computer Wi-Fi IPv4 address")
    ap.add_argument("--local-camera", action="store_true", help="only serve localhost, for USB / virtual cameras")
    ap.add_argument("--port", type=int, default=8843, help="phone HTTPS port (setup: port-1, workstation: port+1)")
    ap.add_argument("--weights", type=Path, help="optional local FastSAM segmentation weights")
    ap.add_argument("--device", default="cpu", help="cpu or a CUDA device such as 0")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)
    ip = "127.0.0.1" if args.local_camera else args.ip
    if not ip:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("192.0.2.1", 9))
            ip = sock.getsockname()[0]
    ip = str(ipaddress.IPv4Address(ip))
    if not 1025 <= args.port <= 65534:
        ap.error("port must be between 1025 and 65534")
    from ultralytics import FastSAM
    import torch

    DATA.mkdir(parents=True, exist_ok=True)
    if args.weights is not None and not args.weights.is_file():
        ap.error("weights file does not exist")
    model = FastSAM(str(args.weights or default_weights()))
    torch.set_num_threads(min(4, os.cpu_count() or 1))

    def predict(image, plane):
        # Crop to the calibrated workspace so a small cube gets more pixels at the model input.
        x, y, right, bottom = plane.crop
        result = model.predict(image[y:bottom, x:right], imgsz=640, device=args.device,
                               retina_masks=True, conf=0.25, verbose=False)[0]
        return [] if result.masks is None else [p + [x, y] for p in result.masks.xy]

    device = Instrument(predict)
    token = f"{secrets.randbelow(10**8):08d}"
    context, ca = (None, None) if args.local_camera else camera_tls(DATA / "tls" / ip, ip)
    bindings = [("desk", "127.0.0.1", args.port+1)]
    if not args.local_camera:
        bindings.extend([("setup", ip, args.port-1), ("phone", ip, args.port)])
    with ExitStack() as stack:
        for mode, host, port in bindings:
            server = stack.enter_context(ThreadingHTTPServer((host, port), handler(mode, device, token, ip, args.port, ca)))
            if mode == "phone":
                server.socket = context.wrap_socket(server.socket, server_side=True)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            stack.callback(server.shutdown)
        desk = f"http://127.0.0.1:{args.port+1}/#{token}"
        print(f"PC: {desk}\nPair: {token}", flush=True)
        if ca:
            print(f"iPhone setup: http://{ip}:{args.port-1}/\niPhone camera: https://{ip}:{args.port}/"
                  f"\nRoot SHA256: {hashlib.sha256(ca.read_bytes()).hexdigest()}", flush=True)
        if not args.no_browser:
            webbrowser.open(desk)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
