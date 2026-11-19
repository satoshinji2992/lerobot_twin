"""Live MuJoCo reconstruction of an external phone and independently moving 29 mm cubes."""

import argparse
import base64
import hashlib
import io
import ipaddress
import json
import os
import secrets
import socket
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
import xml.etree.ElementTree as ET
from collections import deque
from contextlib import ExitStack
from http.server import ThreadingHTTPServer
from pathlib import Path

import cv2
import mujoco
import numpy as np
from PIL import Image

from .cube_completion import EDGES, WEIGHTS as COMPLETION_WEIGHTS, Completion
from .cube_observer import CubeObserver, FastObserver
from .cube_pose import COLORS, HALF, camera_matrix
from .cube_pose_tracking import DEPTH_HOLD_S, HOLD_S
from .cube_trial_observer import load_trial
from .screen_calibration import ScreenCalibration
from .vision import DATA, FRESH_S, default_weights, handler, jpeg_url
from .vision_tls import camera_tls
from .workcell_registration import preview_registration
from .workcell_source import LatestFrames
from .workcell_tracking import Tracker


def transform(value):
    t = np.asarray(value, dtype=float)
    if (
        t.shape != (4, 4)
        or not np.isfinite(t).all()
        or not np.allclose(t[3], [0, 0, 0, 1])
        or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-4)
        or not np.isclose(np.linalg.det(t[:3, :3]), 1, atol=1e-4)
    ):
        raise ValueError("base_from_reference must be a rigid 4×4 transform (translation in mm)")
    return t


class Scene:
    def __init__(self, registered, frame_mode="workcell"):
        self.registered = registered
        self.frame_mode = frame_mode
        if registered:
            xml = Path(__file__).resolve().parents[1] / "assets/so101/so101_new_calib.xml"
            root = ET.parse(xml).getroot()
            root.find("compiler").set("meshdir", str(xml.parent / "assets"))
        else:
            root = ET.fromstring("<mujoco><worldbody/></mujoco>")
        world = root.find("worldbody")
        world.append(ET.fromstring('<light pos="0 0 1.2" dir="0 0 -1"/>'))
        if frame_mode == "workcell":
            world.append(
                ET.fromstring('<geom type="plane" pos="0 0 -.002" size="1 1 .01" rgba=".16 .20 .21 1"/>')
            )
        for end, color in ((".07 0 0", ".9 .24 .20"), ("0 .07 0", ".23 .77 .37"), ("0 0 .07", ".24 .49 .95")):
            ET.SubElement(
                world,
                "geom",
                type="capsule",
                fromto="0 0 0 " + end,
                size=".0015",
                rgba=color + " 1",
                contype="0",
                conaffinity="0",
            )
        for i in range(13):
            body = ET.SubElement(world, "body", name=f"tracked{i}", mocap="true", pos="0 0 -3")
            if i == 0:
                # Lens is the origin; the handset shape is illustrative. CV camera looks along +Z.
                body.append(
                    ET.fromstring(
                        '<geom type="box" size=".036 .073 .004" pos=".022 .05 -.005" rgba=".40 .48 .60 1"/>'
                    )
                )
                body.append(ET.fromstring('<geom type="cylinder" size=".006 .002" rgba=".10 .85 .92 1"/>'))
            else:
                ET.SubElement(body, "geom", name=f"block{i}", type="box", size=".0145 .0145 .0145")
        self.m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        self.d = mujoco.MjData(self.m)
        self.renderer = mujoco.Renderer(self.m, height=360, width=540)
        self.view = mujoco.MjvCamera()
        self.view.azimuth = 132
        self.view.elevation = -19
        self.last_view = None
        self.ids = [self.m.body(f"tracked{i}").mocapid[0] for i in range(13)]

    def draw(self, phone, cubes, view="overview", estimates=(), sources=()):
        self.d.mocap_pos[self.ids] = [0, 0, -3]
        visible = [*cubes, *estimates][:12]
        poses = [phone] + [p for _, p in visible]
        for i, t in enumerate(poses):
            idx = self.ids[i]
            self.d.mocap_pos[idx] = t[:3, 3] / 1000
            mujoco.mju_mat2Quat(self.d.mocap_quat[idx], np.ascontiguousarray(t[:3, :3]).ravel())
            if i:
                inferred_index = i - len(cubes) - 1
                blind = 0 <= inferred_index < len(sources) and sources[inferred_index] in (
                    "predicted",
                    "held",
                )
                self.m.geom(f"block{i}").rgba[:] = [
                    *COLORS[visible[i - 1][0]],
                    1 if i <= len(cubes) else 0.23 if blind else 0.55,
                ]
        pos = self.d.mocap_pos[self.ids[: len(poses)]]
        if view != "overview" and len(pos) > 1:
            pos = pos[1:]
        elif self.registered:
            pos = np.vstack((pos, [0, 0, 0], [0.1, 0, 0.30]))
        self.view.azimuth, self.view.elevation = {
            "overview": (132, -19),
            "detail": (132, -32),
            "side": (90, -6),
            "top": (90, -89.5),
        }[view]
        # Keep the virtual camera from zooming as an occluded cube drops in and out.
        if view != self.last_view or not any(source != "observed" for source in sources):
            self.view.lookat[:] = (pos.max(axis=0) + pos.min(axis=0)) / 2
            self.view.distance = max(
                0.36 if view == "overview" else 0.14, 1.85 * np.linalg.norm(np.ptp(pos, axis=0))
            )
        self.last_view = view
        mujoco.mj_forward(self.m, self.d)
        self.renderer.update_scene(self.d, camera=self.view)
        return jpeg_url(self.renderer.render()[:, :, ::-1])


def reconstruct(detections, base_from_reference):
    ref = next(pose for name, _, pose, _ in detections if name == "red")
    # Reference origin is the centre of the bottom face, +Z toward the observed top face.
    to_top = np.eye(4)
    to_top[2, 3] = 2 * HALF
    world_camera = base_from_reference @ to_top @ np.linalg.inv(ref)
    to_center = np.eye(4)
    to_center[2, 3] = -HALF
    return world_camera, [(name, world_camera @ pose @ to_center) for name, _, pose, _ in detections]


class Workcell:
    def __init__(self, source, config_path, connection, device):
        self.source = source
        self.arm_viewer = None
        self.config_path = config_path
        self.config = json.loads(config_path.read_text("utf-8-sig")) if config_path.exists() else {}
        if not isinstance(self.config, dict):
            raise ValueError("工位配置必须是 JSON 对象")
        self.base = transform(self.config.get("base_from_reference", np.eye(4)))
        self.lock = threading.RLock()
        self.frame_ready = threading.Condition(self.lock)
        self.board_lock = threading.Lock()
        self.batch = self.new_batch()
        self.batch_profile = None
        self.connection = connection
        self.view = "overview"
        self.frame_mode = "workcell"
        self.pipeline = "precision"
        self.device = device
        self.completion = None
        self.fast_observer = None
        self.inference_device = None
        self.revision = 0
        self.seq = 0
        self.session = ""
        self.has_reference = False
        self.quarter = None
        self.axis_epoch = secrets.token_hex(6)
        self.snap = {"status": "等待手机开始传输"}

    def new_batch(self):
        return ScreenCalibration(self.config_path.parent / "screen-calibration" / secrets.token_hex(6))

    def reset_reference(self):
        self.has_reference = False
        self.quarter = None
        self.axis_epoch = secrets.token_hex(6)
        self.revision += 1

    def select_session(self, session):
        if self.session != session:
            self.session = session
            self.reset_reference()
            self.snap = {"status": "相机已连接，等待新会话画面", "rows": [], "history": []}

    def save(self, key, value):
        # Read-modify-replace under the same lock used for live coordinate changes.
        with self.lock:
            path = self.config_path
            config = json.loads(path.read_text("utf-8-sig")) if path.exists() else dict(self.config)
            config[key] = value
            if key == "intrinsics":
                config.pop("landscape_intrinsics", None)
            base = transform(config.get("base_from_reference", np.eye(4)))
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf8", dir=path.parent, delete=False) as tmp:
                temp_path = Path(tmp.name)
                try:
                    json.dump(config, tmp, ensure_ascii=False, indent=2, allow_nan=False)
                except Exception:
                    tmp.close()
                    temp_path.unlink(missing_ok=True)
                    raise
            try:
                temp_path.replace(path)
            finally:
                temp_path.unlink(missing_ok=True)
            self.config, self.base = config, base
            self.reset_reference()
            self.snap = {"status": "配置已应用，正在重新定位"}

    def orient(self, obj):
        with self.lock:
            if isinstance(self.source, LatestFrames):
                self.select_session(self.source.read().session)
            if (
                "base_from_reference" not in self.config
                or not self.has_reference
                or obj.get("epoch") != self.axis_epoch
                or time.perf_counter() - self.snap.get("_stamp", 0) > FRESH_S
            ):
                raise ValueError("参考已变化，请在最新画面中重新核对 +X 边")
            quarter = obj.get("quarter")
            if type(quarter) is not int or quarter not in range(4):
                raise ValueError("请选择参考块的一条边")
            self.quarter = quarter
            self.revision += 1
            self.snap = {"status": "方向已确认，正在更新基座坐标"}

    def board(self, action):
        with self.board_lock:
            try:
                if action == "new":
                    self.batch = self.new_batch()
                    self.batch_profile = None
                    message = "新采集组已就绪"
                elif action == "sample":
                    frame = self.source.read()
                    if not frame.fresh or not frame.jpeg:
                        raise ValueError("等待 iPhone 新画面，请保持手机相机页传输")
                    message = self.batch.offer(frame.jpeg, frame.session)
                    if self.batch.session == frame.session:
                        self.batch_profile = frame.profile
                elif action in ("migrate-cw", "migrate-ccw"):
                    frame = self.source.read()
                    if not frame.fresh or frame.session != self.batch.session or frame.profile != self.batch_profile:
                        raise ValueError("请在同一相机会话中采集并验证横屏图像")
                    result = self.batch.migrate(self.config.get("intrinsics"), frame.profile, action == "migrate-cw")
                    self.save("landscape_intrinsics", result)
                    message = f"横屏内参已验证并应用，验证 RMS {result['validation']['heldout_rms_px']:.3f} px"
                elif action == "solve":
                    result = self.batch.solve()
                    if self.batch_profile is not None:
                        result["capture_profile"] = self.batch_profile
                    self.save("intrinsics", result)
                    message = f"内参已验证并应用，验证 RMS {result['validation']['heldout_rms_px']:.3f} px"
                else:
                    message = "已载入内参" if self.config.get("intrinsics") else "尚未标定，可开始采集"
            except (ValueError, OSError, cv2.error) as exc:
                message = str(exc)
            intr = self.config.get("intrinsics") or {}
            return dict(
                count=len(self.batch.corners),
                message=message,
                **self.batch.report(),
                applied_rms_px=intr.get("rms_px"),
                landscape_validation=(self.config.get("landscape_intrinsics") or {}).get("validation"),
            )

    def state(self, after, wait=False):
        with self.lock:
            now = time.perf_counter()
            if isinstance(self.source, LatestFrames):
                self.select_session(self.source.read().session)
            if wait and after and str(self.snap.get("seq", "")) == after:
                self.frame_ready.wait_for(lambda: str(self.snap.get("seq", "")) != after, timeout=0.14)
                now = time.perf_counter()
            snap = dict(self.snap)
            stamp = snap.pop("_stamp", None)
            if stamp is not None:
                snap["age_ms"] = round((now - stamp) * 1000)
                stale_hold = any(
                    p["source"] != "observed"
                    and (
                        p["missing_ms"] + snap["age_ms"] > HOLD_S * 1000
                        or p["depth_age_ms"] + snap["age_ms"] > DEPTH_HOLD_S * 1000
                    )
                    for p in snap.get("pose_tracks", ())
                )
                if now - stamp > FRESH_S or (stale_hold and now - stamp > 0.12):
                    self.idle("相机画面已过期，等待新的观测")
                    snap = dict(self.snap)
            if snap.get("seq") is not None and str(snap["seq"]) == after:
                snap.pop("image", None)
                snap.pop("sim", None)
            snap.setdefault("view", self.view)
            intr = self.config.get(snap.pop("_intrinsics_key", "intrinsics")) or {}
            snap.setdefault("calibrated", bool(intr))
            if not intr:
                calibration_state = "missing"
            elif "seq" not in snap:
                calibration_state = "saved"
            elif not snap["calibrated"]:
                calibration_state = "mismatch"
            elif intr.get("capture_profile") is None:
                calibration_state = "lens_unverified"
            else:
                calibration_state = "matched"
            return dict(
                snap,
                calibration={
                    "state": calibration_state,
                    "size": intr.get("image_size"),
                    "views": intr.get("views"),
                    "validation_px": (intr.get("validation") or {}).get("heldout_rms_px"),
                },
                fresh="seq" in snap,
                camera_rx=self.source.statistics() if isinstance(self.source, LatestFrames) else None,
                session=self.session,
                revision=self.revision,
                axis_epoch=self.axis_epoch,
                base_configured="base_from_reference" in self.config,
                registered=self.quarter is not None,
                frame_mode=self.frame_mode,
                pipeline=self.pipeline,
                box_only=self.pipeline == "fast" and getattr(self.fast_observer, "box_only", False),
                verified_pose=self.pipeline == "fast" and getattr(self.fast_observer, "verify_pose", False),
                pose_tracking=self.pipeline == "fast"
                and getattr(self.fast_observer, "verify_pose", False)
                and hasattr(self.fast_observer, "pose_tracker"),
                connection=self.connection,
                device=(self.fast_observer or self.completion).device
                if self.pipeline == "fast" and (self.fast_observer or self.completion)
                else self.device,
                model_name=(self.fast_observer.name if self.fast_observer else "方块角点 CNN")
                if self.pipeline == "fast"
                else getattr(self, "model_name", "FastSAM"),
            )

    def phone_status(self, session):
        state = self.state(None)
        if session != state["session"]:
            return {}
        return {"calibration": state["calibration"]["state"], "device": state["device"]}

    def idle(self, message):
        with self.lock:
            if self.has_reference:
                self.reset_reference()
            self.snap = {"status": message, "rows": [], "history": []}


def annotate(image, detections, partial=(), reference=True):
    for candidate in partial:
        x, y, w, h = candidate["box"]
        cv2.rectangle(image, (x, y), (x + w, y + h), (30, 190, 245), 2)
        cv2.putText(image, "2D?", (x, max(14, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (30, 190, 245), 2)
    for name, points, _, _ in detections:
        corners = np.rint(points).astype(np.int32)
        cv2.polylines(image, [corners], True, (85, 232, 170), 2, cv2.LINE_AA)
        if name == "red" and reference:
            for letter, point in zip("ABCD", corners, strict=True):
                xy = tuple(point + [3, -4])
                cv2.putText(image, letter, xy, cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(
                    image, letter, xy, cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA
                )


def aligned_base(cell):
    if cell.quarter is None:
        return np.eye(4)
    # B->C is -Y in the observed frame. Choosing it as physical +X rotates
    # coordinates +90 degrees before applying the taught base transform.
    twist = np.eye(4)
    twist[:2, :2] = np.linalg.matrix_power(np.array([[0, -1], [1, 0]]), cell.quarter)
    return cell.base @ twist


def continuous_reference(previous, detected):
    if not previous or not detected:
        return False
    points, pose = next(v for name, v in previous if name == "red")
    _, new_points, new_pose, _ = next(d for d in detected if d[0] == "red")
    side = np.linalg.norm(points - np.roll(points, 1, axis=0), axis=1).mean()
    change = pose[:3, :3].T @ new_pose[:3, :3]
    return np.linalg.norm(points - new_points, axis=1).mean() < 0.70 * side and np.trace(
        change
    ) > 1 + 2 * np.cos(np.radians(45))


def run_worker(cell, predict, stop):
    scene = None
    previous = []
    after = None
    sid = None
    last_stamp = 0.0
    revision = -1
    tracker = Tracker()
    completion = cell.completion
    observer = CubeObserver(completion)
    fast_observer = cell.fast_observer or FastObserver(completion)
    estimate_tracker = Tracker()
    times = deque(maxlen=60)
    processing = deque(maxlen=60)
    result_ages = deque(maxlen=60)
    arrivals = deque(maxlen=30)
    try:
        while not stop.is_set():
            try:
                processing_start = time.perf_counter()
                frame = cell.source.read(after)
                with cell.lock:
                    if isinstance(cell.source, LatestFrames) and cell.source.read().session != frame.session:
                        continue
                    if sid != frame.session:
                        sid = frame.session
                        cell.select_session(sid)
                        after = None
                        arrivals.clear()
                        times.clear()
                        processing.clear()
                        result_ages.clear()
                    if not frame.fresh:
                        if frame.seq:
                            after = (sid, frame.seq)
                        arrivals.clear()
                        cell.idle("等待新的相机画面")
                    config_revision = cell.revision
                    landscape = cell.config.get("landscape_intrinsics")
                    intr_key = ("landscape_intrinsics" if landscape and landscape["capture_profile"] == frame.profile
                                else "intrinsics")
                    intr = cell.config.get(intr_key)
                    view = cell.view
                    mode = cell.frame_mode
                    pipeline = cell.pipeline
                if not frame.fresh:
                    cell.source.wait_next((sid, frame.seq))
                    continue
                key = sid, frame.seq
                if after == key or not frame.jpeg:
                    cell.source.wait_next(key)
                    continue
                after = key
                if revision != config_revision:
                    tracker = Tracker()
                    estimate_tracker = Tracker()
                    if pipeline == "fast" and cell.fast_observer is None and cell.inference_device:
                        cell.fast_observer = load_trial("pose", cell.inference_device)
                    fast_observer = cell.fast_observer or FastObserver(completion)
                    if hasattr(fast_observer, "reset"):
                        fast_observer.reset()
                    # Orientation confirmation only changes world coordinates;
                    # optical tracking can be reused across that operation.
                    if (
                        pipeline == "precision"
                        and completion is None
                        and cell.inference_device
                        and COMPLETION_WEIGHTS.is_file()
                    ):
                        completion = Completion(cell.inference_device)
                        cell.completion = completion
                    if not cell.has_reference:
                        observer = CubeObserver(completion)
                if not cell.has_reference or frame.stamp - last_stamp > FRESH_S:
                    previous = []
                with Image.open(io.BytesIO(frame.jpeg)) as header:
                    source_w, source_h = header.size
                reduced = pipeline == "fast" and max(source_w, source_h) >= 1280
                decode = cv2.IMREAD_REDUCED_COLOR_2 if reduced else cv2.IMREAD_COLOR
                image = cv2.imdecode(
                    np.frombuffer(frame.jpeg, np.uint8), decode | cv2.IMREAD_IGNORE_ORIENTATION
                )
                if image is None:
                    raise ValueError("相机帧无法解码")
                t = time.perf_counter()
                calibration_note = ""
                calibrated = False
                partial = []
                stage = ""
                visible = []
                estimates = []
                rigid_mode = pipeline == "fast" and getattr(fast_observer, "verify_pose", False)
                pose_tracks = []
                try:
                    current_profile = frame.profile
                    if (current_profile and intr and intr.get("capture_profile")
                            and "screen_angle" not in intr["capture_profile"]):
                        current_profile = {k: v for k, v in current_profile.items() if k != "screen_angle"}
                    if intr and "capture_profile" in intr and intr["capture_profile"] != current_profile:
                        raise ValueError(
                            "相机、倍率或采集方式与标定不匹配。请切回原设置或为当前画面重新标定。"
                        )
                    k, dist = camera_matrix((source_h, source_w), intr)
                    if reduced:
                        ratio = np.array(image.shape[1::-1]) / [source_w, source_h]
                        k = k.copy()
                        k[:2] *= ratio[:, None]
                        k[:2, 2] += (ratio - 1) / 2
                    calibrated = intr is not None
                    if calibrated and intr_key == "landscape_intrinsics":
                        calibration_note = "使用已验证的相机内参"
                    if calibrated and intr.get("capture_profile") is None:
                        calibration_note = "使用实测内参 · 需核对镜头与采集设置"
                    active = fast_observer if pipeline == "fast" else observer
                    visible, partial, stage = active.observe(
                        image,
                        predict,
                        k,
                        dist,
                        frame.stamp,
                        reference=mode == "workcell",
                        **({"source_jpeg": frame.jpeg} if pipeline == "precision" else {}),
                    )
                    # The API rescales boxes and adds track data. Keep these
                    # changes out of the observer's inference-space history.
                    estimates = [dict(item) for item in active.estimates]
                    if rigid_mode:
                        pose_tracks = [dict(item) for item in active.pose_tracks]
                    partial = [dict(item) for item in partial]
                    red_count = sum(n == "red" for n, _, _, _ in visible)
                    detections = visible
                    message = ""
                    if mode == "workcell" and red_count != 1:
                        detections = []
                        message = (
                            "检测到多个红块，工位坐标暂停"
                            if red_count > 1
                            else "等待固定红块定位工位。手持单块时可切换到相机坐标。"
                        )
                    elif not visible:
                        message = (
                            "显示遮挡补全估计，等待实测角点"
                            if estimates
                            else "仅有二维候选，等待可测量的方块面"
                            if partial
                            else "等待彩色方块入镜"
                        )
                        if getattr(active, "box_only", False):
                            message = "全图检测模式显示二维框" if partial else "等待检测到方块"
                        elif getattr(active, "verify_pose", False):
                            message = "YOLO-Pose 已定位方块" if estimates else "等待颜色与角点均吻合的方块"
                except (ValueError, cv2.error) as exc:
                    detections, message = [], str(exc)
                    if rigid_mode:
                        fast_observer.reset()
                    if intr and not calibrated:
                        calibration_note = "内参未适用 · 已暂停三维定位"
                ms = 1000 * (time.perf_counter() - t)
                times.append(ms)
                if not arrivals or frame.seq > arrivals[-1][0] and frame.stamp > arrivals[-1][1]:
                    arrivals.append((frame.seq, frame.stamp))
                fps = frame.source_fps
                if fps is None and len(arrivals) > 1:
                    fps = (arrivals[-1][0] - arrivals[0][0]) / (arrivals[-1][1] - arrivals[0][1])
                with cell.lock:
                    if config_revision != cell.revision:
                        continue
                    if (
                        mode == "workcell"
                        and cell.has_reference
                        and not continuous_reference(previous, detections)
                    ):
                        cell.reset_reference()
                        tracker = Tracker()
                    revision = cell.revision
                    base = aligned_base(cell)
                    registered = cell.quarter is not None
                last_stamp = frame.stamp
                previous = [(name, (points, pose)) for name, points, pose, _ in detections]
                snap = {
                    "status": message or "跟踪中",
                    "rows": [],
                    "history": [],
                    "_stamp": frame.stamp,
                    "source_seq": frame.seq,
                    "view": view,
                    "calibrated": calibrated,
                    "_intrinsics_key": intr_key,
                    "calibration_note": calibration_note,
                    "ms": round(ms),
                    "p95_ms": round(float(np.percentile(times, 95))),
                    "source_fps": fps,
                    "partial": partial,
                    "completions": estimates,
                    "stage": stage,
                }
                annotate(image, visible, partial, reference=mode == "workcell")
                display_poses = pose_tracks if rigid_mode else estimates
                if rigid_mode:
                    snap["pose_tracks"] = pose_tracks
                    if pose_tracks and not estimates:
                        snap["status"] = "遮挡中，显示短时轨迹估计"
                for estimate in display_poses:
                    p = np.int32(np.rint(estimate["pixels"]))
                    source = estimate.get("source", "observed")
                    color = (235, 140, 220) if source == "observed" else (50, 186, 255)
                    for a, b in EDGES:
                        if source in ("predicted", "held"):
                            midpoint = (p[a] + p[b]) // 2
                            cv2.line(image, tuple(p[a]), tuple(midpoint), color, 1, cv2.LINE_AA)
                        else:
                            cv2.line(image, tuple(p[a]), tuple(p[b]), color, 1, cv2.LINE_AA)
                for item in estimates:
                    if "image_segments" in item:
                        for a, b in np.rint(item["image_segments"]).astype(int):
                            cv2.line(image, tuple(a), tuple(b), (216, 225, 32), 1, cv2.LINE_AA)
                        for point in np.rint(item["image_vertices"]).astype(int):
                            cv2.circle(image, tuple(point), 2, (216, 225, 32), -1, cv2.LINE_AA)
                if reduced and (estimates or partial or pose_tracks):
                    # API points always refer to the original calibrated phone frame.
                    for item in [*estimates, *partial]:
                        item["box"] = np.rint(np.array(item["box"]) / np.tile(ratio, 2)).astype(int).tolist()
                    for item in [*estimates, *pose_tracks]:
                        item["pixels"] = ((np.array(item["pixels"]) + 0.5) / ratio - 0.5).round(1).tolist()
                    for item in estimates:
                        if "image_segments" in item:
                            for field in ("image_segments", "image_vertices"):
                                item[field] = ((np.asarray(item[field]) + 0.5) / ratio - 0.5).round(1).tolist()
                cam = np.eye(4) if mode == "camera" else None
                fused = []
                if detections:
                    if mode == "camera":
                        cam = np.eye(4)
                        offset = np.eye(4)
                        offset[2, 3] = -HALF
                        cubes = [(n, p @ offset) for n, _, p, _ in detections]
                    else:
                        cam, cubes = reconstruct(detections, base)
                    raw = [
                        {
                            "color": name,
                            "xyz": p[:3, 3].tolist(),
                            "pose": p.tolist(),
                            "error_px": round(err, 2),
                            "pnp_points": len(pts),
                        }
                        for (name, pts, _, err), (_, p) in zip(detections, cubes, strict=True)
                    ]
                    rows, history = tracker.update(raw, frame.stamp)
                    fused = []
                    for row in rows:
                        pose = np.array(row["pose"])
                        pose[:3, 3] = row["xyz"]
                        row["pose"] = pose.tolist()
                        fused.append((row["color"], pose))
                    snap.update(
                        rows=rows, history=history, camera_pose=cam.tolist() if mode == "workcell" else None
                    )
                elif mode == "camera":
                    _, snap["history"] = tracker.update([], frame.stamp)
                inferred = (
                    []
                    if rigid_mode
                    else [{"color": r["color"], "xyz": r["camera_xyz"], "estimate": r} for r in estimates]
                )
                tracked, estimated_history = estimate_tracker.update(inferred, frame.stamp)
                if cam is not None and (fused or display_poses):
                    inferred_poses = [(r["color"], cam @ np.array(r["camera_pose"])) for r in display_poses]
                    try:
                        if scene is None or scene.registered != registered or scene.frame_mode != mode:
                            if scene:
                                scene.renderer.close()
                                scene = None
                            scene = Scene(registered, mode)
                        snap["sim"] = scene.draw(
                            cam, fused, view, inferred_poses, [r["source"] for r in pose_tracks]
                        )
                    except (OSError, ValueError, RuntimeError, mujoco.FatalError) as exc:
                        snap["status"] = f"定位已更新，三维渲染失败：{exc}"
                preview_scale = min(1.0, 800 / max(image.shape[:2])) if pipeline == "fast" else 1.0
                if preview_scale < 1:
                    image = cv2.resize(
                        image, None, fx=preview_scale, fy=preview_scale, interpolation=cv2.INTER_AREA
                    )
                snap["image"] = jpeg_url(image)
                result_at = time.perf_counter()
                if rigid_mode:
                    fast_observer.pose_tracker.compensate(pose_tracks, result_at)
                    for p in pose_tracks:
                        if p["measurement_index"] is not None:
                            estimates[p["measurement_index"]].update(
                                track_id=p["track_id"],
                                fused_camera_xyz=p["camera_xyz"],
                                lead_camera_xyz=p.get("lead_camera_xyz"),
                                lead_ms=p.get("lead_ms"),
                                source=p["source"],
                            )
                tracker.compensate(snap["rows"], snap["history"], result_at)
                estimate_tracker.compensate(tracked, estimated_history, result_at)
                for item in tracked:
                    item["estimate"].update(
                        track_id=item["track_id"],
                        fused_camera_xyz=item["xyz"],
                        lead_camera_xyz=item.get("lead_xyz"),
                        lead_ms=item.get("lead_ms"),
                    )
                snap["processing_ms"] = round(1000 * (time.perf_counter() - processing_start), 1)
                snap["result_age_ms"] = round(1000 * (time.perf_counter() - frame.stamp), 1)
                processing.append(snap["processing_ms"])
                result_ages.append(snap["result_age_ms"])
                snap["processing_p95_ms"] = round(float(np.percentile(processing, 95)), 1)
                snap["result_age_p95_ms"] = round(float(np.percentile(result_ages, 95)), 1)
                with cell.lock:
                    if isinstance(cell.source, LatestFrames):
                        cell.select_session(cell.source.read().session)
                    if revision == cell.revision and frame.fresh:
                        cell.seq += 1
                        cell.snap = dict(snap, seq=cell.seq)
                        cell.has_reference = mode == "workcell" and bool(detections)
                        cell.frame_ready.notify_all()
                    elif revision == cell.revision:
                        cell.idle("处理结束时画面已过期，等待新帧")
            except Exception as exc:
                # A broken frame, disconnected source or failed inference must not kill acquisition.
                cell.idle(f"等待恢复：{type(exc).__name__} · {exc}")
                previous = []
                tracker = Tracker()
                observer = CubeObserver(completion)
                stop.wait(0.25)
    finally:
        if scene:
            scene.renderer.close()


def desktop_handler(cell, token, phone_pair):
    parent = handler("desk", cell.source, phone_pair, "127.0.0.1", 0, None, cell.phone_status)
    connection_id = secrets.token_hex(8)
    pages = {"/": "robot_view.html", "/board": "screen_board.html", "/registration": "registration.html"}

    class Handler(parent):
        def authorized(self):
            return secrets.compare_digest(self.headers.get("X-Pair", "").encode(), token.encode())

        def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler entry point
            if self.path == "/":
                from .webui import console_url
                from .workcell_link import endpoint

                try:
                    active, _ = endpoint()
                except (OSError, ValueError):
                    active = None
                if active == f"http://127.0.0.1:{self.server.server_port}" and (console := console_url()):
                    self.send_response(302)
                    self.send_header("Location", console + "#vision")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    return
            if self.path == "/camera":
                return super().do_GET()
            if self.path in pages:
                page = Path(__file__).with_name(pages[self.path]).read_text("utf-8")
                page = page.replace(
                    "CAMERA_MODE", "实测内参" if cell.config.get("intrinsics") else "估计内参"
                )
                page = page.replace("FRAME_NAME", "参考块底面坐标")
                return self.send(page.encode(), "text/html; charset=utf-8")
            if self.path not in ("/state", "/sim-state"):
                return self.send({"error": "页面不存在"}, code=404)
            if not self.authorized():
                return self.send({"error": "桌面配对码不正确"}, code=403)
            if self.path == "/sim-state":
                state = cell.state(None)
                return self.send(
                    {
                        k: state[k]
                        for k in ("fresh", "registered", "frame_mode", "camera_pose", "rows", "completions", "pose_tracks")
                        if k in state
                    }
                )
            state = cell.state(self.headers.get("X-After"), self.headers.get("X-Wait") == "1")
            state["connection_id"] = connection_id
            if self.headers.get("X-Known-Connection") == connection_id:
                state.pop("connection", None)
            if self.headers.get("X-Scene-Image") == "skip":
                state.pop("sim", None)
            return self.send(state)

        def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler entry point
            if self.path in ("/camera/start", "/camera/ticket", "/camera/frame"):
                return super().do_POST()
            if not self.authorized():
                return self.send({"error": "桌面配对码不正确"}, code=403)
            try:
                if self.path == "/arm-viewer":
                    with cell.lock:
                        if cell.arm_viewer is None or cell.arm_viewer.poll() is not None:
                            cell.arm_viewer = subprocess.Popen(
                                [sys.executable, "-m", "twin_depot.arm_viewer", str(self.server.server_port)],
                                cwd=Path(__file__).resolve().parents[1],
                                env={**os.environ, "TWIN_SIM_PAIR": token},
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                            )
                    return self.send({"started": True})
                if self.path in ("/board/status", "/board/new", "/board/sample", "/board/solve",
                                 "/board/migrate-cw", "/board/migrate-ccw"):
                    return self.send(cell.board(self.path.rsplit("/", 1)[1]))
                if self.path in (
                    "/view",
                    "/frame-mode",
                    "/registration/preview",
                    "/registration/apply",
                    "/registration/orient",
                ):
                    length = int(self.headers.get("Content-Length", 0))
                    if not 0 < length <= 8192:
                        raise ValueError("配准数据长度无效")
                    self.connection.settimeout(4)
                    obj = json.loads(self.rfile.read(length))
                    if not isinstance(obj, dict):
                        raise ValueError("配准输入需要 JSON 对象")
                    if self.path == "/frame-mode":
                        mode = obj.get("mode")
                        pipeline = obj.get("pipeline", "fast" if mode == "camera" else "precision")
                        if mode not in ("camera", "workcell"):
                            raise ValueError("请选择相机或工位坐标")
                        if pipeline not in ("fast", "precision") or (
                            mode == "workcell" and pipeline == "fast"
                        ):
                            raise ValueError("工位坐标请选择精测模式")
                        with cell.lock:
                            if (cell.frame_mode, cell.pipeline) != (mode, pipeline):
                                cell.frame_mode = mode
                                cell.pipeline = pipeline
                                cell.reset_reference()
                                cell.snap = {"status": "坐标系已切换，等待新帧"}
                        return self.send({"mode": mode, "pipeline": pipeline})
                    if self.path == "/view":
                        if obj.get("view") not in ("overview", "detail", "side", "top"):
                            raise ValueError("未知的三维视角")
                        with cell.lock:
                            cell.view = obj["view"]
                        return self.send({"ok": True, "view": obj["view"]})
                    if self.path.endswith("/orient"):
                        cell.orient(obj)
                        return self.send({"ok": True})
                    result = preview_registration(obj)
                    if self.path.endswith("/apply"):
                        if not result["can_apply"]:
                            raise ValueError("独立检查点超出容差，请重新测量")
                        cell.save("base_from_reference", result["base_from_reference"])
                    return self.send(result)
            except (ValueError, OSError, TypeError) as exc:
                return self.send({"error": str(exc)}, code=400)
            return self.send({"error": "接口不存在"}, code=404)

    return Handler


def setup_qr(url):
    if not url or not hasattr(cv2, "QRCodeEncoder_create"):
        return None
    qr = cv2.QRCodeEncoder_create().encode(url)
    qr = cv2.copyMakeBorder(qr, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255)
    qr = cv2.resize(qr, None, fx=5, fy=5, interpolation=cv2.INTER_NEAREST)
    return "data:image/png;base64," + base64.b64encode(cv2.imencode(".png", qr)[1]).decode()


def build_predictor(device, weights, variant="auto"):
    import torch
    from ultralytics import FastSAM

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    cv2.setNumThreads(2)
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if device != "cpu":
            try:
                from torchvision.ops import nms

                nms(torch.zeros((1, 4), device=device), torch.ones(1, device=device), 0.5)
            except (RuntimeError, NotImplementedError) as exc:
                print(f"[workcell] CUDA unavailable, using CPU: {exc}", flush=True)
                device = "cpu"
    name = device if device == "cpu" else torch.cuda.get_device_name(torch.device(device))
    if variant == "auto":
        variant = "x" if device != "cpu" and (DATA / "FastSAM-x.pt").is_file() else "s"
    weights = weights or default_weights(variant)
    net = FastSAM(str(weights))

    def predict(crop):
        masks = net.predict(
            crop, imgsz=512, device=device, half=device != "cpu", conf=0.25, retina_masks=False, verbose=False
        )[0].masks
        return [] if masks is None else masks.xy

    predict.model_name = Path(weights).stem
    return predict, name


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8874, help="localhost workstation port")
    parser.add_argument("--config", type=Path, default=DATA / "workcell.json")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "cuda:0"), default="auto")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--model", choices=("auto", "s", "x"), default="auto")
    parser.add_argument(
        "--trial-model",
        choices=("detect", "pose", "pose-raw", "vit", "rescue"),
        help="fast camera model override (default: verified YOLO11n-Pose + PnP)",
    )
    parser.add_argument("--frame", choices=("camera", "workcell"), default="camera")
    parser.add_argument("--pipeline", choices=("auto", "fast", "precision"), default="auto")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--ip", help="computer Wi-Fi IPv4 address")
    parser.add_argument(
        "--camera-port", type=int, default=8843, help="phone HTTPS port; setup uses the preceding port"
    )
    parser.add_argument("--local-camera", action="store_true", help="USB / virtual camera on localhost only")
    args = parser.parse_args(argv)
    DATA.mkdir(parents=True, exist_ok=True)
    if not 0 < args.port < 65536:
        parser.error("invalid desktop port")
    token = os.environ.get("TWIN_WORKCELL_TOKEN") or secrets.token_hex(6)
    if not 1025 <= args.camera_port <= 65535:
        parser.error("camera-port must be between 1025 and 65535")
    port = args.camera_port
    ip = args.ip
    if not args.local_camera and not ip:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("192.0.2.1", 9))
            ip = probe.getsockname()[0]
    ip = str(ipaddress.IPv4Address(ip or "127.0.0.1"))
    phone_pair = os.environ.get("TWIN_WORKCELL_PHONE_PAIR") or f"{secrets.randbelow(10**8):08d}"
    if len(phone_pair) != 8 or not phone_pair.isascii() or not phone_pair.isdigit():
        parser.error("TWIN_WORKCELL_PHONE_PAIR requires eight digits")
    source = LatestFrames()
    connection = {"pair": phone_pair, "local_url": f"http://127.0.0.1:{args.port}/camera#{phone_pair}"}
    if not args.local_camera:
        connection.update(setup_url=f"http://{ip}:{port - 1}/", camera_url=f"https://{ip}:{port}/")
    if connection.get("camera_url"):
        connection["join_url"] = connection["camera_url"] + "#" + phone_pair
        connection["setup_qr"] = setup_qr(connection["join_url"])
    connection["transport"] = "direct"
    with ExitStack() as stack:
        # Bind all requested sockets before loading the model or announcing a usable service.
        cell = Workcell(source, args.config, connection, "启动中")
        cell.frame_mode = args.frame
        cell.pipeline = (
            ("fast" if args.frame == "camera" else "precision") if args.pipeline == "auto" else args.pipeline
        )
        if args.frame == "workcell" and cell.pipeline == "fast":
            parser.error("use --frame camera for the fast pipeline")
        if args.trial_model and cell.pipeline != "fast":
            parser.error("--trial-model requires the fast camera mode")
        desktop = stack.enter_context(
            ThreadingHTTPServer(("127.0.0.1", args.port), desktop_handler(cell, token, phone_pair))
        )
        servers = []
        if not args.local_camera:
            context, ca = camera_tls(DATA / "tls" / ip, ip)
            connection["root_sha256"] = hashlib.sha256(ca.read_bytes()).hexdigest()
            for mode, listen in (("setup", port - 1), ("phone", port)):
                srv = stack.enter_context(
                    ThreadingHTTPServer(
                        (ip, listen), handler(mode, source, phone_pair, ip, port, ca, cell.phone_status)
                    )
                )
                if mode == "phone":
                    srv.socket = context.wrap_socket(srv.socket, server_side=True)
                servers.append(srv)
        cv2.setNumThreads(2)
        cell.inference_device = args.device
        if cell.pipeline == "fast":
            cell.fast_observer = load_trial(args.trial_model or "pose", args.device)
        elif COMPLETION_WEIGHTS.is_file():
            cell.completion = Completion(args.device)
        loaded = None
        unavailable = None

        def predict(image):
            nonlocal loaded, unavailable
            if unavailable is not None:
                raise ValueError(unavailable)
            if loaded is None:
                try:
                    loaded, cell.device = build_predictor(args.device, args.weights, args.model)
                except ImportError:
                    unavailable = "精测模式需要安装 requirements-vision.txt 中的依赖，可切回手持轻量模式。"
                    raise ValueError(unavailable) from None
                except SystemExit as exc:
                    unavailable = str(exc)
                    raise ValueError(unavailable) from None
                cell.model_name = getattr(loaded, "model_name", "FastSAM")
            return loaded(image)

        if cell.pipeline == "precision":
            loaded, cell.device = build_predictor(args.device, args.weights, args.model)
            cell.model_name = getattr(loaded, "model_name", "FastSAM")
        else:
            cell.device = cell.fast_observer.device
        for srv in servers:
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            stack.callback(srv.shutdown)
        done = threading.Event()
        worker = threading.Thread(target=run_worker, args=(cell, predict, done), daemon=True)
        worker.start()
        desk = f"http://127.0.0.1:{args.port}/#{token}"
        from .workcell_link import publish

        publish(args.port, token)
        print(desk, flush=True)
        print(f"Pair: {phone_pair}", flush=True)
        if connection.get("setup_url"):
            print("iPhone setup:", connection["setup_url"], flush=True)
        if not args.no_browser:
            from .webui import announce_console, console_url, make_server
            from twin import cached_env_python

            console = console_url()
            if not console:
                panel_server, panel, console = make_server(
                    Path(__file__).resolve().parents[1] / "twin.py", cached_env_python(), 8765)
                stack.enter_context(panel_server)
                stack.enter_context(announce_console(panel_server))
                threading.Thread(target=panel_server.serve_forever, daemon=True).start()
                stack.callback(panel.shutdown_jobs)
                stack.callback(panel_server.shutdown)
            webbrowser.open(console + "#vision")
        try:
            desktop.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            done.set()
            worker.join(timeout=4)
