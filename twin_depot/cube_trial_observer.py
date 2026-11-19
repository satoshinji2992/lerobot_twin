"""Full-frame YOLO cube detection and 29 mm pose estimation in camera coordinates."""

import logging
from pathlib import Path

import cv2
import numpy as np

from .cube_pose_tracking import RigidCubeTracker
from .cube_pose import FACE, HALF, _color_outline, _overlap
from .cuboid_pnp import CENTRED, SYMMETRIES, TEMPLATES

COLORS = ("red", "green", "blue", "yellow")
INPUT = 384
POSE_WEIGHTS = Path(__file__).resolve().parents[1] / "assets/cube-yolo11n-pose.onnx"
MODEL_NAMES = {"detect": "YOLO11n 检测", "pose": "YOLO11n 八角点", "vit": "YOLO11n + ViT 八角点"}


def box_distance(a, b):
    """Centre distance in box widths; incompatible colours and sizes cannot match."""
    if a["color"] != b["color"]:
        return float("inf")
    x, y, w, h = a["box"]
    bx, by, bw, bh = b["box"]
    if not 0.35 < w * h / (bw * bh) < 2.9:
        return float("inf")
    return np.hypot(x + w / 2 - bx - bw / 2, y + h / 2 - by - bh / 2) / max(bw, bh)


def visible_face_poses(value, mask, origin, k, dist):
    """Use a uniformly shaded square face as an independent planar PnP observation."""
    values = value[mask > 0]
    if len(values) < 150 or values.std() < 5:
        return
    level = cv2.threshold(values, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    for region in (value > level, value <= level):
        contours = cv2.findContours(np.uint8(region & (mask > 0)), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)[0]
        if not contours:
            continue
        face = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(face)
        if not 0.19 * len(values) < area < 0.95 * len(values):
            continue
        quad = cv2.approxPolyDP(cv2.convexHull(face), 0.026 * cv2.arcLength(face, True), True)
        if len(quad) != 4 or abs(cv2.contourArea(quad) - area) > 0.09 * area:
            continue
        pts = quad.reshape(4, 2) + origin
        if cv2.contourArea(quad, oriented=True) < 0:
            pts = pts[::-1]
        pts = np.ascontiguousarray(pts, dtype=float)
        ok, rotations, translations, errors = cv2.solvePnPGeneric(FACE, pts, k, dist, flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            continue
        for rv, tv, err in zip(rotations, translations, errors, strict=True):
            r = cv2.Rodrigues(rv)[0]
            if err.item() > max(1.0, 0.025 * np.sqrt(area)) or r[:, 2] @ tv.ravel() >= 0:
                continue
            centre = np.eye(4)
            centre[:3, :3] = r
            centre[:3, 3] = tv.ravel() - HALF * r[:, 2]
            yield centre


def rim_poses(contour, origin, k, dist):
    """A six-edge cube rim has two geometrically distinct corner assignments."""
    hull = cv2.convexHull(contour)
    if cv2.contourArea(contour) < 0.88 * cv2.contourArea(hull):
        return
    perimeter = cv2.arcLength(hull, True)
    for fraction in (0.01, 0.016, 0.024, 0.034):
        polygon = cv2.approxPolyDP(hull, max(0.8, fraction * perimeter), True)
        if len(polygon) < 6:
            return
        if len(polygon) > 6:
            continue
        corners = np.ascontiguousarray(polygon.reshape(-1, 2) + origin, dtype=float)
        for template in TEMPLATES[6]:
            ok, rv, tv = cv2.solvePnP(template, corners, k, dist, flags=cv2.SOLVEPNP_SQPNP)
            if not ok or tv[2, 0] < 58:
                continue
            rim = cv2.projectPoints(template, rv, tv, k, dist)[0].reshape(-1, 2)
            if np.linalg.norm(rim - corners, axis=1).mean() > max(1.5, 0.012 * perimeter):
                continue
            candidate = np.eye(4)
            candidate[:3, :3] = cv2.Rodrigues(rv)[0]
            candidate[:3, 3] = tv.ravel()
            yield candidate
        return


_NEIGHBOURS = [np.flatnonzero(np.sum(v != CENTRED, axis=1) == 1) for v in CENTRED]


def ridge_support(centre, pixels, distances):
    """Test the three ridges meeting at the nearest vertex against interior image edges."""
    vertices = CENTRED @ centre[:3, :3].T + centre[:3, 3]
    nearest = np.argmin(np.sum(vertices * vertices, axis=1))
    start = pixels[nearest]
    delta = pixels[_NEIGHBOURS[nearest]] - start
    samples = start + delta[:, None] * np.linspace(.18, .82, 7)[:, None]
    x, y = np.moveaxis(np.rint(samples).astype(int), -1, 0)
    x = np.clip(x, 0, distances.shape[1] - 1)
    y = np.clip(y, 0, distances.shape[0] - 1)
    scores = np.mean(np.maximum(0, 1 - distances[y, x] / 3), axis=1)
    scores[np.linalg.norm(delta, axis=1) < 6] = 0
    return float(np.sort(scores)[-2:].mean())


def refine_outline(hsv, color, pose, pixels, k, dist):
    """Accept an independent complete silhouette only when it improves the image fit."""
    outline = _color_outline(hsv, pixels, color)
    if outline is None:
        return None
    origin, mask = outline
    _, initial_iou = _overlap(mask, pixels - origin)
    if initial_iou < 0.50:
        return None
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    contour = max(contours, key=cv2.contourArea)
    x, y = origin
    value = hsv[y:y + mask.shape[0], x:x + mask.shape[1], 2]
    candidates = list(visible_face_poses(value, mask, origin, k, dist))
    candidates.extend(rim_poses(contour, origin, k, dist))
    if not candidates:
        return None
    # Exclude the silhouette from the brightness edges so an inside-out cube
    # cannot score a visible face boundary by tracing its exterior instead.
    edge = cv2.Canny(value, 20, 50) & cv2.erode(mask, np.ones((5, 5), np.uint8))
    distances = cv2.distanceTransform(np.uint8(edge == 0), cv2.DIST_L2, 3)
    old_ridges = ridge_support(pose, pixels - origin, distances)
    best = None
    best_score = initial_iou + 0.12 * old_ridges + 0.025
    for centre in candidates:
        if abs(centre[2, 3] - pose[2, 3]) > 0.15 * pose[2, 3]:
            continue
        rotation = cv2.Rodrigues(centre[:3, :3])[0]
        refined = cv2.projectPoints(CENTRED, rotation, centre[:3, 3], k, dist)[0].reshape(-1, 2)
        _, iou = _overlap(mask, refined - origin)
        if iou < max(0.81, initial_iou - 0.015):
            continue
        ridges = ridge_support(centre, refined - origin, distances)
        score = iou + 0.12 * ridges
        if score > best_score and ridges + 0.20 >= old_ridges:
            best_score = score
            best = centre
    if best is None:
        return None
    # Keep equivalent axis labels closest to the input orientation.
    best[:3, :3] = min((best[:3, :3] @ s for s in SYMMETRIES),
                        key=lambda r: np.linalg.norm(r - pose[:3, :3]))
    rotation = cv2.Rodrigues(best[:3, :3])[0]
    refined = cv2.projectPoints(CENTRED, rotation, best[:3, 3], k, dist)[0].reshape(-1, 2)
    return best, refined


def load_trial(name, device, completion=None):
    if name in ("pose", "pose-raw"):
        model = TrialObserver(POSE_WEIGHTS, device, verify_pose=name == "pose")
        if name == "pose-raw":
            model.name = "YOLO11n-Pose + PnP（原始输出）"
        return model
    model = "pose" if name == "rescue" else name
    path = Path(__file__).resolve().parents[1] / "data/vision/model-trial/runs" / model / "weights/best.onnx"
    if not path.is_file():
        raise FileNotFoundError(f"Missing exported model {path}; run evaluate_cube_models.py {name}")
    yolo = TrialObserver(path, device)
    return RescueObserver(yolo, device, completion) if name == "rescue" else yolo


class RescueObserver:
    """Use a current-frame full-image pose only near a recently verified cube."""

    name = "方块角点 CNN + YOLO-Pose 补检"

    def __init__(self, yolo, device, completion=None):
        from .cube_completion import Completion

        self.yolo = yolo
        self.device = yolo.device
        self.completion = completion or Completion(device)
        self.reset()

    def reset(self):
        from .cube_observer import FastObserver

        self.primary = FastObserver(self.completion)
        self.anchors = []
        self.last_at = -float("inf")
        self.estimates = []

    def observe(self, image, predict, k, dist, at, *, reference=False):
        if not 0 < at - self.last_at <= 0.25:
            self.reset()
        self.last_at = at
        _, partial, stage = self.primary.observe(image, predict, k, dist, at, reference=reference)
        self.estimates = list(self.primary.estimates)
        anchors = [(b, t) for b, t in self.anchors if at - t <= 0.25]
        missing = []
        for b, t in anchors:
            if not any(box_distance(r, b) < 0.78 for r in self.estimates):
                missing.append((b, t))
        self.anchors = [(r, at) for r in self.estimates] + missing
        if missing:
            self.yolo.observe(image, None, k, dist, at)
            for r in self.yolo.estimates:
                if any(box_distance(r, e) < 0.78 for e in self.estimates):
                    continue
                matches = [
                    i
                    for i, (b, _) in enumerate(missing)
                    if box_distance(r, b) < 0.78
                    and abs(r["camera_xyz"][2] - b["camera_xyz"][2]) < 0.30 * b["camera_xyz"][2]
                ]
                if len(matches) == 1:
                    missing.pop(matches[0])
                    self.estimates.append({**r, "rescued": True})
                    partial = [p for p in partial if box_distance(p, r) >= 0.78]
                    stage = "YOLO-Pose 补检"
        return [], partial, stage


class TrialObserver:
    def __init__(self, weights, device="cuda", *, verify_pose=False):
        self.verify_pose = verify_pose
        self.name = (
            "YOLO11n-Pose + PnP" if verify_pose else MODEL_NAMES.get(Path(weights).parent.parent.name, "YOLO")
        )
        self.session = None
        self.net = None
        self.device = "CPU · OpenCV"
        probe = np.zeros((1, 3, INPUT, INPUT), np.float32)
        if device != "cpu":
            try:
                self._load_cuda(weights)
                output = self.forward(probe)
            except Exception as exc:
                if device != "auto":
                    raise
                self.session = None
                self.device = "CPU · OpenCV"
                logging.getLogger(__name__).warning("YOLO CUDA unavailable, using OpenCV CPU: %s", exc)
        if self.session is None:
            self.net = cv2.dnn.readNetFromONNX(str(weights))
            output = self.forward(probe)
        self.box_only = output.shape[1] == 8
        self.reset()

    def _load_cuda(self, weights):
        import onnxruntime as ort

        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("Install ONNX Runtime GPU to run YOLO on CUDA")
        ort.preload_dlls()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        providers = [("CUDAExecutionProvider", {"device_id": 0, "cudnn_conv_algo_search": "HEURISTIC"})]
        session = ort.InferenceSession(str(weights), sess_options=options, providers=providers)
        session.disable_fallback()
        if session.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError("YOLO could not activate CUDA")
        self.input = session.get_inputs()[0].name
        self.session = session
        self.device = "CUDA:0 · ONNX Runtime"

    def forward(self, blob):
        if self.session is not None:
            return self.session.run(None, {self.input: blob})[0]
        self.net.setInput(blob)
        return self.net.forward()

    def reset(self):
        self.estimates = []
        self.boxes = []
        self.confirmed = []
        self.pose_tracker = RigidCubeTracker()
        self.pose_tracks = []
        self.last_at = -float("inf")
        self.frame_size = None

    @staticmethod
    def color_support(image, pixels, color):
        """Fraction of the visible projected hull carrying the predicted pigment."""
        h, w = image.shape[:2]
        hull = cv2.convexHull(np.float32(pixels))
        x, y, bw, bh = cv2.boundingRect(hull)
        x0, y0, x1, y1 = max(0, x), max(0, y), min(w, x + bw), min(h, y + bh)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        roi = image[y0:y1, x0:x1]
        inside = np.zeros(roi.shape[:2], np.uint8)
        cv2.fillConvexPoly(inside, np.rint(hull - (x0, y0)).astype(np.int32), 1)
        count = cv2.countNonZero(inside)
        if count < 9:
            return 0.0
        hue, saturation, value = cv2.split(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV))
        if color == "red":
            # Deep reds may appear on either side of zero. Skin often shares
            # the orange-red end of this interval at lower saturation.
            chroma = ((hue <= 7) | (hue >= 169)) & (saturation >= 135)
            chroma &= (hue <= 2) | (hue >= 169) | (saturation >= 185)
        else:
            lo, hi = {"green": (36, 88), "blue": (94, 136), "yellow": (16, 35)}[color]
            chroma = (hue >= lo) & (hue <= hi) & (saturation >= 65)
        # An almost black display can have a strong hue but very little chroma.
        valid = chroma & (inside != 0) & (value >= 24) & (saturation.astype(np.uint16) * value >= 20 * 255)
        if not valid.any():
            return 0.0
        # OpenCV hue wraps at 180. Select the most populated 30-degree arc so
        # cube faces may differ in brightness without admitting a printed pattern.
        hist = np.bincount(hue[valid], minlength=180)
        width = 15
        start = np.convolve(np.r_[hist, hist[: width - 1]], np.ones(width, int), "valid").argmax()
        end = (start + width - 1) % 180
        arc = (hue >= start) & (hue <= end) if end >= start else (hue >= start) | (hue <= end)
        solid = (valid & arc).astype(np.uint8)
        _, _, stats, _ = cv2.connectedComponentsWithStats(solid, connectivity=8)
        largest = stats[1:, cv2.CC_STAT_AREA].max()
        return min(cv2.countNonZero(solid), 2 * int(largest)) / count

    def verify(self, candidates, at):
        # Only confident, colour-supported frames can start or renew a track.
        # A weak pose still needs pixels in this frame and a unique recent match.
        anchors = [(p, t) for p, t in self.confirmed if 0 < at - t <= 0.35] if at > self.last_at else []
        self.last_at = at
        strong, weak = [], []
        for p in candidates:
            if (
                p["confidence"] >= 0.55
                and p["color_support"] >= 0.42
                or (p["confidence"] >= 0.40 and p["color_support"] >= 0.75)
            ):
                strong.append(p)
            elif p["color_support"] >= 0.10:
                weak.append(p)
        remaining = [(p, t) for p, t in anchors if not any(box_distance(s, p) < 0.8 for s in strong)]
        self.confirmed = [(p, at) for p in strong] + remaining
        accepted = list(strong)
        for p in weak:
            matches = [
                i
                for i, (anchor, _) in enumerate(remaining)
                if box_distance(p, anchor) < 0.8
                and abs(p["camera_xyz"][2] - anchor["camera_xyz"][2]) < 0.30 * anchor["camera_xyz"][2]
            ]
            if len(matches) == 1:
                remaining.pop(matches[0])
                accepted.append(p)
        return accepted

    def observe(self, image, predict, k, dist, at, *, reference=False, refine_edges=True):
        if reference:
            raise ValueError("全图 YOLO 模式使用相机坐标")
        h, w = image.shape[:2]
        if self.verify_pose and self.frame_size != (w, h):
            self.reset()
            self.frame_size = (w, h)
        scale = INPUT / max(w, h)
        nw, nh = round(w * scale), round(h * scale)
        left, top = (INPUT - nw) // 2, (INPUT - nh) // 2
        canvas = np.full((INPUT, INPUT, 3), 114, np.uint8)
        canvas[top : top + nh, left : left + nw] = cv2.resize(image, (nw, nh))
        blob = cv2.dnn.blobFromImage(canvas, 1 / 255, swapRB=True)
        output = self.forward(blob)[0].T
        ids = output[:, 4:8].argmax(axis=1)
        conf = output[np.arange(len(ids)), 4 + ids]
        rows = output[conf >= 0.25]
        ids, conf = ids[conf >= 0.25], conf[conf >= 0.25]
        self.boxes, self.estimates = [], []
        partial = []
        if not len(rows):
            if self.verify_pose:
                self.verify([], at)
                self.pose_tracks = self.pose_tracker.update([], at, k, dist)
            return [], partial, "全图检测"
        xywh = rows[:, :4].copy()
        xywh[:, :2] -= xywh[:, 2:] / 2 + [left, top]
        xywh /= np.array([nw / w, nh / h, nw / w, nh / h])
        indices = cv2.dnn.NMSBoxesBatched(xywh.tolist(), conf.tolist(), ids.tolist(), 0.25, 0.5)
        hsv = None
        for i in np.asarray(indices).ravel():
            x, y, bw, bh = xywh[i]
            lo = np.clip([x, y], [0, 0], [w - 1, h - 1])
            hi = np.clip([x + bw, y + bh], [0, 0], [w, h])
            if min(hi - lo) < 3:
                continue
            label = {
                "color": COLORS[ids[i]],
                "box": np.rint(np.r_[lo, hi - lo]).astype(int).tolist(),
                "confidence": float(conf[i]),
            }
            if not self.verify_pose:
                self.boxes.append(label)
            if rows.shape[1] == 8:
                partial.append(label)
                continue
            corners = (rows[i, 8:].reshape(8, 2) - [left, top]) / [nw / w, nh / h]
            pose = self.fit(corners, k, dist, max(bw, bh))
            if pose is None:
                if not self.verify_pose:
                    partial.append(label)
                continue
            matrix, pixels = pose
            if self.verify_pose:
                label["color_support"] = self.color_support(image, pixels, label["color"])
                if refine_edges and label["color_support"] >= 0.55:
                    if hsv is None:
                        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                    refined = refine_outline(hsv, label["color"], matrix, pixels, k, dist)
                    if refined is not None:
                        matrix, pixels = refined
                        label["outline_refined"] = True
            self.estimates.append(
                {
                    **label,
                    "camera_pose": matrix.tolist(),
                    "camera_xyz": matrix[:3, 3].tolist(),
                    "pixels": pixels.tolist(),
                    "strong": True,
                }
            )
        if self.verify_pose:
            self.estimates = self.verify(self.estimates, at)
            self.pose_tracks = self.pose_tracker.update(self.estimates, at, k, dist, image)
            self.boxes = self.estimates.copy()
            return [], partial, "YOLO 八角点 · PnP · 颜色与时序核验"
        return [], partial, "全图角点 + PnP" if rows.shape[1] > 8 else "全图检测框"

    @staticmethod
    def fit(corners, k, dist, side):
        if not np.isfinite(corners).all():
            return None
        try:
            ok, rv, tv = cv2.solvePnP(CENTRED, corners, k, dist, flags=cv2.SOLVEPNP_SQPNP)
        except cv2.error:
            return None
        if not ok or not np.isfinite(tv).all() or tv[2, 0] < 58:
            return None
        pixels = cv2.projectPoints(CENTRED, rv, tv, k, dist)[0].reshape(8, 2)
        if np.sqrt(np.mean(np.sum((corners - pixels) ** 2, axis=1))) > 0.10 * side:
            return None
        matrix = np.eye(4)
        matrix[:3, :3] = cv2.Rodrigues(rv)[0]
        matrix[:3, 3] = tv.ravel()
        return matrix, pixels
