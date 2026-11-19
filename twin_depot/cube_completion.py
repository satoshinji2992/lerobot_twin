"""Amodal image-corner estimator for partially visible, unmarked 29 mm cubes."""

import logging
from pathlib import Path

import cv2
import numpy as np

from .cuboid_pnp import CENTRED, HALF, silhouette_pose

SIZE = 40
WEIGHTS = Path(__file__).resolve().parents[1] / "assets" / "cube-completion.onnx"
EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))


def patch_input(value, mask, box):
    x, y, w, h = box
    side = 1.65 * max(w, h)
    center = np.array([x + w / 2, y + h / 2])
    origin = center - side / 2
    scale = SIZE / side
    affine = np.float32([[scale, 0, -origin[0] * scale], [0, scale, -origin[1] * scale]])
    m = cv2.warpAffine(mask, affine, (SIZE, SIZE), flags=cv2.INTER_NEAREST)
    v = cv2.warpAffine(value, affine, (SIZE, SIZE))
    foreground = m > 0
    if foreground.sum() < 60:
        return None
    bright = max(20, np.percentile(v[foreground], 90))
    intensity = np.uint8(np.minimum(255, 220 * v.astype(float) / bright)) * foreground
    return np.stack((foreground.astype(np.uint8) * 255, intensity)), origin, side


class Completion:
    def __init__(self, device="auto"):
        if device not in ("auto", "cuda", "cuda:0", "cpu"):
            raise ValueError(f"Unknown inference device {device}")
        self.device = "CPU · OpenCV"
        self.session = None
        self.candidates = []
        if device != "cpu":
            try:
                import onnxruntime as ort

                if "CUDAExecutionProvider" not in ort.get_available_providers():
                    raise RuntimeError("CUDAExecutionProvider unavailable")
                ort.preload_dlls()
                options = ort.SessionOptions()
                options.intra_op_num_threads = 1
                options.inter_op_num_threads = 1
                options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
                provider = ("CUDAExecutionProvider", {"device_id": 0, "cudnn_conv_algo_search": "HEURISTIC"})
                session = ort.InferenceSession(str(WEIGHTS), sess_options=options, providers=[provider])
                session.disable_fallback()
                if session.get_providers()[0] != "CUDAExecutionProvider":
                    raise RuntimeError("ONNX Runtime did not activate CUDA")
                session.run(None, {"crop": np.zeros((1, 2, SIZE, SIZE), np.float32)})
                self.session = session
                self.device = "CUDA:0 · ONNX Runtime"
            except Exception as exc:
                # ONNX Runtime's native status exceptions inherit directly from Exception.
                if device != "auto":
                    raise RuntimeError(
                        f"轻量模型 CUDA 初始化失败，请核对 requirements-vision-gpu.txt 和显卡驱动。{exc}"
                    ) from exc
                logging.getLogger(__name__).warning("CUDA unavailable, using CPU for cube corners: %s", exc)
        if self.session is None:
            self.net = cv2.dnn.readNetFromONNX(str(WEIGHTS))

    def forward(self, tensor):
        if self.session is not None:
            return self.session.run(None, {"crop": tensor})[0]
        self.net.setInput(tensor)
        return self.net.forward()

    def estimate(self, image, regions, k, dist, prior_boxes=()):
        crops = []
        self.candidates = []
        for color, box, contour in regions:
            x, y, w, h = box
            if min(w, h) < 14:
                continue
            mask = np.zeros((h, w), np.uint8)
            cv2.fillPoly(mask, [np.int32(contour - [x, y])], 255)
            value = image[y : y + h, x : x + w].max(axis=2)
            interior = cv2.erode(mask, np.ones((5, 5), np.uint8)) > 0
            if not interior.any():
                continue
            # A flat patch of cardboard or tape may have the same outline as
            # an occluded cube. Require some face-shading evidence for depth.
            samples = value[interior]
            lo, hi = np.partition(samples, [len(samples) // 20, 19 * len(samples) // 20])[
                [len(samples) // 20, 19 * len(samples) // 20]
            ]
            if float(hi) - float(lo) < max(16, 0.09 * float(hi)):
                continue
            if np.mean(cv2.absdiff(value, cv2.medianBlur(value, 5))[interior] > 8) > 0.16:
                continue
            if color == "red":
                # Red fingertips and red plastic overlap in hue. Only veto an
                # orange-red patch when the immediate surround also looks like skin.
                pad = max(3, min(w, h) // 3)
                left, top = max(0, x - pad), max(0, y - pad)
                context = cv2.cvtColor(image[top : y + h + pad, left : x + w + pad], cv2.COLOR_BGR2HSV)
                face = context[y - top : y - top + h, x - left : x - left + w][interior]
                orange = np.mean((face[:, 0] >= 2) & (face[:, 0] <= 8))
                if orange > 0.48:
                    ring = np.ones(context.shape[:2], bool)
                    ring[y - top : y - top + h, x - left : x - left + w] = False
                    hue, sat, light = context[ring].T
                    if hue.size and np.mean((hue <= 23) & (sat > 48) & (sat < 215) & (light > 32)) > 0.48:
                        continue
            patch = patch_input(value, mask, (0, 0, w, h))
            if patch is not None:
                tensor, origin, side = patch
                crops.append((tensor, origin + [x, y], side, color, box, mask, contour))
        if not crops:
            return []
        # Evaluate the same visible pixels in four orientations in one GPU batch.
        batch = np.float32([a[0] for a in crops]) / 255
        rotations = [np.rot90(batch, turns, axes=(2, 3)) for turns in range(4)]
        output = self.forward(np.concatenate(rotations)).reshape(4, len(crops), 17)
        guesses = []
        for index, (_, origin, side, color, box, mask, contour) in enumerate(crops):
            x, y, w, h = box
            prior = next(
                (
                    p
                    for n, p in prior_boxes
                    if n == color
                    and np.hypot(x + w / 2 - p[0] - p[2] / 2, y + h / 2 - p[1] - p[3] / 2) < 0.55 * max(p[2:])
                    and 0.44 < (w * h) / (p[2] * p[3]) < 2.2
                ),
                None,
            )
            # With the silhouette cut off at the sensor border, a printed strip
            # can fit a cube of arbitrary size. Reacquire inside the frame first.
            clipped = min(x, y, image.shape[1] - x - w, image.shape[0] - y - h) <= 1
            if clipped and prior is None:
                continue
            logits = output[:, index, -1]
            if (
                not prior
                and np.count_nonzero(logits > 1.5) < 2
                and max(logits) < 5.0
                or prior
                and max(logits) < -5.5
            ):
                continue
            self.candidates.append((color, box))
            poses = []
            for turns, result in enumerate(output[:, index]):
                if result[-1] < (-7 if prior else -1):
                    continue
                norm = result[:16].reshape(8, 2) * 0.5 + 0.5
                for _ in range(turns):
                    norm = np.column_stack(((SIZE - 1) / SIZE - norm[:, 1], norm[:, 0]))
                corners = norm * side + origin
                ok, rv, tv = cv2.solvePnP(CENTRED, corners, k, dist, flags=cv2.SOLVEPNP_SQPNP)
                if not ok or not np.isfinite(tv).all() or tv[2, 0] < 2 * 29:
                    continue
                pixels = cv2.projectPoints(CENTRED, rv, tv, k, dist)[0].reshape(8, 2)
                residual = np.sqrt(np.mean(np.sum((pixels - corners) ** 2, axis=1)))
                if residual > 0.076 * side:
                    continue
                hull = cv2.convexHull(np.float32(pixels - box[:2]))
                support = np.zeros_like(mask)
                cv2.fillConvexPoly(support, np.int32(np.rint(hull)), 255)
                common = np.count_nonzero(mask & support)
                if common < (0.70 if prior else 0.80) * np.count_nonzero(
                    mask
                ) or common < 0.30 * cv2.contourArea(hull):
                    continue
                score = residual / side + 0.12 * (1 - min(1, common / np.count_nonzero(mask)))
                poses.append((tv, rv, pixels, score))
            if not poses:
                continue
            centers = np.array([t.ravel() for t, *_ in poses])
            distances = np.linalg.norm(centers[:, None] - centers, axis=2)
            votes = (distances < np.maximum(9, centers[:, 2:3] * 0.09)).sum(axis=1)
            best = min(range(len(poses)), key=lambda i: (-votes[i], poses[i][3]))
            if not prior and (votes[best] < 2 or (min(w, h) < 26 and votes[best] < 3)):
                continue
            tv, rv, pixels, _ = poses[best]
            pose = np.eye(4)
            pose[:3, :3] = cv2.Rodrigues(rv)[0]
            pose[:3, 3] = tv.ravel()
            # Where six independent silhouette vertices are visible, refine
            # the network's depth against the actual edge pixels.
            geometry = silhouette_pose(contour, k, dist, pose, intact_only=True)
            refined = False
            if geometry:
                _, face, _ = geometry
                center = face[:3, 3] - HALF * face[:3, 2]
                if np.linalg.norm(center - tv.ravel()) < 0.13 * tv[2, 0]:
                    tv = center.reshape(3, 1)
                    pose[:3, :3], pose[:3, 3] = face[:3, :3], center
                    pixels = cv2.projectPoints(CENTRED, cv2.Rodrigues(face[:3, :3])[0], tv, k, dist)[
                        0
                    ].reshape(-1, 2)
                    refined = True
            guesses.append(
                {
                    "color": color,
                    "box": list(box),
                    "pixels": pixels.round(1).tolist(),
                    "strong": bool(votes[best] >= 2 and np.median(logits) > 1.5),
                    "camera_xyz": tv.ravel().round(1).tolist(),
                    "camera_pose": pose.tolist(),
                    "outline_refined": refined,
                }
            )
        return guesses
