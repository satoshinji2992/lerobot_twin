"""Detect instances on keyframes and remeasure visible cube edges between them."""

import cv2
import numpy as np

from .cube_pose import (
    _color,
    _color_outline,
    _cube_fit,
    _exposed_face,
    _intersections,
    infer_faces,
    visible_regions,
)
from .cuboid_pnp import silhouette_pose


def _follow(image, hsv, gray, old_gray, previous):
    """Track corners forwards and backwards, then refit against this JPEG at full resolution."""
    points = np.float32([pts for _, (pts, _) in previous]).reshape(-1, 1, 2)
    opts = {
        "winSize": (21, 21),
        "maxLevel": 3,
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
    }
    target, ok, _ = cv2.calcOpticalFlowPyrLK(old_gray, gray, points, None, **opts)
    back, ok_back, _ = cv2.calcOpticalFlowPyrLK(gray, old_gray, target, None, **opts)
    good = (
        ok.ravel().astype(bool)
        & ok_back.ravel().astype(bool)
        & (np.linalg.norm(back - points, axis=2).ravel() < 1.2)
    )
    for i, (label, prior) in enumerate(previous):
        group = slice(i * 4, i * 4 + 4)
        q = target[group, 0].astype(float)
        if (
            not good[group].all()
            or not np.isfinite(q).all()
            or np.any(q < 2)
            or np.any(q >= [gray.shape[1] - 2, gray.shape[0] - 2])
            or not cv2.isContourConvex(np.float32(q))
        ):
            continue
        side = np.min(np.linalg.norm(q - np.roll(q, 1, axis=0), axis=1))
        if side < 15:
            continue
        inside = np.rint(0.55 * q + 0.45 * q.mean(axis=0)).astype(int)
        if any(s < 62 or v < 40 or _color(h) != label for h, s, v in hsv[inside[:, 1], inside[:, 0]]):
            continue
        refined = _intersections(image, q, np.clip(side * 0.09, 3.0, 6.0))
        if len(refined) == 4:
            quad = np.array([refined[n] for n in range(4)])
            outline = _color_outline(hsv, quad, label)
            if outline is not None:
                yield label, quad, prior, outline


def _near(box, reference):
    x, y, w, h = box
    bx, by, bw, bh = reference
    return max(w / bw, bw / w, h / bh, bh / h) < 2.4 and np.hypot(
        x + w / 2 - bx - bw / 2, y + h / 2 - by - bh / 2
    ) < 0.56 * max(bw, bh)


class CubeObserver:
    def __init__(self, completion=None):
        self.gray = None
        self.poses = []
        self.boxes = []
        self.last_scan = -float("inf")
        self.last_frame = -float("inf")
        self.completion = completion
        self.estimates = []
        self.last_input = self.last_result = None

    def observe(self, image, predict, k, dist, at, *, reference=False, source_jpeg=None):
        # Re-fitting an identical JPEG by optical flow can move its corners even
        # though only the workcell axes changed. The source bytes precede annotation.
        key = (
            None
            if source_jpeg is None
            else (reference, k.tobytes(), None if dist is None else np.asarray(dist).tobytes(), source_jpeg)
        )
        if key is not None and key == self.last_input and self.poses and 0 < at - self.last_frame < 0.30:
            self.last_frame = at
            return self.last_result[0], self.last_result[1], "相同画面 · 沿用角点"
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        regions = visible_regions(hsv)
        if self.gray is None or self.gray.shape != gray.shape or not 0 < at - self.last_frame < 0.30:
            self.poses, self.boxes = [], []
            self.last_scan = -float("inf")
        detections = []
        candidates = set()
        scan = at - self.last_scan >= 0.40
        if self.poses:
            for label, quad, prior, outline in _follow(image, hsv, gray, self.gray, self.poses):
                fit = _cube_fit(quad, k, dist, prior, outline, image)
                if not fit and not (reference and label == "red"):
                    fit = _exposed_face(image, quad, k, dist, outline, prior)
                if fit:
                    detections.append((label, *fit))
        if not scan:
            for label, box, _ in regions:
                if any(label == old and _near(box, b) for old, b in self.boxes):
                    candidates.add((label, box))
            if not detections and at - self.last_scan >= 0.12:
                scan = True
        if scan:
            detections, _ = infer_faces(
                image,
                predict,
                k,
                dist,
                self.poses,
                require_reference=False,
                regions=regions,
                candidates=candidates,
                measured=detections,
                fixed_reference=reference,
            )
            self.last_scan = at
            # Promote candidates between model keyframes only while nearby pixels of the same colour persist.
            self.boxes = list(candidates) + [
                (n, cv2.boundingRect(np.float32(p))) for n, p, _, _ in detections
            ]
        self.gray, self.last_frame = gray, at
        self.poses = [(n, (p, t)) for n, p, t, _ in detections]
        valid = [(name, cv2.boundingRect(np.float32(points))) for name, points, _, _ in detections]
        # In workcell mode only a fully identified face may supply the reference axes.
        for label, box, contour in regions:
            if (label, box) not in candidates or (reference and label == "red"):
                continue
            if any(label == n and _near(box, b) for n, b in valid):
                continue
            fit = silhouette_pose(contour, k, dist)
            if fit:
                detections.append((label, *fit))
                valid.append((label, box))
        partial = [
            {"color": label, "box": list(box)}
            for label, box in sorted(candidates)
            if not any(label == n and _near(box, b) for n, b in valid)
        ]
        unmatched = [r for r in regions if not any(r[0] == n and _near(r[1], b) for n, b in valid)]
        self.estimates = (
            self.completion.estimate(image, unmatched, k, dist) if self.completion is not None else []
        )
        self.last_input = key
        self.last_result = detections, partial, "模型重检" if scan else "光流与本帧边缘"
        return self.last_result


class FastObserver:
    """One batched 40×40 network pass and a PnP per proposal; no segment-anything model."""

    def __init__(self, completion):
        self.completion = completion
        self.estimates = []
        self.priors = []
        self.fragments = []
        self.tentative = []
        self.stamp = -float("inf")

    def observe(self, image, predict, k, dist, at, *, reference=False):
        if reference:
            raise ValueError("固定工位坐标需要精测模式")
        if self.completion is None:
            raise ValueError("轻量模型缺失，请载入 assets/cube-completion.onnx")
        h, w = image.shape[:2]
        scale = min(1.0, 640 / max(h, w))
        cropped = (
            cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
            if scale < 1
            else image
        )
        ratio = np.array(cropped.shape[1::-1]) / np.array([w, h])
        small_k = k.copy()
        small_k[:2] *= ratio[:, None]
        small_k[:2, 2] += (ratio - 1) / 2
        regions = visible_regions(cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV))
        if not 0 < at - self.stamp <= 0.25:
            self.priors = []
        self.fragments = []
        self.stamp = at
        priors = [(n, b, t) for n, b, t in self.priors if 0 <= at - t < 0.55]
        self.estimates = self.completion.estimate(
            cropped, regions, small_k, dist, [(n, b) for n, b, _ in priors]
        )
        # A finger can split one silhouette into two colour components. A
        # small neighbouring component often solves at a much greater depth.
        # Give it time to establish an independent track before drawing two cubes.
        pending, confirmed, tentative = [], [], []
        for item in self.estimates:
            x, y, w, h = item["box"]
            known = any(n == item["color"] and _near(item["box"], b) for n, b, _ in priors)
            if not known and (min(w, h) < 26 or not item["strong"]):
                hits = 1 + max(
                    (hits for n, b, hits in self.tentative if n == item["color"] and _near(item["box"], b)),
                    default=0,
                )
                tentative.append((item["color"], item["box"], hits))
                if hits < 3:
                    continue
            fragment = False
            for other in self.estimates:
                bx, by, bw, bh = other["box"]
                if other["color"] != item["color"] or w * h > 0.60 * bw * bh:
                    continue
                gap = np.hypot(max(0, bx - x - w, x - bx - bw), max(0, by - y - h, y - by - bh))
                if gap < 0.26 * max(bw, bh) and item["camera_xyz"][2] > 1.27 * other["camera_xyz"][2]:
                    fragment = True
                    break
            if fragment:
                start = next(
                    (t for n, b, t in self.fragments if n == item["color"] and _near(item["box"], b)), at
                )
                pending.append((item["color"], item["box"], start))
                if at - start < 0.30:
                    continue
            confirmed.append(item)
        self.fragments, self.estimates, self.tentative = pending, confirmed, tentative
        updated = []
        for r in self.estimates:
            n, b = r["color"], r["box"]
            seen = max((t for name, old, t in priors if n == name and _near(b, old)), default=-float("inf"))
            updated.append((n, b, at if r["strong"] else seen))
        self.priors = updated + [
            (n, b, t) for n, b, t in priors if not any(n == n2 and _near(b, b2) for n2, b2, _ in updated)
        ]

        def original_box(box):
            return np.rint(np.asarray(box) / np.tile(ratio, 2)).astype(int).tolist()

        accepted = {(r["color"], tuple(r["box"])) for r in self.estimates}
        partial = [
            {"color": n, "box": original_box(box)}
            for n, box in self.completion.candidates
            if (n, tuple(box)) not in accepted
        ]
        for r in self.estimates:
            r["pixels"] = ((np.asarray(r["pixels"]) + 0.5) / ratio - 0.5).round(1).tolist()
            r["box"] = original_box(r["box"])
        return [], partial, "角点补全 + 八点 PnP"
