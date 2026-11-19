"""Metric pose of 29 mm cubes from model masks, face edges and visible silhouettes."""

import math

import cv2
import numpy as np

from .workcell_link import COLORS

HALF = 14.5
FACE = np.float64([[-HALF, HALF, 0], [HALF, HALF, 0], [HALF, -HALF, 0], [-HALF, -HALF, 0]])
VERTICES = np.vstack((FACE, FACE - [0.0, 0.0, 2 * HALF]))


def camera_matrix(shape, config):
    h, w = shape[:2]
    if config is None:
        focal = math.hypot(w, h) / (2 * math.tan(math.radians(68 / 2)))
        return np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]]), None
    if not isinstance(config, dict):
        raise ValueError("相机内参配置格式无效")
    size = config.get("image_size")
    if size != [w, h]:
        raise ValueError(f"内参要求尺寸 {size}，视频实际为 {[w, h]}")
    k = np.asarray(config.get("K"), dtype=float)
    dist = np.asarray(config.get("dist", []), dtype=float).reshape(-1)
    if (
        k.shape != (3, 3)
        or not np.isfinite(k).all()
        or min(k[0, 0], k[1, 1]) <= 0
        or not np.allclose(k[2], [0, 0, 1])
        or not np.isfinite(dist).all()
        or dist.size not in (0, 4, 5, 8, 12, 14)
    ):
        raise ValueError("相机内参矩阵或畸变系数无效，请重新载入标定结果")
    return k, dist if dist.size else None


def _ordered_face(quad, previous):
    pts = np.asarray(quad, dtype=np.float64)
    start = (
        int(np.argmin(pts.sum(axis=1)))
        if previous is None
        else min(range(4), key=lambda i: np.linalg.norm(np.roll(pts, -i, axis=0) - previous[0]))
    )
    return np.ascontiguousarray(np.roll(pts, -start, axis=0))


def _edge_line(image, a, b, radius, face_interior=False):
    """Sample the edge normal away from bevelled corners, then fit its supporting line."""
    delta = b - a
    length = np.linalg.norm(delta)
    if length < 8:
        return None
    normal = np.array([-delta[1], delta[0]]) / length
    offsets = np.arange(-radius, radius + 0.25, 0.5)
    fractions = np.linspace(0.16, 0.84, 17)
    centers = a + fractions[:, None] * delta
    samples = centers[:, None, :] + offsets[None, :, None] * normal
    strip = cv2.remap(
        image,
        samples[:, :, 0].astype("f"),
        samples[:, :, 1].astype("f"),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    ).astype(float)
    if strip.ndim == 2:
        strip = strip[:, :, None]
    gradients = np.linalg.norm(strip[:, 2:] - strip[:, :-2], axis=2)
    raw_gradients = gradients
    if face_interior:
        # The supporting line of the observed face ends at the *inner* edge
        # of a narrow visible side. Its outer silhouette often has more contrast.
        inside = centers + (radius + 3.0) * normal
        interior = cv2.remap(
            image,
            inside[:, 0:1].astype("f"),
            inside[:, 1:2].astype("f"),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        ).astype(float)
        inner_samples = strip[:, (np.arange(gradients.shape[1]) + 6).clip(0, strip.shape[1] - 1)]
        distance = np.linalg.norm(inner_samples - interior, axis=2)
        gradients = gradients * (0.15 + 0.85 / (1 + (distance / 16.0) ** 2))
    # A bevel produces two parallel responses. Select one coherent straight
    # edge across the strip before fitting individual subpixel samples.
    bins = np.arange(gradients.shape[1])
    drift = 2 * length * (fractions - 0.5) * np.linspace(-0.18, 0.18, 11)[:, None]
    predicted = np.rint(bins[None, :, None] + drift[:, None, :]).astype(int)
    valid = (predicted >= 0) & (predicted < len(bins))
    votes = gradients[np.arange(len(fractions)), predicted.clip(0, len(bins) - 1)]
    kth = int(0.9 * (gradients.size - 1))
    clip = max(12.0, np.partition(gradients.ravel(), kth)[kth])
    scores = np.mean(np.minimum(votes, clip) * valid, axis=2)
    slope, intercept = np.unravel_index(np.argmax(scores), scores.shape)
    idx = predicted[slope, intercept].clip(0, len(bins) - 1)
    nearby = (idx[:, None] + [-1, 0, 1]).clip(0, len(bins) - 1)
    idx = nearby[np.arange(len(idx)), np.argmax(gradients[np.arange(len(idx))[:, None], nearby], axis=1)]
    good = raw_gradients[np.arange(len(idx)), idx] >= 6
    if good.sum() < 11:
        return None
    positions = centers + offsets[idx + 1, None] * normal
    vx, vy, x, y = cv2.fitLine(positions[good].astype("f"), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    line = np.array([-vy, vx])
    residual = np.abs((positions - [x, y]) @ line)
    if np.count_nonzero(good & (residual < 1.25)) < 11 or abs(line @ delta / length) > 0.22:
        return None
    return np.array([*line, -line @ [x, y]])


def _intersections(image, polygon, radius):
    edges = [
        _edge_line(image, a, b, radius, face_interior=True)
        for a, b in zip(polygon, np.roll(polygon, -1, axis=0), strict=True)
    ]
    result = {}
    for i, q in enumerate(polygon):
        if edges[i - 1] is None or edges[i] is None:
            continue
        qh = np.cross(edges[i - 1], edges[i])
        if abs(qh[2]) > 0.25 and np.linalg.norm(qh[:2] / qh[2] - q) <= 2 * radius:
            result[i] = qh[:2] / qh[2]
    return result


def _color(hue):
    if hue < 10 or hue > 169:
        return "red"
    if 17 <= hue <= 35:
        return "yellow"
    if 36 <= hue <= 88:
        return "green"
    if 95 <= hue <= 135:
        return "blue"
    return None


def _color_outline(hsv, q, label):
    """Local connected colour support provides an independent outer silhouette."""
    center = q.mean(axis=0)
    radius = int(1.7 * np.linalg.norm(np.ptp(q, axis=0)))
    xy = np.maximum(np.rint(center).astype(int) - radius, 0)
    x, y = xy
    patch = hsv[y : int(center[1]) + radius + 1, x : int(center[0]) + radius + 1]
    h, s, v = cv2.split(patch)
    bounds = {"green": (36, 88), "blue": (95, 135), "yellow": (17, 35)}
    if label == "red":
        hue = ((h < 10) | (h > 169)) & (s > 158)
    else:
        lo, hi = bounds[label]
        hue = (h >= lo) & (h <= hi)
    mask = np.uint8(hue & (s > 52) & (v > 21))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    for c in contours:
        if cv2.pointPolygonTest(c, tuple((center - xy).tolist()), False) >= 0:
            bx, by, w, height = cv2.boundingRect(c)
            if bx == 0 or by == 0 or bx + w >= mask.shape[1] or by + height >= mask.shape[0]:
                return None
            # Retain the whole connected object and a margin for projected edges.
            start = np.maximum([bx - 18, by - 18], 0)
            end = np.minimum([bx + w + 18, by + height + 18], mask.shape[::-1])
            compact = np.zeros(tuple((end - start)[::-1]), np.uint8)
            cv2.drawContours(compact, [c - start.astype(np.int32)], 0, 255, -1)
            return xy + start, compact
    return None


def _side_constraints(mask, projected, hull, radius, bgr):
    constraints = []
    for a, b in zip(hull, np.roll(hull, -1), strict=True):
        if a < 4 and b < 4:
            continue
        p, q = projected[[a, b]]
        line = _edge_line(mask, p, q, radius)
        if line is None:
            continue
        if bgr is not None:
            # A JPEG's chroma boundary can extend beyond the physical edge.
            # Use its full-resolution appearance when both boundaries agree.
            aligned = np.array([v - line[:2] * (line[:2] @ v + line[2]) for v in (p, q)])
            intensity_line = _edge_line(bgr, *aligned, 3.5)
            if intensity_line is not None:
                line = intensity_line
        constraints.extend([(a, line), (b, line)])
    return constraints


def _fit_edges(pts, k, dist, rv, tv, origin, constraints):
    # A visible side edge constrains its normal direction even if an end is
    # hidden. The analytic projection Jacobian includes lens distortion.
    indices = [idx for idx, _ in constraints]
    lines = np.array([line for _, line in constraints])
    for _ in range(6):
        projected, jac = cv2.projectPoints(VERTICES, rv, tv, k, dist)
        projected = projected.reshape(-1, 2)
        jac = jac[:, :6].reshape(-1, 2, 6)
        errors = ((projected[indices] - origin) * lines[:, :2]).sum(axis=1) + lines[:, 2]
        jac_lines = (jac[indices] * lines[:, :2, None]).sum(axis=1)
        residual = np.r_[(projected[:4] - pts).ravel(), errors]
        jacobian = np.vstack([jac[:4].reshape(-1, 6), jac_lines])
        weights = np.sqrt(np.minimum(1.0, 2.0 / np.maximum(abs(residual), 1e-6)))
        step = np.linalg.lstsq(jacobian * weights[:, None], -residual * weights, rcond=None)[0]
        rv = rv + step[:3, None]
        tv = tv + step[3:, None]
        if np.linalg.norm(step) < 0.01:
            break
    return rv, tv


def _overlap(mask, pixels):
    if (
        not np.isfinite(pixels).all()
        or pixels.min() < 0
        or pixels[:, 0].max() >= mask.shape[1]
        or pixels[:, 1].max() >= mask.shape[0]
    ):
        return (), 0.0
    hull = cv2.convexHull(np.float32(pixels), returnPoints=False).ravel()
    support = np.zeros_like(mask)
    cv2.fillConvexPoly(support, np.int32(np.rint(pixels[hull])), 255)
    return hull, np.count_nonzero(support & mask) / max(1, np.count_nonzero(support | mask))


def _cube_fit(quad, k, dist, previous, outline, image=None):
    quad = np.asarray(quad, dtype=np.float64)
    if (
        quad.shape != (4, 2)
        or not np.isfinite(quad).all()
        or not cv2.isContourConvex(np.float32(quad))
        or cv2.contourArea(np.float32(quad)) < 1
    ):
        return None
    pts = _ordered_face(quad, previous)
    _, rs, ts, _ = cv2.solvePnPGeneric(FACE, pts, k, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    roots = [
        (r, t)
        for r, t in zip(rs, ts, strict=True)
        if np.isfinite(r).all()
        and np.isfinite(t).all()
        and t[2, 0] > 0
        and cv2.Rodrigues(r)[0][:, 2] @ t[:, 0] < 0
    ]
    if not roots:
        # IPPE can return a back-facing solution or NaNs at exactly 180°
        # rotation (a centred face seen straight on). The planar homography
        # initialization of iterative PnP is well-defined in this case.
        ok, rv, tv = cv2.solvePnP(FACE, pts, k, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if ok and np.isfinite(rv).all() and np.isfinite(tv).all() and tv[2, 0] > 0:
            roots.append((rv, tv))
    options = []
    for rv, tv in roots:
        silhouette_cost = 0.0
        if outline is not None:
            origin, mask = outline
            pixels = cv2.projectPoints(VERTICES, rv, tv, k, dist)[0].reshape(-1, 2) - origin
            hull, overlap = _overlap(mask, pixels)
            if overlap < 0.76:
                continue
            radius = np.clip(0.18 * np.sqrt(cv2.contourArea(np.float32(pts))), 3.5, 14.0)
            x, y = origin
            bgr = None if image is None else image[y : y + mask.shape[0], x : x + mask.shape[1]]
            constraints = _side_constraints(mask, pixels, hull, radius, bgr)
            if constraints:
                rv, tv = _fit_edges(pts, k, dist, rv, tv, origin, constraints)
                if not np.isfinite(tv).all() or tv[2, 0] <= 0:
                    continue
                pixels = cv2.projectPoints(VERTICES, rv, tv, k, dist)[0].reshape(-1, 2) - origin
                _, overlap = _overlap(mask, pixels)
                if overlap < 0.76:
                    continue
            silhouette_cost = 12 * (1 - overlap)
        rot = cv2.Rodrigues(rv)[0]
        if rot[:, 2] @ tv[:, 0] >= 0:
            continue
        fit = cv2.projectPoints(FACE, rv, tv, k, dist)[0].reshape(4, 2)
        err = np.sqrt(np.mean(np.sum((fit - pts) ** 2, axis=1)))
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = rot, tv[:, 0]
        continuity = 0 if previous is None else 0.5 * np.linalg.norm(rot - previous[1][:3, :3])
        if err < 2.5:
            options.append((err + continuity + silhouette_cost, err, pose))
    if options:
        options.sort(key=lambda v: v[0])
        score, err, pose = options[0]
        if outline is not None and len(options) > 1:
            other_score, _, other = options[1]
            if other_score - score < 0.35 and pose[:3, 2] @ other[:3, 2] < math.cos(math.radians(18)):
                return None
        return pts, pose, float(err)


def face_pose(quad, k, dist, previous=None):
    """Estimate pose when only the four observed face corners are available."""
    return _cube_fit(quad, k, dist, previous, None)


def _exposed_face(image, points, k, dist, outline, previous):
    """Four measured face corners with a clipped side. Reject ambiguous centre estimates."""
    xy, mask = outline
    pts = _ordered_face(points, previous)
    local = pts - xy
    inside = np.zeros_like(mask)
    cv2.fillConvexPoly(inside, np.int32(np.rint(local)), 255)
    if np.count_nonzero(inside & mask) < 0.89 * max(1, np.count_nonzero(inside)):
        return None
    options = []
    _, rs, ts, _ = cv2.solvePnPGeneric(FACE, pts, k, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    for rv, tv in zip(rs, ts, strict=True):
        if not np.isfinite(tv).all() or tv[2, 0] <= 2 * HALF:
            continue
        r = cv2.Rodrigues(rv)[0]
        if r[:, 2] @ tv.ravel() >= 0:
            continue
        pix = cv2.projectPoints(VERTICES, rv, tv, k, dist)[0].reshape(-1, 2)
        error = np.sqrt(np.mean(np.sum((pix[:4] - pts) ** 2, axis=1)))
        hull = cv2.convexHull(np.float32(pix - xy))
        support = np.zeros_like(mask)
        cv2.fillConvexPoly(support, np.int32(np.rint(hull)), 255)
        common = np.count_nonzero(support & mask)
        if error > 1.8 or common < 0.88 * np.count_nonzero(mask) or common < 0.58 * cv2.contourArea(hull):
            continue
        pose = np.eye(4)
        pose[:3, :3] = r
        pose[:3, 3] = tv.ravel()
        cost = error + 3 * (1 - common / max(1, np.count_nonzero(mask)))
        options.append((cost, float(error), pose))
    if not options:
        return None
    options.sort(key=lambda v: v[0])
    score, err, pose = options[0]
    centre = pose[:3, 3] - HALF * pose[:3, 2]
    if any(
        cost - score < 0.55 and np.linalg.norm(other[:3, 3] - HALF * other[:3, 2] - centre) > 4
        for cost, _, other in options[1:]
    ):
        return None
    return pts, pose, err


def red_regions(hsv):
    h, s, v = cv2.split(hsv)
    mask = np.uint8(((h < 10) | (h > 169)) & (s > 95) & (v > 65))
    cs = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    boxes = []
    for c in cs:
        x, y, w, h = cv2.boundingRect(c)
        if (
            min(w, h) >= 19
            and max(w, h) < min(mask.shape) * 0.45
            and 0.45 < w / h < 2
            and cv2.contourArea(c) > 0.53 * w * h
        ):
            boxes.append((x, y, w, h))
    return boxes


def _lit_faces(patch, mask, offset):
    vals = patch[mask > 0]
    if len(vals) < 250 or np.std(vals) < 10:
        return []
    threshold = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    faces = []
    for level in (threshold, threshold + 0.28 * np.std(vals)):
        lit = np.uint8((patch > level) & (mask > 0))
        contours = cv2.findContours(lit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if contours:
            face = max(contours, key=cv2.contourArea)
            if 0.27 * len(vals) < cv2.contourArea(face) < 0.96 * len(vals):
                faces.append(face.reshape(-1, 2) + offset)
    return faces


def _work_crop(hsv, boxes):
    # Keep targets at the far end of the belt in the model input. The red
    # reference supplies scale for the crop, not a maximum target distance.
    height, width = hsv.shape[:2]
    scale = max(max(b[2:]) for b in boxes)
    left = max(0, min(b[0] for b in boxes) - 4 * scale)
    top = max(0, min(b[1] for b in boxes) - 4 * scale)
    right = min(width, max(b[0] + b[2] for b in boxes) + 4 * scale)
    bottom = min(height, max(b[1] + b[3] for b in boxes) + 4 * scale)
    h, s, v = cv2.split(hsv)
    mask = np.uint8((h >= 17) & (h <= 135) & (s > 65) & (v > 45))
    for c in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        x, y, w, bh = cv2.boundingRect(c)
        if not (min(w, bh) >= 17 and max(w, bh) <= 3 * scale and cv2.contourArea(c) > 0.40 * w * bh):
            continue
        margin = max(w, bh) // 2
        left, top = max(0, min(left, x - margin)), max(0, min(top, y - margin))
        right, bottom = min(width, max(right, x + w + margin)), min(height, max(bottom, y + bh + margin))
    return left, top, right, bottom, scale


def visible_regions(hsv):
    """Bounded proposals from actual coloured pixels. These alone are not cube poses."""
    h, s, v = cv2.split(hsv)
    saturated = (s > 68) & (v > 23)
    regions = []
    for label, chroma in (
        ("red", ((h < 6) | (h > 169)) & (s > 165)),
        ("green", (h >= 36) & (h <= 88)),
        ("blue", (h >= 95) & (h <= 135)),
        ("yellow", (h >= 17) & (h <= 35)),
    ):
        mask = cv2.morphologyEx(np.uint8(saturated & chroma), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        for contour in contours:
            x, y, w, bh = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            if (
                min(w, bh) < 15
                or max(w, bh) > min(h.shape) * 0.67
                or not 0.35 < w / bh < 2.8
                or area < max(180, 0.28 * w * bh)
            ):
                continue
            if label == "red":
                # A finger touching a red face can join the component through
                # orange-red skin. Prefer its purer core when most of the colour survives.
                patch = h[y : y + bh, x : x + w]
                core = np.uint8(
                    ((patch <= 1) | (patch >= 172))
                    & (s[y : y + bh, x : x + w] > 182)
                    & (v[y : y + bh, x : x + w] > 23)
                )
                core = cv2.morphologyEx(core, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
                solid = cv2.findContours(core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                if solid:
                    peak = max(solid, key=cv2.contourArea)
                    if cv2.contourArea(peak) > 0.48 * area:
                        contour = np.int32(peak + [x, y])
                        x, y, w, bh = cv2.boundingRect(contour)
                        area = cv2.contourArea(contour)
            hull = cv2.convexHull(contour)
            if area < 0.53 * cv2.contourArea(hull):
                continue
            regions.append((label, (x, y, w, bh), contour.reshape(-1, 2)))
    return regions


def infer_faces(
    image,
    model,
    k,
    dist,
    previous,
    *,
    require_reference=True,
    regions=None,
    candidates=None,
    measured=(),
    fixed_reference=False,
):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    boxes = red_regions(hsv)
    if not boxes and require_reference and candidates is None:
        return [], "请保留一块清晰可见的红色参考块"
    # A hand or a label can also contain red blobs. Only count references
    # after checking their 3-D cube geometry, not at the colour-proposal stage.
    if regions is None:
        regions = visible_regions(hsv)
    if not regions and not boxes:
        return [], "等待彩色方块入镜"
    if require_reference and boxes:
        left, top, right, bottom, scale = _work_crop(hsv, boxes)
    else:
        all_boxes = [b for _, b, _ in regions] or boxes
        # Keep separated targets and their surroundings in a single model input.
        left = max(0, min(x - w // 2 for x, y, w, h in all_boxes))
        top = max(0, min(y - h // 2 for x, y, w, h in all_boxes))
        right = min(image.shape[1], max(x + w + w // 2 for x, y, w, h in all_boxes))
        bottom = min(image.shape[0], max(y + h + h // 2 for x, y, w, h in all_boxes))
        scale = max(max(b[2:]) for b in all_boxes)
    crop = image[top:bottom, left:right]
    validated = []
    for color, points, _, _ in measured:
        center = points.mean(axis=0)
        same = [
            (w * h, (x, y, w, h))
            for n, (x, y, w, h), p in regions
            if n == color
            and x <= center[0] < x + w
            and y <= center[1] < y + h
            and cv2.pointPolygonTest(p, tuple(center), False) >= 0
        ]
        if same:
            validated.append((color, min(same)[1]))

    def already_measured(label, xy):
        return any(n == label and x <= xy[0] < x + w and y <= xy[1] < y + h for n, (x, y, w, h) in validated)

    polygons = []
    for poly in model(crop):
        points = np.asarray(poly, dtype=np.float32)
        if points.size < 8 or points.shape[-1] != 2 or not np.isfinite(points).all():
            continue
        points = points.reshape(-1, 2)
        if points.min() < 0 or points[:, 0].max() >= crop.shape[1] or points[:, 1].max() >= crop.shape[0]:
            continue
        polygons.append(points)
    # FastSAM often returns the complete solid. Split its illuminated face from darker sides
    # inside that instance before fitting the four corresponding corners.
    crop_hsv = hsv[top:bottom, left:right]
    value = crop_hsv[:, :, 2]
    refinements = []
    seen_masks = set()
    for poly in polygons:
        if len(poly) < 4:
            continue
        bx, by, bw, bh = cv2.boundingRect(np.float32(poly))
        if min(bw, bh) < 18 or max(bw, bh) > 3 * scale:
            continue
        mask = np.zeros((bh, bw), np.uint8)
        cv2.fillPoly(mask, [np.int32(poly - [bx, by])], 255)
        patch = value[by : by + bh, bx : bx + bw]
        if patch.shape != mask.shape:
            continue
        local_hsv = crop_hsv[by : by + bh, bx : bx + bw]
        hues = local_hsv[:, :, 0][(mask > 0) & (local_hsv[:, :, 1] > 62) & (patch > 40)]
        if len(hues) < 0.60 * cv2.countNonZero(mask):
            continue
        label = _color(np.bincount(hues).argmax())
        if label is None:
            continue
        if already_measured(label, (left + bx + bw / 2, top + by + bh / 2)):
            continue
        if candidates is not None:
            # Only label a 2-D cue if an independent model mask agrees with a
            # compact colour component. A fingertip may hide one or more corners.
            for color, (x, y, w, h), contour in regions:
                if label != color:
                    continue
                short, long = sorted(cv2.minAreaRect(contour)[1])
                known = any(
                    n == label and np.linalg.norm(p[0].mean(axis=0) - [x + w / 2, y + h / 2]) < max(w, h)
                    for n, p in previous
                )
                if not known and (short < 24 or long > 2.15 * short):
                    continue
                ax, ay = bx + left, by + top
                intersection = max(0, min(ax + bw, x + w) - max(ax, x)) * max(
                    0, min(ay + bh, y + h) - max(ay, y)
                )
                if intersection / (bw * bh + w * h - intersection) > 0.42 or (
                    intersection > 0.86 * bw * bh and bw * bh > 0.24 * w * h
                ):
                    candidates.add((color, (x, y, w, h)))
        refinements.extend(_lit_faces(patch, mask, [bx, by]))
        # SAM can shave pixels off a bevel or join a cube to a fingertip.
        # Recover the full colour-connected silhouette touched by this instance.
        box = np.float32([[bx, by], [bx + bw, by], [bx + bw, by + bh], [bx, by + bh]])
        outline = _color_outline(crop_hsv, box, label)
        if outline is None:
            continue
        xy, component = outline
        cx, cy, cw, ch = cv2.boundingRect(component)
        region_key = (label, int(xy[0] + cx), int(xy[1] + cy), cw, ch)
        if region_key in seen_masks or min(cw, ch) < 18 or max(cw, ch) > 2.5 * max(bw, bh):
            continue
        seen_masks.add(region_key)
        cx, cy = xy
        refinements.extend(
            _lit_faces(value[cy : cy + component.shape[0], cx : cx + component.shape[1]], component, xy)
        )
    # A colour-supported face can be visible inside a segmentation that also
    # contains a hand. Keep the native-resolution edges of this component.
    if candidates is not None:
        for label, (x, y, w, bh), contour in regions:
            if already_measured(label, (x + w / 2, y + bh / 2)) or (label, (x, y, w, bh)) not in candidates:
                continue
            local_contour = np.int32(contour) - [left, top]
            refinements.append(local_contour)
            bx, by = x - left, y - top
            patch = value[by : by + bh, bx : bx + w]
            if patch.shape == (bh, w):
                compact = np.zeros((bh, w), np.uint8)
                cv2.fillPoly(compact, [np.int32(contour - [x, y])], 255)
                refinements.extend(_lit_faces(patch, compact, [bx, by]))
    polygons.extend(refinements)
    detections = {}
    seen_quads = set()
    for polygon in polygons:
        contour = np.float32(polygon) + np.float32([left, top])
        if len(contour) < 4:
            continue
        hull = cv2.convexHull(contour)
        q = cv2.approxPolyDP(hull, 0.027 * cv2.arcLength(hull, True), True).reshape(-1, 2)
        if (
            len(q) != 4
            or cv2.contourArea(q) < 250
            or abs(cv2.contourArea(q) - cv2.contourArea(contour)) > 0.11 * cv2.contourArea(q)
        ):
            continue
        if cv2.contourArea(q, oriented=True) < 0:
            q = q[::-1]
        quad_key = tuple(np.roll(q, -int(np.argmin(q.sum(axis=1))), axis=0).ravel())
        if quad_key in seen_quads:
            continue
        seen_quads.add(quad_key)
        lengths = np.linalg.norm(np.roll(q, 1, axis=0) - q, axis=1)
        if min(lengths) < 16 or max(lengths) > min(lengths) * 1.9:
            continue
        center = q.mean(axis=0)
        samples = np.rint(0.55 * q + 0.45 * center).astype(int)
        px = hsv[samples[:, 1], samples[:, 0]]
        if min(px[:, 1]) < 62 or min(px[:, 2]) < 40:
            continue
        labels = [_color(hue) for hue in px[:, 0]]
        label = labels[0]
        if not label or any(other != label for other in labels):
            continue
        if already_measured(label, center):
            continue
        if (
            require_reference
            and label == "red"
            and not any(bx <= center[0] < bx + bw and by <= center[1] < by + bh for bx, by, bw, bh in boxes)
        ):
            continue
        priors = [
            p
            for name, p in previous
            if name == label and np.linalg.norm(center - p[0].mean(axis=0)) < max(lengths)
        ]
        prior = min(priors, key=lambda p: np.linalg.norm(center - p[0].mean(axis=0)), default=None)
        radius = np.clip(0.13 * min(lengths), 3.0, 9.0)
        refined = _intersections(image, q, radius)
        if len(refined) != 4:
            continue
        q = np.array([refined[i] for i in range(4)])
        outline = _color_outline(hsv, q, label)
        if outline is None:
            continue
        fit = _cube_fit(q, k, dist, prior, outline, image)
        if (
            not fit
            and candidates is not None
            and not (fixed_reference and label == "red")
            and any(
                n == label and x <= center[0] < x + w and y <= center[1] < y + h
                for n, (x, y, w, h) in candidates
            )
        ):
            fit = _exposed_face(image, q, k, dist, outline, prior)
        if fit:
            pts, pose, err = fit
            # Different masks can describe different faces of the very same
            # cube. Match their observed component instead of merging in 3-D,
            # where individual depth estimates have substantial uncertainty.
            xy, component = outline
            bx, by, bw, bh = cv2.boundingRect(component)
            key = (label, bx + xy[0], by + xy[1], bw, bh)
            normal = pose[:3, 2]
            facing = -normal @ pose[:3, 3] / np.linalg.norm(pose[:3, 3])
            score = err + 0.8 * (1 - facing)
            if prior is not None:
                score += 0.8 * (1 - normal @ prior[1][:3, 2])
            if key not in detections or score < detections[key][0]:
                detections[key] = (score, (label, pts, pose, err))
    kept = list(measured) + [value[1] for value in detections.values()]
    if not require_reference:
        return kept, "" if kept else "等待可解算三维姿态的方块面"
    refs = [p for p in kept if p[0] == "red"]
    if len(refs) > 1:
        return [], "检测到多个红色参考块，请只保留一个"
    if len(refs) != 1:
        return [], "红块面不清晰或姿态有歧义，请让完整顶面入镜"
    return kept, ""
