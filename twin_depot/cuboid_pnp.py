"""Recover a 29 mm cube centre from visible vertices on multiple faces."""

from itertools import combinations, permutations, product

import cv2
import numpy as np

from .cube_pose import HALF, VERTICES

# Removing two opposite corners leaves the six-edge silhouette of a cube.
CENTRED = VERTICES + [0.0, 0.0, HALF]
RIM = np.array([1, 2, 3, 7, 4, 5])
SYMMETRIES = [
    m
    for p in permutations(range(3))
    for signs in product((-1, 1), repeat=3)
    if np.linalg.det(m := np.eye(3)[:, p] * signs) > 0
]


def _templates(count):
    # Shifts by two along this rim are equivalent cube rotations. Retain
    # both parities, in each winding, and every position of a missing corner.
    found = {}
    for chain in (RIM, RIM[::-1]):
        for shift in range(6):
            order = np.roll(chain, shift)
            for keep in combinations(range(6), count):
                obj = np.ascontiguousarray(CENTRED[order[list(keep)]])
                # Distances and signed volume distinguish reflected assignments.
                delta = obj[1:] - obj[0]
                gram = tuple((delta @ delta.T).astype(int).ravel())
                volumes = (
                    round(np.linalg.det(delta[list(rows)])) for rows in combinations(range(len(delta)), 3)
                )
                orientation = next((volume for volume in volumes if volume), 0)
                found.setdefault((gram, orientation), obj)
    return tuple(found.values())


TEMPLATES = {n: _templates(n) for n in (5, 6)}


def silhouette_pose(contour, k, dist, previous=None, *, intact_only=False):
    contour = np.int32(contour).reshape(-1, 2)
    x, y, w, h = cv2.boundingRect(contour)
    if min(w, h) < 22:
        return None
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    if intact_only and cv2.contourArea(contour) < 0.95 * cv2.contourArea(hull):
        return None
    tolerance = max(0.9, 0.011 * perimeter) if intact_only else max(1.3, 0.020 * perimeter)
    q = cv2.approxPolyDP(hull, tolerance, True).reshape(-1, 2).astype(float)
    if (intact_only and len(q) != 6) or not 5 <= len(q) <= 7:
        return None
    # Use held-out outline pixels as additional evidence. A chord through
    # the hand-cut part of a mask is not an observed physical cube edge.
    margin = 4
    origin = np.array([x - margin, y - margin])
    mask = np.zeros((h + 2 * margin, w + 2 * margin), np.uint8)
    cv2.fillPoly(mask, [np.int32(contour - origin)], 1)
    boundary = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    distance = cv2.distanceTransform(1 - boundary, cv2.DIST_L2, 3)
    solutions = []
    # Six intact corners provide the strongest constraint. A fifth observed
    # corner plus visible independent edges can rescue a clipped silhouette.
    for count in (6,) if intact_only else (6, 5):
        if len(q) < count:
            continue
        for subset in combinations(range(len(q)), count):
            pixels = np.ascontiguousarray(q[list(subset)])
            for obj in TEMPLATES[count]:
                ok, rv, tv = cv2.solvePnP(obj, pixels, k, dist, flags=cv2.SOLVEPNP_SQPNP)
                if not ok or not np.isfinite(tv).all() or tv[2, 0] <= 2 * HALF:
                    continue
                fit = cv2.projectPoints(obj, rv, tv, k, dist)[0].reshape(-1, 2)
                err = np.sqrt(np.mean(np.sum((fit - pixels) ** 2, axis=1)))
                if err > min(2.1, 0.024 * max(w, h)):
                    continue
                proj = cv2.projectPoints(CENTRED, rv, tv, k, dist)[0].reshape(-1, 2) - origin
                rim = cv2.convexHull(np.float32(proj)).reshape(-1, 2)
                if len(rim) != 6:
                    continue
                shape = np.zeros_like(mask)
                cv2.fillConvexPoly(shape, np.int32(np.rint(rim)), 1)
                common = np.count_nonzero(shape & mask)
                # Count projected area even when the object extends outside the patch.
                if common < 0.89 * mask.sum() or common < 0.73 * cv2.contourArea(rim):
                    continue
                samples = (
                    rim[:, None, :]
                    + (np.roll(rim, -1, axis=0) - rim)[:, None, :] * np.linspace(0.15, 0.85, 9)[None, :, None]
                )
                px = np.rint(samples).astype(int)
                good = np.all((px >= 0) & (px < [mask.shape[1], mask.shape[0]]), axis=2)
                px = np.clip(px, 0, [mask.shape[1] - 1, mask.shape[0] - 1])
                supported = np.mean(good & (distance[px[:, :, 1], px[:, :, 0]] < 1.8), axis=1)
                if np.count_nonzero(supported >= 0.66) < 4:
                    continue
                score = err + 2 * (1 - supported.mean())
                solutions.append((score, float(err), rv, tv, pixels))
        if solutions:
            break
    if not solutions:
        return None
    solutions.sort(key=lambda s: s[0])
    best, err, rv, tv, pixels = solutions[0]
    if any(score - best < 0.65 and np.linalg.norm(t - tv) > 5 for score, _, _, t, _ in solutions[1:]):
        return None
    r = cv2.Rodrigues(rv)[0]
    # Preserve the centre for every symmetry. The face rotation is conventional
    # for a plain cube and must never define the robot reference axes.
    rotations = [r @ s for s in SYMMETRIES]
    if previous is None:
        rot = min(rotations, key=lambda a: a[:, 2] @ tv.ravel())
    else:
        rot = min(rotations, key=lambda a: np.linalg.norm(a - previous[:3, :3]))
    pose = np.eye(4)
    pose[:3, :3] = rot
    pose[:3, 3] = tv.ravel() + HALF * rot[:, 2]
    return pixels, pose, err
