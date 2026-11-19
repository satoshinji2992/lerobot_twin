"""Recover square faces and three-edge cube corners from measured image lines."""

from itertools import combinations
from functools import lru_cache

import cv2
import numpy as np

from .cube_pose import HALF


@lru_cache(maxsize=16)
def _line_triples(count):
    return np.array(list(combinations(range(count), 3)))


def image_segments(value, mask):
    # The colour boundary supplies exterior edges even on a similarly bright
    # background. Inside the object, brightness changes supply shared ridges.
    plane = np.where(mask != 0, value, 0).astype(np.uint8)
    # A binary colour boundary contains one-pixel staircase steps. Suppress
    # them before upsampling or LSD can split a long oblique edge into steps.
    plane = cv2.GaussianBlur(plane, (5, 5), 0.85)
    upscale = 2 if min(plane.shape) < 100 else 1
    if upscale == 2:
        plane = cv2.resize(plane, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    detected = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(plane)[0]
    if detected is None:
        return np.empty((0, 2, 2))
    lines = (detected.reshape(-1, 2, 2).astype(float) + 0.5) / upscale - 0.5
    lengths = np.linalg.norm(lines[:, 1] - lines[:, 0], axis=1)
    ids = np.flatnonzero(lengths >= max(4.5, 0.13 * np.sqrt(cv2.countNonZero(mask))))
    ids = ids[np.argsort(-lengths[ids])[:16]]
    return lines[ids]


def _line_equations(lines):
    lengths = np.linalg.norm(lines[:, 1] - lines[:, 0], axis=1)
    direction = (lines[:, 1] - lines[:, 0]) / lengths[:, None]
    normal = np.column_stack((-direction[:, 1], direction[:, 0]))
    offset = np.einsum("ij,ij->i", normal, lines[:, 0])
    return lengths, direction, normal, offset


def face_quads(lines, mask):
    if len(lines) < 4:
        return []
    lengths, direction, normal, offset = _line_equations(lines)
    cosine = abs(direction @ direction.T)
    p, q = np.where(np.triu((cosine > 0.91) & (abs(lines.mean(1) @ normal.T - offset) > 4), 1))
    if len(p) < 2:
        return []
    a, b = np.triu_indices(len(p), 1)
    i, j, m, n = p[a], q[a], p[b], q[b]
    cross_angles = np.maximum.reduce([cosine[i, m], cosine[i, n], cosine[j, m], cosine[j, n]])
    good = cross_angles < 0.95
    if not good.any():
        return []
    sides = np.stack((i[good], m[good], j[good], n[good]), axis=1)
    after = np.roll(sides, -1, axis=1)
    u, v = normal[sides], normal[after]
    det = u[:, :, 0] * v[:, :, 1] - u[:, :, 1] * v[:, :, 0]
    xy = np.stack((offset[sides] * v[:, :, 1] - offset[after] * u[:, :, 1],
                   offset[after] * u[:, :, 0] - offset[sides] * v[:, :, 0]), axis=2) / det[:, :, None]
    # A vertex may be interrupted by a finger. Only extrapolate when each of
    # its two supporting segments actually spans much of the associated edge.
    prev = np.roll(xy, 1, axis=1)
    start = lines[sides, 0]
    along = direction[sides]
    t0 = ((prev - start) * along).sum(axis=2)
    t1 = ((xy - start) * along).sum(axis=2)
    lo, hi = np.minimum(t0, t1), np.maximum(t0, t1)
    span = hi - lo
    overlap = np.minimum(hi, lengths[sides]) - np.maximum(lo, 0.)
    support = overlap / np.maximum(span, 1e-6)
    excess = (np.maximum(-lo, 0.) + np.maximum(hi - lengths[sides], 0.))
    within = np.all((xy >= 0) & (xy < mask.shape[::-1]), axis=(1, 2))
    valid = (within & (span.min(1) > 5) & (support.min(1) > 0.45)
             & (excess <= 3 + 0.50 * span).all(1))
    quality = np.mean(support, axis=1) + 0.3 * support.min(1)
    quads = []
    for idx in np.flatnonzero(valid)[np.argsort(-quality[valid])]:
        corners = xy[idx]
        poly = corners.astype(np.float32)
        area = cv2.contourArea(poly, oriented=True)
        if abs(area) < max(40., 0.13 * cv2.countNonZero(mask)) or not cv2.isContourConvex(poly):
            continue
        # Only accept faces supported by the cube's pigment across their interior.
        uv = np.rint(corners.mean(0) + 0.72 * (corners - corners.mean(0))).astype(int)
        uv = np.clip(uv, 0, np.array(mask.shape[::-1]) - 1)
        if np.count_nonzero(mask[uv[:, 1], uv[:, 0]]) < 3:
            continue
        quads.append((np.ascontiguousarray(corners if area > 0 else corners[::-1]), lines[sides[idx]]))
        if len(quads) == 6:
            break
    return quads


def corner_junctions(lines):
    """Keep intersections where three distinct segments actually terminate."""
    if len(lines) < 3:
        return
    lengths, _, normal, offset = _line_equations(lines)
    a, b = normal[:, None], normal[None, :]
    det = a[:, :, 0] * b[:, :, 1] - a[:, :, 1] * b[:, :, 0]
    valid_pair = abs(det) > 0.23
    denom = np.where(valid_pair, det, 1.)
    xy = np.stack((offset[:, None] * b[:, :, 1] - offset[None, :] * a[:, :, 1],
                   offset[None, :] * a[:, :, 0] - offset[:, None] * b[:, :, 0]), axis=2) / denom[:, :, None]
    tri = _line_triples(len(lines))
    adjacent = np.roll(tri, -1, axis=1)
    tri = tri[valid_pair[tri, adjacent].all(1)]
    if not len(tri):
        return
    points = xy[tri, np.roll(tri, -1, axis=1)]
    corner = points.mean(1)
    spread = np.linalg.norm(points - corner[:, None], axis=2).max(1)
    distances = np.linalg.norm(lines[tri] - corner[:, None, None], axis=3)
    near = distances.min(2)
    good = (spread < 2.4) & (near < 2.8 + 0.14 * lengths[tri]).all(1)
    indices = np.flatnonzero(good)
    cost = spread + (near / lengths[tri]).mean(1)
    for i in indices[np.argsort(cost[indices])[:8]]:
        yield corner[i], lines[tri[i]]


def junction_pose(corner, segments, origin, k, dist, seed_depth):
    # Look down the bearing to the junction. The directions of the measured
    # lines in this local image are proportional to the first two components
    # of the three orthogonal 3D edges, regardless of their visible lengths.
    far = np.linalg.norm(segments - corner, axis=2).argmax(1)
    points = np.vstack((corner, segments[np.arange(3), far])) + origin
    uv = cv2.undistortPoints(points.astype(float), k, dist).reshape(4, 2)
    rays = np.column_stack((uv, np.ones(4)))
    axis = rays[0] / np.linalg.norm(rays[0])
    horizontal = np.cross([0., 1., 0.], axis)
    horizontal /= np.linalg.norm(horizontal)
    basis = np.column_stack((horizontal, np.cross(axis, horizontal), axis))
    projected = rays[1:] @ basis[:, :2]
    lengths = np.linalg.norm(projected, axis=1)
    if not np.isfinite(lengths).all() or lengths.min() < 1e-6:
        return None
    x, y = (projected / lengths[:, None]).T
    # The first two rows of a rotation are orthonormal. Solve for the squared
    # projected lengths of its columns from their three measured directions.
    system = np.stack((x * x, y * y, x * y))
    if abs(np.linalg.det(system)) < 0.10:
        return None
    weights = np.linalg.solve(system, [1., 1., 0.])
    if weights.min() < 0.04 or weights.max() > 1.000001:
        return None
    first, second = np.stack((x, y)) * np.sqrt(weights)
    depth_axes = np.cross(first, second)
    # To see all three edges, at least two faces at this corner must face the
    # camera. This resolves the reflected interpretation of the same 2D rays.
    if np.count_nonzero(depth_axes > 0) < 2:
        depth_axes *= -1
    axes = basis @ np.stack((first, second, depth_axes))
    shift = HALF * axes.sum(axis=1)
    pose = np.eye(4)
    pose[:3, 3] = axis * ((seed_depth - shift[2]) / axis[2]) + shift
    pose[:3, :3] = axes if np.linalg.det(axes) > 0 else axes[:, [1, 0, 2]]
    return pose
