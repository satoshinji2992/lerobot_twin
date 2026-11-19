"""Recover cube orientation from a convex silhouette and its visible interior edges."""

from functools import lru_cache

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .cuboid_pnp import CENTRED

_ANGLES = np.arange(16) * np.pi / 16
_DIRECTIONS = np.stack((np.cos(_ANGLES), np.sin(_ANGLES)), axis=1)
_NEIGHBOURS = np.array([np.flatnonzero(np.sum(v != CENTRED, axis=1) == 1) for v in CENTRED])
_SAMPLES = np.linspace(0.16, 0.84, 9)


@lru_cache(maxsize=1)
def _templates():
    # Widths of a projected cube along a 2D direction are the sum of the
    # absolute widths of its three orthogonal edges. Scale cancels in the index.
    rotations = Rotation.random(16000, random_state=1023).as_matrix()
    widths = np.abs(_DIRECTIONS @ rotations[:, :2]).sum(axis=2)
    return rotations, cKDTree(widths / widths.mean(axis=1, keepdims=True))


def _bounds(points):
    supports = points @ _DIRECTIONS.T
    return supports.min(axis=-2), supports.max(axis=-2)


def _camera_candidates(hull, k, dist):
    rays = cv2.undistortPoints(hull.astype(float), k, dist).reshape(-1, 2)
    if not np.isfinite(rays).all():
        return None
    axis = np.r_[0.5 * (rays.max(0) + rays.min(0)), 1.0]
    axis /= np.linalg.norm(axis)
    horizontal = np.cross([0., 1., 0.], axis)
    horizontal /= np.linalg.norm(horizontal)
    basis = np.stack((horizontal, np.cross(axis, horizontal), axis)).T
    local = np.column_stack((rays, np.ones(len(rays)))) @ basis
    if local[:, 2].min() <= 0:
        return None
    lo, hi = _bounds(local[:, :2] / local[:, 2:])
    widths = hi - lo
    if widths.min() < 1e-9:
        return None
    rotations, index = _templates()
    _, ids = index.query(widths / widths.mean(), k=16)
    r = rotations[ids]
    # The orthographic silhouette is unchanged by a half-turn around the
    # viewing ray. Its interior ridges generally change sides.
    r = np.concatenate((r, r * np.array([-1., -1., 1.])[None, :, None]))
    vertices = CENTRED @ r.transpose(0, 2, 1)
    a, b = _bounds(vertices[:, :, :2])
    depth = (b - a).mean(1) / widths.mean()
    xy = np.zeros((len(r), 2))
    for _ in range(2):
        projected = (vertices[:, :, :2] + xy[:, None]) / (vertices[:, :, 2:] + depth[:, None, None])
        a, b = _bounds(projected)
        # Opposing supports estimate range separately from image translation.
        xy += ((lo + hi - a - b) @ _DIRECTIONS / len(_DIRECTIONS)) * depth[:, None]
        depth *= (b - a).mean(1) / widths.mean()
    return basis @ r, np.column_stack((xy, depth)) @ basis.T


def search_outline(contour, origin, pose, k, dist, distances):
    hull = cv2.convexHull(contour)
    area = cv2.contourArea(hull)
    if area < 180 or cv2.contourArea(contour) < 0.84 * area or distances.min() >= 3:
        return None
    candidates = _camera_candidates(hull + origin.reshape(1, 1, 2), k, dist)
    if candidates is None:
        return None
    rotations, translations = candidates
    rotations = np.concatenate((pose[None, :3, :3], rotations))
    translations = np.vstack((pose[:3, 3], translations))
    vertices = CENTRED @ rotations.transpose(0, 2, 1) + translations[:, None]
    projected = cv2.projectPoints(vertices.reshape(-1, 3), np.zeros(3), np.zeros(3), k, dist)[0]
    projected = projected.reshape(-1, 8, 2) - origin
    lo, hi = _bounds(hull.reshape(-1, 2))
    a, b = _bounds(projected)
    boundary = np.sqrt(((a - lo) ** 2 + (b - hi) ** 2).mean(1) / 2)

    # Rank both interpretations using the observed ridges inside the mask.
    rows = np.arange(len(vertices))
    nearest = np.linalg.norm(vertices, axis=2).argmin(1)
    start = projected[rows, nearest][:, None]
    delta = projected[rows[:, None], _NEIGHBOURS[nearest]] - start
    samples = start[:, :, None] + delta[:, :, None] * _SAMPLES[None, :, None]
    x, y = np.moveaxis(np.rint(samples).astype(int), -1, 0)
    y = np.clip(y, 0, distances.shape[0] - 1)
    x = np.clip(x, 0, distances.shape[1] - 1)
    support = np.maximum(0., 1. - distances[y, x] / 3.).mean(2)
    support[np.linalg.norm(delta, axis=2) < 6] = 0
    ridges = np.sort(support, axis=1)[:, -2:].mean(1)
    tolerance = max(1.4, 0.032 * np.sqrt(area))
    cost = boundary / tolerance - 1.8 * ridges
    eligible = ((abs(translations[:, 2] / pose[2, 3] - 1) < 0.15)
                & (boundary < tolerance) & (ridges > 0.30))
    eligible[0] = False
    best = np.where(eligible, cost, np.inf).argmin()
    if not eligible[best] or cost[best] + 0.22 >= cost[0]:
        return None
    candidate = np.eye(4)
    candidate[:3, :3], candidate[:3, 3] = rotations[best], translations[best]
    return candidate
