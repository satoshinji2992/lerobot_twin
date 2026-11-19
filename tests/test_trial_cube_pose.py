import cv2
import numpy as np
import pytest
from twin_depot.cube_trial_observer import RescueObserver, TrialObserver, refine_outline
from twin_depot.cuboid_pnp import CENTRED, SYMMETRIES
from twin_depot.cube_outline_search import search_outline
from twin_depot.cube_image_edges import corner_junctions, face_quads, image_segments, junction_pose


def test_truncated_edges_measure_orientation_at_a_three_way_corner():
    # All three distant endpoints are hidden by different amounts. Only their
    # directions and their common visible vertex may be used for rotation.
    k = np.array([[620., 0., 200.], [0., 640., 150.], [0., 0., 1.]])
    dist = np.array([0.04, -0.01, 0.001, 0.002, 0.])
    t = np.array([48., -35., 310.])
    rv = np.array([0.45, -0.55, 0.25])
    rotation = cv2.Rodrigues(rv)[0]
    cam = CENTRED @ rotation.T + t
    near = np.linalg.norm(cam, axis=1).argmin()
    ends = cam[np.sum(CENTRED[near] != CENTRED, axis=1) == 1]
    def project(fractions):
        p = cam[near] + fractions[:, None] * (ends - cam[near])
        return cv2.projectPoints(p, np.zeros(3), np.zeros(3), k, dist)[0].reshape(-1, 2)
    lines = np.stack((project(np.full(3, 0.02)), project(np.array([0.41, 0.70, 0.60]))), axis=1)
    origin = np.array([100., 20.])
    (corner, segments), = corner_junctions(lines - origin)
    pose = junction_pose(corner, segments, origin, k, dist, t[2])
    cosine = max((np.trace(rotation.T @ pose[:3, :3] @ s) - 1) / 2 for s in SYMMETRIES)
    assert np.degrees(np.arccos(np.clip(cosine, -1, 1))) < 0.1
    assert np.linalg.norm(pose[:3, 3] - t) < 0.1
    assert not list(corner_junctions(lines[:2]))


def test_face_vertices_are_intersections_of_detected_image_segments():
    value = np.zeros((74, 80), np.uint8)
    mask = np.zeros_like(value)
    face = np.int32([[17, 18], [54, 14], [61, 48], [24, 53]])
    cv2.fillConvexPoly(mask, face, 255)
    cv2.fillConvexPoly(value, face, 150)
    quads = face_quads(image_segments(value, mask), mask)
    assert any(np.linalg.norm(q[:, None] - face, axis=2).min(1).max() < 2 for q, _ in quads)


def test_ridges_disambiguate_a_similar_off_axis_silhouette():
    k = np.array([[620., 0., 200.], [0., 640., 150.], [0., 0., 1.]])
    dist = np.array([0.06, -0.02, 0.001, 0.002, 0.])
    t = np.array([48., -35., 310.])
    rv = np.array([0.45, -0.55, 0.25])
    rotation = cv2.Rodrigues(rv)[0]
    pixels = cv2.projectPoints(CENTRED, rv, t, k, dist)[0].reshape(-1, 2)
    origin = np.floor(pixels.min(0) - 6).astype(int)
    pixels = np.int32(np.rint(pixels - origin))
    hull = cv2.convexHull(pixels)
    edge_map = np.ones(tuple((pixels.max(0) + 7)[::-1]), np.uint8)
    seed = np.eye(4)
    seed[:3, :3] = cv2.Rodrigues(np.pi * t / np.linalg.norm(t))[0] @ rotation
    seed[:3, 3] = t
    assert search_outline(hull, origin, seed, k, dist, np.full(edge_map.shape, 20.)) is None
    nearest = np.linalg.norm(CENTRED @ rotation.T + t, axis=1).argmin()
    for j in np.flatnonzero(np.sum(CENTRED[nearest] != CENTRED, axis=1) == 1):
        cv2.line(edge_map, pixels[nearest], pixels[j], 0, 1)
    pose = search_outline(hull, origin, seed, k, dist, cv2.distanceTransform(edge_map, cv2.DIST_L2, 3))
    cosine = max((np.trace(rotation.T @ pose[:3, :3] @ s) - 1) / 2 for s in SYMMETRIES)
    assert np.degrees(np.arccos(np.clip(cosine, -1, 1))) < 5
    assert np.linalg.norm(pose[:3, 3] - t) < 3


def test_image_outline_corrects_self_consistent_thirty_degree_error():
    k = np.array([[800., 0, 320], [0, 800, 360], [0, 0, 1]])
    truth = cv2.Rodrigues(np.array([0.35, -0.30, 0.10]))[0]
    seed = np.eye(4)
    seed[:3, :3] = truth @ cv2.Rodrigues(np.array([0., 0., np.pi / 6]))[0]
    seed[:3, 3] = [0, 0, 350]
    def project(r):
        return cv2.projectPoints(CENTRED, cv2.Rodrigues(r)[0], seed[:3, 3], k, None)[0].reshape(-1, 2)
    hsv = np.zeros((720, 640, 3), np.uint8)
    wrong = project(seed[:3, :3])
    assert refine_outline(hsv, "green", seed, wrong, k, None) is None
    cv2.fillConvexPoly(hsv, cv2.convexHull(project(truth).astype(np.int32)), (62, 220, 160))
    refined, pixels = refine_outline(hsv, "green", seed, wrong, k, None)
    cosine = max((np.trace(truth.T @ refined[:3, :3] @ s) - 1) / 2 for s in SYMMETRIES)
    assert np.degrees(np.arccos(np.clip(cosine, -1, 1))) < 3
    assert np.linalg.norm(refined[:3, 3] - seed[:3, 3]) < 4


def test_full_frame_pose_uses_metric_centre_and_rejects_invalid_geometry():
    camera = np.array([[510.0, 0, 180], [0, 510, 320], [0, 0, 1.0]])
    center = np.array([17.0, -12.0, 340.0])
    corners = cv2.projectPoints(CENTRED, np.array([0.5, -0.45, 0.1]), center, camera, None)[0].reshape(8, 2)
    pose, _ = TrialObserver.fit(corners, camera, None, 70)
    assert pose[:3, 3] == pytest.approx(center, abs=0.01)
    corners[0] += 150
    assert TrialObserver.fit(corners, camera, None, 70) is None


def test_rescue_requires_a_recent_primary_observation_and_current_yolo_evidence():
    cube = {"color": "green", "box": [80, 70, 40, 45], "camera_xyz": [5, -10, 330]}
    clutter = {**cube, "color": "yellow"}

    class Frames:
        def __init__(self, frames):
            self.frames = iter(frames)

        def observe(self, *args, **kwargs):
            self.estimates = next(self.frames)
            return [], [], ""

    observer = RescueObserver.__new__(RescueObserver)
    observer.last_at = 0.0
    observer.anchors = []
    observer.primary = Frames([[cube], [], [], []])
    observer.yolo = Frames([[cube, clutter], []])
    for stamp, expected in [(0.03, 1), (0.08, 1), (0.13, 0), (0.30, 0)]:
        observer.observe(None, None, None, None, stamp)
        assert len(observer.estimates) == expected
        if stamp == 0.08:
            assert observer.estimates[0]["rescued"] and observer.estimates[0]["color"] == "green"


@pytest.mark.parametrize(
    ("color", "hues"),
    [
        ("red", (177, 1)),
        ("green", (58, 62)),
        ("blue", (110, 113)),
        ("yellow", (25, 28)),
    ],
)
def test_matte_faces_can_differ_in_brightness_but_scattered_print_is_rejected(color, hues):
    hsv = np.empty((48, 48, 3), np.uint8)
    hsv[:, :24] = (hues[0], 230, 180)
    hsv[:, 24:] = (hues[1], 230, 55)
    pixels = np.array([[0, 0], [47, 0], [47, 47], [0, 47]])
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    assert TrialObserver.color_support(image, pixels, color) > 0.95
    image[:26] = (65, 105, 158)  # a finger across both faces
    assert 0.42 < TrialObserver.color_support(image, pixels, color) < 0.5
    printed = np.full_like(image, 205)
    printed[::2, ::2] = image[-1, ::2]
    assert TrialObserver.color_support(printed, pixels, color) < 0.01


def test_weak_pose_needs_unique_recent_yolo_match_and_same_frame_colour():
    observer = TrialObserver.__new__(TrialObserver)
    observer.reset()
    a = {
        "color": "green",
        "box": [10, 20, 32, 32],
        "camera_xyz": [0, 0, 350],
        "confidence": 0.85,
        "color_support": 0.75,
    }
    b = {**a, "box": [160, 20, 32, 32]}
    low = {**a, "confidence": 0.30, "color_support": 0.15}
    assert observer.verify([low], 1.00) == []
    assert observer.verify([a, b], 1.03) == [a, b]
    assert observer.verify([], 1.10) == []
    assert observer.verify([{**low, "color_support": 0.03}], 1.20) == []
    assert observer.verify([low, dict(low)], 1.33) == [low]
    assert observer.verify([low], 1.43) == []
    assert observer.verify([a], 2.00) == [a]
    assert observer.verify([low], 1.99) == []
