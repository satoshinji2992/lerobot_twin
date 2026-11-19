import cv2
import numpy as np
import pytest
from twin_depot.cube_trial_observer import RescueObserver, TrialObserver
from twin_depot.cuboid_pnp import CENTRED


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
