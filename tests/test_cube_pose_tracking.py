import cv2
import numpy as np
import pytest
from twin_depot.cube_pose_tracking import RigidCubeTracker
from twin_depot.cuboid_pnp import CENTRED

K = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1.0]])


def cube(x=0.0, z=300.0, support=0.9, confidence=0.9):
    pose = np.eye(4)
    pose[:3, 3] = [x, 0, z]
    return {
        "color": "green",
        "camera_xyz": pose[:3, 3].tolist(),
        "camera_pose": pose.tolist(),
        "color_support": support,
        "confidence": confidence,
    }


def test_partial_cube_moves_laterally_without_shrinking_then_expires():
    tracker = RigidCubeTracker()
    for i in range(5):
        (last,) = tracker.update([cube()], i * 0.04, K, None)
    # The shrinking fragment at the same physical range produces a false PnP depth.
    (part,) = tracker.update([cube(12, 440, support=0.25)], 0.20, K, None)
    assert part["source"] == "depth_hold" and part["camera_xyz"][2] == pytest.approx(300.0)
    assert part["camera_xyz"][0] > 0 and part["track_id"] == last["track_id"]
    p = np.asarray(part["camera_pose"])
    pixels = cv2.projectPoints(CENTRED, cv2.Rodrigues(p[:3, :3])[0], p[:3, 3], K, None)[0].reshape(8, 2)
    assert np.allclose(pixels, part["pixels"])
    assert np.ptp(pixels[:, 1]) == pytest.approx(np.ptp(np.asarray(last["pixels"])[:, 1]))
    (ghost,) = tracker.update([], 0.26, K, None)
    assert ghost["measurement_index"] is None and ghost["source"] == "predicted"
    (hold,) = tracker.update([], 0.40, K, None)
    (hold2,) = tracker.update([], 0.48, K, None)
    assert hold["source"] == "held" and hold["camera_xyz"] == hold2["camera_xyz"]
    assert tracker.update([], 0.56, K, None) == []


def test_partial_observations_cannot_renew_depth_and_real_retreat_can_reacquire():
    tracker = RigidCubeTracker()
    for i in range(5):
        (first,) = tracker.update([cube()], i * 0.04, K, None)
    for i in range(5, 21):
        shown = tracker.update([cube(0, 420, support=0.24)], i * 0.04, K, None)
    assert shown == []
    for i in range(21, 30):
        shown = tracker.update([cube(0, 420)], i * 0.04, K, None)
    (p,) = shown
    assert p["track_id"] == first["track_id"] and p["source"] == "observed"
    assert p["camera_xyz"][2] == pytest.approx(420.0, abs=2.0)


def test_same_colour_tracks_remain_distinct_and_single_detections_leave_no_ghost():
    tracker = RigidCubeTracker()
    for i in range(5):
        obs = [cube(-36), cube(40)]
        left, right = tracker.update(obs, i * 0.04, K, None)
    shown = tracker.update([cube(40)], 0.2, K, None)
    assert [(p["track_id"], p["source"]) for p in shown] == [
        (left["track_id"], "predicted"),
        (right["track_id"], "observed"),
    ]
    tracker = RigidCubeTracker()
    tracker.update([cube()], 0, K, None)
    assert tracker.update([], 0.05, K, None) == []


def test_confident_small_patch_surrounded_by_a_hand_cannot_reinitialize_depth():
    tracker = RigidCubeTracker()
    for i in range(5):
        tracker.update([cube()], i * 0.04, K, None)
    small = cube(z=420.0)
    pixels = cv2.projectPoints(CENTRED, np.zeros(3), np.array(small["camera_xyz"]), K, None)[0]
    small["pixels"] = pixels.reshape(8, 2).tolist()
    image = np.full((480, 640, 3), (65, 105, 158), np.uint8)
    hull = cv2.convexHull(np.rint(pixels).astype(np.int32))
    cv2.fillConvexPoly(image, hull, (30, 145, 30))
    for i in range(5, 14):
        [pose] = tracker.update([small], i * 0.04, K, None, image)
        assert pose["source"] == "depth_hold" and pose["camera_xyz"][2] == pytest.approx(300)
    image[:] = (70, 70, 70)
    cv2.fillConvexPoly(image, hull, (30, 145, 30))
    for i in range(14, 22):
        [pose] = tracker.update([small], i * 0.04, K, None, image)
    assert pose["source"] == "observed" and pose["camera_xyz"][2] == pytest.approx(420)
