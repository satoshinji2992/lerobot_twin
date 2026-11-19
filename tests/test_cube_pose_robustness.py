"""Exercise the pose path with rendered, metrically known cubes and mask proposals."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from twin_depot.cube_pose import FACE, HALF, VERTICES, camera_matrix, face_pose, infer_faces

K = np.array([[680.0, 0.0, 410.0], [0.0, 680.0, 240.0], [0.0, 0.0, 1.0]])
DIST = np.array([0.035, -0.025, 0.001, -0.002, 0.01])
CAMERA_ROTATION = cv2.Rodrigues(np.array([np.pi - 0.47, 0.0, 0.0]))[0]
RGB = {"red": (34, 39, 227), "green": (54, 204, 48), "blue": (207, 66, 39), "yellow": (33, 204, 213)}


def render_cube(image, label, x, y, lift=0.0, yaw=0.18):
    rotation = CAMERA_ROTATION @ cv2.Rodrigues(np.array([0.0, 0.0, yaw]))[0]
    translation = CAMERA_ROTATION @ [x, y, 29.0 + lift] + [0.0, 0.0, 382.0]
    coords = VERTICES @ rotation.T + translation
    projected = cv2.projectPoints(VERTICES, cv2.Rodrigues(rotation)[0], translation, K, DIST)[0].reshape(8, 2)
    faces = [
        ([4, 5, 6, 7], 0.40),
        ([0, 1, 5, 4], 0.56),
        ([1, 2, 6, 5], 0.75),
        ([2, 3, 7, 6], 0.53),
        ([3, 0, 4, 7], 0.66),
        ([0, 1, 2, 3], 1.0),
    ]
    # Render pixel-area coverage so the known 29 mm edge does not inherit the
    # one-pixel stroke that fillConvexPoly(..., LINE_AA) adds to a polygon.
    x, y, w, h = cv2.boundingRect(np.float32(projected))
    patch = image[y - 2 : y + h + 2, x - 2 : x + w + 2]
    hi = cv2.resize(patch, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    subpixels = (projected - [x - 2, y - 2] + 0.5) * 4 - 0.5
    for indices, shade in sorted(faces, key=lambda f: -coords[f[0], 2].mean()):
        cv2.fillConvexPoly(
            hi,
            np.int32(np.rint(16 * subpixels[indices])),
            tuple((np.array(RGB[label]) * shade).tolist()),
            shift=4,
        )
    patch[:] = cv2.resize(hi, (patch.shape[1], patch.shape[0]), interpolation=cv2.INTER_AREA)
    return translation + rotation @ [0.0, 0.0, -HALF], projected


def instances(crop):
    h, s, v = cv2.split(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV))
    masks = [(h < 10) | (h > 169), (h >= 36) & (h <= 88), (h >= 95) & (h <= 135), (h >= 17) & (h <= 35)]
    return [
        contour.reshape(-1, 2)
        for mask in masks
        for contour in cv2.findContours(
            np.uint8(mask & (s > 65) & (v > 30)), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )[0]
        if cv2.contourArea(contour) > 200
    ]


def frame():
    return np.full((480, 820, 3), 50, np.uint8)


@pytest.mark.parametrize("color,lift", [("green", 0.0), ("blue", 36.0), ("yellow", 22.0)])
def test_3d_positions_survive_rasterization_and_distortion(color, lift):
    image = frame()
    ref, _ = render_cube(image, "red", -90.0, 55.0)
    target, _ = render_cube(image, color, 78.0, -32.0, lift, yaw=-0.31)
    # Skin/printed red clutter is a colour proposal, but has no cube-shaped sides.
    cv2.ellipse(image, (460, 330), (30, 38), 18, 0, 360, (43, 50, 161), -1)
    image = cv2.GaussianBlur(image, (3, 3), 0.6)
    detections, msg = infer_faces(image, instances, K, DIST, [])
    assert msg == ""
    assert sorted(d[0] for d in detections) == sorted(["red", color])
    measured = {label: pose[:3, 3] - HALF * pose[:3, 2] for label, _, pose, _ in detections}
    assert measured["red"] == pytest.approx(ref, abs=5)
    assert measured[color] == pytest.approx(target, abs=5)
    red_pose = next(pose for label, _, pose, _ in detections if label == "red")
    assert red_pose[:3, 2] @ (measured[color] - measured["red"]) == pytest.approx(lift, abs=5)


def test_distant_instance_is_included_in_model_crop():
    image = frame()
    render_cube(image, "red", -143.0, 10.0)
    render_cube(image, "blue", 148.0, 10.0)
    detections, msg = infer_faces(image, instances, K, DIST, [])
    assert not msg
    assert {d[0] for d in detections} == {"red", "blue"}


def test_duplicate_reference_and_hidden_target_are_not_reported_as_valid_poses():
    image = frame()
    render_cube(image, "red", -54.0, 0.0)
    render_cube(image, "red", 60.0, 0.0)
    det, msg = infer_faces(image, instances, K, DIST, [])
    assert not det and "多个" in msg
    image = frame()
    render_cube(image, "red", -54.0, 0.0)
    _, pixels = render_cube(image, "green", 60.0, 0.0)
    min_xy = pixels.min(axis=0).astype(int) - 2
    right = int(pixels[:, 0].mean() + 6)
    bottom = int(pixels[:, 1].max() + 3)
    cv2.rectangle(image, min_xy, (right, bottom), (74, 74, 74), -1)
    det, msg = infer_faces(image, instances, K, DIST, [])
    assert not msg
    assert [d[0] for d in det] == ["red"]


def test_invalid_or_collapsed_face_has_no_pose():
    assert face_pose(np.ones((4, 2)), K, DIST) is None
    assert face_pose(np.full((4, 2), np.nan), K, DIST) is None


@pytest.mark.parametrize("tilt", [0.0, 0.005])
def test_overhead_view_survives_ippe_singularity(tilt):
    translation = np.array([0.0, 0.0, 450.0])
    pixel = cv2.projectPoints(FACE, np.array([np.pi - tilt, 0.0, 0.0]), translation, K, None)[0].reshape(4, 2)
    _, pose, error = face_pose(pixel, K, None)
    assert error < 1e-4
    assert pose[:3, 3] == pytest.approx(translation, abs=1e-4)
    assert pose[2, 2] < -0.99


def test_invalid_intrinsics_do_not_produce_metric_coordinates():
    with pytest.raises(ValueError):
        camera_matrix((480, 820, 3), {"image_size": [820, 480], "K": K, "dist": [float("nan")] * 5})


def test_green_cube_can_be_measured_in_camera_coordinates_without_reference():
    image = frame()
    target, _ = render_cube(image, "green", 35.0, -12.0, 24.0)
    poses, _ = infer_faces(image, instances, K, DIST, [], require_reference=False)
    assert [n for n, _, _, _ in poses] == ["green"]
    pose = poses[0][2]
    assert pose[:3, 3] - HALF * pose[:3, 2] == pytest.approx(target, abs=5)


def test_cross_face_pnp_uses_metric_nonplanar_vertices():
    from twin_depot.cuboid_pnp import CENTRED, silhouette_pose

    t = np.array([26.0, -18.0, 260.0])
    pixels = cv2.projectPoints(CENTRED, np.array([0.6, -0.5, 0.2]), t, K, DIST)[0]
    outline = np.int32(np.rint(cv2.convexHull(np.float32(pixels))))
    pts, pose, err = silhouette_pose(outline, K, DIST)
    assert len(pts) == 6 and err < 1.0
    assert pose[:3, 3] - HALF * pose[:3, 2] == pytest.approx(t, abs=3.0)


@pytest.mark.parametrize("color", RGB)
def test_fast_completion_refines_a_complete_coloured_cube(color):
    from twin_depot.cube_completion import Completion
    from twin_depot.cube_observer import FastObserver

    image = frame()
    expected, _ = render_cube(image, color, 42, 31)
    observer = FastObserver(Completion("cpu"))
    observer.observe(image, None, K, DIST, 0.0)
    (estimate,) = observer.estimates
    assert estimate["color"] == color and estimate["outline_refined"]
    assert estimate["camera_xyz"][:2] == pytest.approx(expected[:2], abs=3)
    assert estimate["camera_xyz"][2] == pytest.approx(expected[2], abs=15)
