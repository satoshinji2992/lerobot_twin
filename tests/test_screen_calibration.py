"""Screen calibration should reject observations that look valid but give a bad camera."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from twin_depot.screen_calibration import ScreenCalibration


def target():
    board = np.full((360, 480), 245, np.uint8)
    for y in range(7):
        for x in range(10):
            if (x+y) % 2 == 0:
                board[(y+1)*40:(y+2)*40, (x+1)*40:(x+2)*40] = 10
    return board


def jpeg(im):
    ok, data = cv2.imencode(".jpg", im)
    assert ok
    return data.tobytes()


def test_rejects_motion_blur_duplicates_and_stream_switch(tmp_path):
    board = target()
    batch = ScreenCalibration(tmp_path)
    blurred = cv2.GaussianBlur(board, (0, 0), 3.2)
    assert "模糊" in batch.offer(jpeg(blurred), "phone")
    assert batch.report()["quality"]["blur_px"] > 3.5
    assert not batch.corners
    assert "已记录" in batch.offer(jpeg(board), "phone")
    assert "已采过" in batch.offer(jpeg(board), "phone")
    assert "对比太低" in batch.offer(jpeg((board*.1+100).astype(np.uint8)), "phone")
    with pytest.raises(ValueError, match="重连"):
        batch.offer(jpeg(board), "another-phone-session")
    assert len(list(tmp_path.glob("*.jpg"))) == 1


@pytest.fixture(scope="module")
def views(tmp_path_factory):
    batch = ScreenCalibration(tmp_path_factory.mktemp("calibration"))
    board = target()
    shape = (800, 600)
    k = np.array([[800., 0, 397.], [0, 803., 307.], [0, 0, 1.]])
    corners = np.float32([[-6, -4.5, 0], [6, -4.5, 0], [6, 4.5, 0], [-6, 4.5, 0]])
    src = np.float32([[0, 0], [480, 0], [480, 360], [0, 360]])
    rng = np.random.default_rng(42)
    for _ in range(240):
        rot = rng.uniform([-.55, -.6, -.3], [.55, .6, .3])
        pos = rng.uniform([-3, -2, 15], [3, 2, 22])
        dst = cv2.projectPoints(corners, rot, pos, k, None)[0].reshape(4, 2)
        if np.any(dst < [0, 0]) or np.any(dst > [799, 599]):
            continue
        im = cv2.warpPerspective(board, cv2.getPerspectiveTransform(src, dst), shape, borderValue=160)
        batch.offer(jpeg(im), "phone")
        if len(batch.corners) >= 20:
            break
    assert len(batch.corners) == 20
    return batch, k


def test_applied_model_predicts_independent_views(views):
    batch, k = views
    result = batch.solve()
    assert result["image_size"] == [800, 600]
    assert np.allclose(np.array(result["K"])[:2, :], k[:2, :], atol=5)
    metrics = result["validation"]
    assert metrics["heldout_images"] == [5, 10, 15, 20]
    assert metrics["train_views"] == 16
    assert metrics["heldout_p95_px"] < .5
    assert metrics["passed"]


def test_bad_holdout_is_rejected_even_when_training_fit_is_good(views, tmp_path):
    original, _ = views
    batch = ScreenCalibration(tmp_path)
    batch.size = original.size
    batch.corners = [p.copy() for p in original.corners]
    # Damage only held-out corners used to evaluate a pose, never the fitted intrinsics.
    batch.corners[9][1::2, 0, 1] += 7
    with pytest.raises(ValueError, match="留出图像"):
        batch.solve()
    assert batch.validation["fit_rms_px"] < .5
    assert not batch.validation["passed"]


def test_frontal_translations_do_not_establish_intrinsics(tmp_path):
    batch = ScreenCalibration(tmp_path)
    batch.size = (800, 600)
    grid = np.mgrid[:9, :6].T.reshape(-1, 1, 2).astype(np.float32)
    for i in range(15):
        batch.corners.append(grid*41 + np.float32([50 + (i%5)*90, 30 + (i//5)*120]))
    with pytest.raises((ValueError, cv2.error)):
        batch.solve()


@pytest.mark.parametrize("clockwise", [True, False])
def test_quarter_turn_migration_preserves_distorted_projections(tmp_path, clockwise):
    # Generate portrait measurements first, then rotate the measured pixels independently.
    k = np.array([[710., 0, 297.], [0, 734., 416.], [0, 0, 1.]])
    dist = np.array([.09, -.12, .007, -.004, 0.])
    old = dict(source="phone", camera_key="lens", native_size=[600, 800], video_size=[600, 800], zoom=1)
    profile = dict(old, native_size=[800, 600], video_size=[800, 600])
    model = dict(image_size=[600, 800], K=k.tolist(), dist=dist.tolist(), capture_profile=old)
    batch = ScreenCalibration(tmp_path)
    batch.size = (800, 600)
    grid = np.zeros((54, 3), np.float32)
    grid[:, :2] = np.mgrid[:9, :6].T.reshape(-1, 2)
    for x, y in [(-7, -9), (0, -9), (-7, 4), (0, 4)]:
        uv = cv2.projectPoints(grid, np.array([.3, -.17, .12]), np.array([x, y, 21.]), k, dist)[0]
        u, v = uv[..., 0], uv[..., 1]
        batch.corners.append(np.stack((799-v, u) if clockwise else (v, 599-u), axis=-1))
    result = batch.migrate(model, profile, clockwise)
    assert result["validation"]["heldout_rms_px"] < .001
    assert model["image_size"] == [600, 800]
    with pytest.raises(ValueError, match="镜头"):
        batch.migrate(model, dict(profile, zoom=2), clockwise)
    batch.corners[-1][1::2, 0, 0] += 8
    with pytest.raises(ValueError, match="验证未通过"):
        batch.migrate(model, profile, clockwise)
