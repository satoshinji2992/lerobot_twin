"""Perspective, occlusion, frame lifetime and reacquisition regressions."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from twin_depot.cube_measure import Plane, project
from twin_depot.cube_tracking import CubeTracker


def scene():
    frame = np.full((480, 640, 3), 205, np.uint8)
    corners = np.float32([(82, 60), (540, 110), (475, 405), (105, 345)])
    for x, y in corners.astype(int):
        cv2.drawMarker(frame, (x, y), (0, 0, 0), markerSize=19, thickness=3)
    return frame, corners


def test_perspective_center_and_reference_loss():
    frame, corners = scene()
    plane = Plane(frame, corners, 190, 120)
    face = cv2.boxPoints(((83, 57), (29, 29), 27))
    # Two masks of the same top and a long rectangular distractor.
    polys = [project(q, plane.to_px) for q in (face, face + 0.25, face * [2, 0.5])]
    cubes = plane.cubes(polys)
    assert len(cubes) == 1
    assert (cubes[0]["x_mm"], cubes[0]["y_mm"]) == pytest.approx((83, 57), abs=0.3)
    assert plane.matches(frame)
    frame[47:75, 69:95] = 150
    assert not plane.matches(frame)


def test_cube_reference_sees_neighbor_outside_its_own_face():
    frame = np.full((480, 640, 3), 180, np.uint8)
    corners = np.float32([(220, 160), (285, 165), (280, 225), (225, 222)])
    for p in corners.astype(int):
        cv2.drawMarker(frame, p, (0, 0, 0), markerSize=19, thickness=3)
    plane = Plane(frame, corners, None, None, cube_reference=True)
    face = np.float32([[0, 0], [29, 0], [29, 29], [0, 29]])
    polys = [project(face+offset, plane.to_px) for offset in (0, [-42, 0], [180, 0])]
    cubes = plane.cubes(polys)
    assert len(cubes) == 1
    assert (cubes[0]["x_mm"], cubes[0]["y_mm"]) == pytest.approx((-27.5, 14.5))
    assert plane.crop[0] < polys[1][:, 0].min()


def test_measurement_expires_and_occlusion_preserves_only_history(monkeypatch):
    pytest.importorskip("cryptography")
    from twin_depot import vision

    frame, corners = scene()
    t = [100.]
    monkeypatch.setattr(vision.time, "monotonic", lambda: t[0])
    visible = True
    def model(_, p):
        return [project([[40, 40], [69, 40], [69, 69], [40, 69]], p.to_px)] if visible else []

    dev = vision.Instrument(model)
    def feed(img):
        seq = dev.ticket(dev.session)
        dev.receive(cv2.imencode(".jpg", img)[1].tobytes(), dev.session, seq)
        return seq

    feed(frame)
    dev.calibrate(dict(id=dev.freeze()["id"], points=corners, width=200, height=130))
    for i in range(3):
        t[0] += 0.2
        seq = feed(frame)
        assert dev.state(None)["cubes"][0]["stable"] == (i == 2)
    dev.hold(dict(seq=seq, session=dev.session, index=0))
    visible = False
    feed(frame)
    assert dev.state(None)["cubes"] == []
    assert len(dev.state(None)["predictions"]) == 1
    assert dev.state(None)["held"]["x_mm"] == 54.5
    visible = True
    t[0] += 0.2
    feed(frame)
    assert not dev.state(None)["cubes"][0]["stable"]
    t[0] += vision.FRESH_S + 1
    assert not dev.state(None)["fresh"] and dev.state(None)["cubes"] == []
    assert dev.state(None)["predictions"] == []
    seq = dev.ticket(dev.session)
    dev.start()
    with pytest.raises(ValueError):
        dev.receive(b"", "old-session", seq)


def test_tracks_keep_identity_at_variable_fps_and_bridge_a_short_gap():
    tracker = CubeTracker()
    ids = {}
    for i, t in enumerate([0, 0.09, 0.23, 0.31, 0.46, 0.55, 0.68, 0.8]):
        cubes = [dict(x_mm=x+35*t, y_mm=y) for x, y in [(30, 40), (10, 90)]]
        if i % 2:
            cubes.reverse()
        assert tracker.update(cubes, t) == []
        for c in cubes:
            assert c["track_id"] == ids.setdefault(c["y_mm"], c["track_id"])
            assert not c["stable"]
    predictions = tracker.update([], 1.0)
    assert len(predictions) == 2
    for pred in predictions:
        initial = 30 if pred["track_id"] == ids[40] else 10
        assert pred["fused_x_mm"] == pytest.approx(initial+35, abs=1)
        assert pred["vx_mm_s"] == pytest.approx(35, abs=2)
    assert tracker.update([], 1.17) == []
    c = dict(x_mm=30+35*1.21, y_mm=40)
    tracker.update([c], 1.21)
    assert c["track_id"] == ids[40] and not c["stable"]
    tracker.reset()
    assert tracker.update([], 1.3) == []


def test_slow_camera_can_lock_static_cube_but_cannot_coast_for_a_second():
    tracker = CubeTracker()
    for t in (0, 0.28, 1.30, 1.58):
        cube = dict(x_mm=50., y_mm=60.)
        tracker.update([cube], t)
        assert cube["track_id"] == 1
    assert cube["stable"]
    assert tracker.update([], 2.58) == []
