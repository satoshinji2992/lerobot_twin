import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from twin_depot.workcell_tracking import HISTORY_S, Tracker


def cube(x, color="green", y=0.0, z=14.5, **extra):
    return dict(color=color, xyz=[x, y, z], **extra)


def test_preserves_rows_and_smooths_three_axes_without_mutating_input():
    tracker = Tracker()
    rng = np.random.default_rng(96)
    raw, fused, ids = [], [], set()
    for i, noise in enumerate(rng.normal(0, 1.6, (65, 3))):
        xyz = np.array([30, 50, 32]) + noise
        row = dict(color="yellow", xyz=xyz, error_px=0.27, label="tray")
        [result], history = tracker.update([row], at=i * 0.04)
        assert result["raw_xyz"] == xyz.tolist()
        assert result["error_px"] == 0.27 and result["label"] == "tray"
        assert "track_id" not in row and not history
        ids.add(result["track_id"])
        raw.append(result["raw_xyz"])
        fused.append(result["xyz"])
    assert len(ids) == 1
    assert np.all(np.std(fused[12:], axis=0) < np.std(raw[12:], axis=0) * 0.88)


def test_same_colour_assignment_uses_velocity_and_is_independent_of_input_order():
    tracker = Tracker()
    ids = {}
    for i in range(17):
        a = cube(-64 + 8 * i, y=-4, label="a")
        b = cube(64 - 8 * i, y=4, label="b")
        c = cube(64 - 8 * i, color="blue", y=4, label="c")
        rows = [a, b, c] if i % 2 else [c, b, a]
        live, missing = tracker.update(rows, at=i * 0.08)
        assert not missing
        assert [r["label"] for r in live] == [r["label"] for r in rows]
        for row in live:
            assert row["track_id"] == ids.setdefault(row["label"], row["track_id"])
    assert len(set(ids.values())) == 3


def test_crossing_after_missing_frame_does_not_swap_identity():
    tracker = Tracker()
    for i in range(6):
        live, _ = tracker.update([cube(-60 + i * 8), cube(60 - i * 8)], i * 0.08)
    a, b = (r["track_id"] for r in live)
    assert tracker.update([], 0.55)[0] == []
    # A further observation arrives after the cubes have passed in X.
    live, missing = tracker.update([cube(-12), cube(12)], 0.72)
    assert [r["track_id"] for r in live] == [b, a]
    assert not missing


def test_outlier_is_withheld_and_persistent_new_cube_acquires_its_own_id():
    tracker = Tracker()
    [first], _ = tracker.update([cube(0)], 2.0)
    ident = first["track_id"]
    current, past = tracker.update([cube(280)], 2.08)
    assert current == []
    assert past == [dict(track_id=ident, color="green", xyz=[0.0, 0.0, 14.5], missing_ms=80)]
    [again], past = tracker.update([cube(1)], 2.16)
    assert again["track_id"] == ident and past == []
    current, _ = tracker.update([cube(1), cube(80)], 2.24)
    assert len(current) == 1
    current, _ = tracker.update([cube(81), cube(1)], 2.32)
    assert [r["track_id"] == ident for r in current] == [False, True]


def test_history_is_last_fused_sighting_not_predicted_motion_and_expires():
    tracker = Tracker()
    for i in range(6):
        [last], _ = tracker.update([cube(6 * i, z=22 + i)], i * 0.1)
    live, past = tracker.update([], 0.8)
    assert live == []
    assert past == [dict(track_id=last["track_id"], color="green", xyz=last["xyz"], missing_ms=300)]
    [recovered], past = tracker.update([cube(48, z=30)], 0.82)
    assert past == [] and recovered["track_id"] == last["track_id"]
    assert tracker.update([], 0.83 + HISTORY_S) == ([], [])
    [new], _ = tracker.update([cube(48)], 0.9 + HISTORY_S)
    assert new["track_id"] != last["track_id"]


def test_ambiguous_measurement_and_overlapping_masks_never_spawn_two_locks():
    tracker = Tracker()
    pair, _ = tracker.update([cube(-11), cube(11)], 0)
    live, past = tracker.update([cube(0)], 0.1)
    assert not live and {r["track_id"] for r in past} == {r["track_id"] for r in pair}
    live, _ = tracker.update([cube(10, error_px=0.7), cube(-11), cube(11, error_px=0.2)], 0.2)
    assert len(live) == 2 and live[-1]["raw_xyz"][0] == 11
    assert [r["track_id"] for r in live] == [r["track_id"] for r in pair]


def test_late_frames_and_invalid_coordinates_do_not_damage_active_tracks():
    tracker = Tracker()
    [first], _ = tracker.update([cube(20), cube(np.nan)], 10)
    assert tracker.update([cube(200)], 9.9) == ([], [])
    assert tracker.update([cube(200)], 10) == ([], [])
    with pytest.raises(ValueError):
        tracker.update([], float("nan"))
    [again], _ = tracker.update([cube(21)], 10.1)
    assert again["track_id"] == first["track_id"]


def test_short_extrapolation_is_separate_and_expires():
    tracker=Tracker()
    for i in range(7):
        rows,history=tracker.update([cube(i*2.)],1.+i*.04)
    measured=rows[0]["xyz"].copy()
    tracker.compensate(rows,history,1.30)
    assert rows[0]["xyz"]==measured and rows[0]["lead_xyz"][0]>measured[0]
    assert rows[0]["lead_ms"]==60
    rows,history=tracker.update([],1.40)
    tracker.compensate(rows,history,1.41)
    assert not rows and "lead_xyz" not in history[0]
