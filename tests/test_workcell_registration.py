"""Geometry and acceptance gates for teaching the camera reference cube into the robot base."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from twin_depot.workcell_registration import preview_registration


def identity_example():
    return dict(top_center=[100., 200., 129.], x_point=[114.5, 200., 129.], y_point=[100., 214.5, 129.])


def test_recovers_tilted_pose_and_maps_fourth_point():
    cy, sy = math.cos(.8), math.sin(.8)
    cp, sp = math.cos(.3), math.sin(.3)
    rotation = np.array([[cy*cp, -sy, cy*sp], [sy*cp, cy, sy*sp], [-sp, 0., cp]])
    bottom = np.array([203., -50., 11.])
    def base(xyz):
        return (rotation @ xyz + bottom).tolist()
    data = dict(top_center=base([0, 0, 29]), x_point=base([45, 0, 29]), y_point=base([0, 60, 29]),
                check=dict(reference=[36, -20, 12], base=base([36, -20, 12])))
    preview = preview_registration(data)
    transform = np.array(preview["base_from_reference"])
    np.testing.assert_allclose(transform[:3, :3], rotation, atol=1e-12)
    np.testing.assert_allclose(transform[:3, 3], bottom)
    np.testing.assert_allclose(preview["axes"]["z"], rotation[:, 2])
    np.testing.assert_allclose(preview["bottom_center"], bottom)
    np.testing.assert_allclose(preview["lengths_mm"], [45., 60.])
    np.testing.assert_allclose(preview["check_predicted_base"], data["check"]["base"])
    assert preview["check_error_mm"] < 1e-9 and preview["can_apply"]
    assert preview["warnings"] == []


def test_perpendicular_projection_reports_skew_and_unvalidated_fit():
    data = identity_example()
    data["y_point"][0] += 1.
    info = preview_registration(data)
    expected_angle = math.degrees(math.atan2(14.5, 1.))
    assert info["angle_deg"] == pytest.approx(expected_angle)
    assert info["off_axis_mm"] == pytest.approx(1.)
    assert info["bottom_center"] == [100., 200., 100.]
    np.testing.assert_allclose(np.array(info["base_from_reference"])[:3,:3], np.eye(3))
    assert info["can_apply"] and info["check_error_mm"] is None
    assert any("尚未" in s for s in info["warnings"])


@pytest.mark.parametrize(("field", "value"), [
    ("x_point", [100,200,129]), ("y_point", [100,202,129]), ("y_point", [125,200,129]),
    ("y_point", [89,214.5,129]), ("top_center", [0,0,float("nan")]), ("x_point", [0,0,float("inf")]),
    ("x_point", [10**900,0,0]), ("y_point", [None, 1, 2]), ("top_center", [True,0,0]),
    ("x_point", [1,2]), ("x_point", "x"), ("tolerance_mm", 0), ("tolerance_mm", float("nan")),
    ("tolerance_mm", 101), ("tolerance_mm", True), ("check", {"reference":[0,0,0]})
])
def test_invalid_inputs_yield_readable_error(field, value):
    data = identity_example()
    data[field] = value
    with pytest.raises(ValueError):
        preview_registration(data)


def test_check_can_prevent_apply_and_equal_tolerance_can_apply():
    data = identity_example()
    data["check"] = dict(reference=[14.5,14.5,29.], base=[114.5,214.5,132.])
    bad = preview_registration(data)
    assert not bad["can_apply"] and bad["check_error_mm"] == pytest.approx(3.)
    data["tolerance_mm"] = 3.
    assert preview_registration(data)["can_apply"]


@pytest.mark.parametrize("ref,actual", [([0,0,29],[100,200,129]), ([14.5,14.5,29],[100.1,200,129]),
                                        ([14.5,0,29],[114.5,214.5,129])])
def test_fourth_point_is_independent(ref, actual):
    data = identity_example()
    data["check"] = dict(reference=ref, base=actual)
    with pytest.raises(ValueError, match="独立"):
        preview_registration(data)


def test_inverted_normal_warns_operator_and_preserves_right_handed_rotation():
    data = identity_example()
    data["x_point"], data["y_point"] = data["y_point"], data["x_point"]
    preview = preview_registration(data)
    assert preview["axes"]["z"] == [0.,0.,-1.]
    assert preview["bottom_center"] == [100.,200.,158.]
    assert any("负 Z" in s for s in preview["warnings"])
    assert np.linalg.det(np.array(preview["base_from_reference"])[:3,:3]) == pytest.approx(1.)
