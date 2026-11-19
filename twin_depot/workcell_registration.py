"""Fit the red cube reference frame to three TCP points in robot-base millimetres."""

import math

import numpy as np

CUBE_SIDE_MM = 29.0
MIN_BASELINE_MM = 8.0
MAX_SKEW_DEG = 8.0


def _point(value, label):
    # Reject booleans and blank/missing values rather than silently converting them to coordinates.
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{label}需要三个坐标分量（X、Y、Z），单位 mm")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in value):
        raise ValueError(f"{label}的三个坐标都应为数字")
    try:
        point = np.array(value, dtype=float)
    except OverflowError as exc:
        raise ValueError(f"{label}数值超出坐标范围") from exc
    if not np.isfinite(point).all():
        raise ValueError(f"{label}包含无效数值")
    return point


def preview_registration(data):
    """Return T_base_from_reference and residuals. The three teaching points lie on the top plane."""
    if not isinstance(data, dict):
        raise ValueError("配准输入应为 JSON 对象")
    top = _point(data.get("top_center"), "顶面中心")
    px = _point(data.get("x_point"), "+X 方向点")
    py = _point(data.get("y_point"), "+Y 方向点")
    tolerance = data.get("tolerance_mm", 2.)
    if isinstance(tolerance, bool) or not isinstance(tolerance, (float, int)) or not 0 < tolerance <= 100:
        raise ValueError("检查点允许误差应大于 0 且不超过 100 mm")

    dx, dy = px - top, py - top
    lx, ly = float(np.linalg.norm(dx)), float(np.linalg.norm(dy))
    if not np.isfinite([lx, ly]).all():
        raise ValueError("示教基线超出坐标范围")
    if min(lx, ly) < MIN_BASELINE_MM:
        raise ValueError(f"方向点距离顶面中心太近。每条示教基线至少 {MIN_BASELINE_MM:g} mm")
    x_axis = dx / lx
    cosine = np.clip(np.dot(x_axis, dy/ly), -1., 1.)
    angle = math.degrees(math.acos(cosine))
    if abs(angle-90.) > MAX_SKEW_DEG:
        raise ValueError(f"两方向夹角为 {angle:.1f}°，请重教约 90° 的 +X、+Y 点（容差 ±{MAX_SKEW_DEG:g}°）")
    # +X is the primary direction. Project the Y teaching vector onto its perpendicular.
    z_axis = np.cross(x_axis, dy/ly)
    z_axis /= np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis)
    rotation = np.column_stack((x_axis, y_axis, z_axis))
    bottom = top - CUBE_SIDE_MM*z_axis
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = bottom
    residual = float(abs(np.dot(x_axis, dy)))
    warnings = []
    if min(lx, ly) < CUBE_SIDE_MM:
        warnings.append("示教基线较短，小的触点误差会放大为角度误差。可使用与红块固定且同面的轴向延长标记。")
    if z_axis[2] < 0:
        warnings.append("参考 +Z 朝向基座负 Z。若红块放在水平桌面上，请核对 +X、+Y 的次序及正向。")

    check_error = None
    predicted = None
    check = data.get("check")
    if check is not None:
        if not isinstance(check, dict):
            raise ValueError("检查点需要参考系坐标和基座实测坐标")
        ref = _point(check.get("reference"), "检查点参考系")
        actual = _point(check.get("base"), "检查点基座系")
        predicted = rotation @ ref + bottom
        if not np.isfinite(predicted).all():
            raise ValueError("检查点超出坐标范围")
        if any(math.dist(point, p) < 3. for point in (predicted, actual) for p in (top, px, py)):
            raise ValueError("检查点与示教点重合或太近。请选取独立的第四个实物点")
        check_error = math.dist(predicted, actual)
        if not math.isfinite(check_error):
            raise ValueError("检查点距离超出坐标范围")
        if check_error > tolerance:
            warnings.append("独立检查点超过容差，请核对 TCP 偏置、底面原点和轴向，再重新示教。")
    else:
        warnings.append("尚未用独立点验证基座定位误差。")

    return dict(base_from_reference=transform.tolist(), bottom_center=bottom.tolist(),
                axes=dict(x=x_axis.tolist(), y=y_axis.tolist(), z=z_axis.tolist()),
                lengths_mm=[lx, ly], angle_deg=angle, off_axis_mm=residual,
                check_error_mm=check_error, check_predicted_base=predicted.tolist() if predicted is not None else None,
                tolerance_mm=float(tolerance), can_apply=check_error is None or check_error <= tolerance,
                warnings=warnings)
