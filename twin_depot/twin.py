"""Mapping between LeRobot SO-101 motor readings and the MuJoCo twin's joint angles.

LeRobot's "new calibration" puts each joint's zero at the middle of its recorded range, which is
exactly the zero of ``so101_new_calib.xml``. So for the five arm joints:

    twin_rad = deg2rad(degrees_from_mid_range) * sign + offset

regardless of whether the arm runs with ``use_degrees=True`` (DEGREES) or the default
``RANGE_M100_100`` normalisation - we undo the normalisation with the arm's calibration file.
The gripper is ``RANGE_0_100`` (0 = closed, 100 = open) and is mapped linearly onto the jaw hinge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import layout

STS3215_MAX_RES = 4095  # resolution - 1, as used by LeRobot's normalisation


@dataclass
class TwinMapping:
    sign: dict[str, float] = field(default_factory=lambda: dict.fromkeys(layout.ARM_JOINTS, 1.0))
    offset_deg: dict[str, float] = field(default_factory=lambda: dict.fromkeys(layout.ARM_JOINTS, 0.0))
    gripper_closed_rad: float = layout.GRIPPER_CLOSED_RAD
    gripper_open_rad: float = layout.GRIPPER_OPEN_RAD

    def to_twin(self, arm_deg: dict[str, float], gripper_pct: float) -> np.ndarray:
        """Degrees-from-mid (5 arm joints) + gripper percent -> 6 MuJoCo joint targets (rad)."""
        q = np.zeros(6)
        for i, name in enumerate(layout.ARM_JOINTS):
            q[i] = math.radians(arm_deg[name] * self.sign[name] + self.offset_deg[name])
        g = min(100.0, max(0.0, gripper_pct)) / 100.0
        q[5] = self.gripper_closed_rad + g * (self.gripper_open_rad - self.gripper_closed_rad)
        return q

    def from_twin(self, q: np.ndarray) -> tuple[dict[str, float], float]:
        """Inverse of :meth:`to_twin` (used by the replay recorder and tests)."""
        arm = {
            n: (math.degrees(q[i]) - self.offset_deg[n]) * self.sign[n]
            for i, n in enumerate(layout.ARM_JOINTS)
        }
        pct = (q[5] - self.gripper_closed_rad) / (self.gripper_open_rad - self.gripper_closed_rad) * 100.0
        return arm, float(np.clip(pct, 0.0, 100.0))

    @staticmethod
    def parse_offsets(spec: str | None) -> dict[str, float]:
        """``"shoulder_lift=3,elbow_flex=-2.5"`` -> dict."""
        out = dict.fromkeys(layout.ARM_JOINTS, 0.0)
        if not spec:
            return out
        for item in spec.split(","):
            name, _, val = item.partition("=")
            name = name.strip()
            if name not in out:
                raise ValueError(f"unknown joint '{name}' in offsets; expected one of {layout.ARM_JOINTS}")
            out[name] = float(val)
        return out


def lerobot_to_degrees(
    value: float, norm_mode: str, range_min: int, range_max: int, drive_mode: int = 0
) -> float:
    """Undo LeRobot's MotorsBus normalisation and return degrees relative to the range middle.

    ``norm_mode`` is the MotorNormMode value: "degrees", "range_m100_100" or "range_0_100".
    Mirrors ``MotorsBus._normalize`` (lerobot/motors/motors_bus.py).
    """
    mid = (range_min + range_max) / 2
    span = range_max - range_min
    if norm_mode == "degrees":
        return float(value)
    if norm_mode == "range_m100_100":
        v = -value if drive_mode else value
        raw = (v + 100) / 200 * span + range_min
    elif norm_mode == "range_0_100":
        v = 100 - value if drive_mode else value
        raw = v / 100 * span + range_min
    else:
        raise ValueError(f"unsupported norm mode {norm_mode}")
    return (raw - mid) * 360 / STS3215_MAX_RES


def gripper_percent(
    value: float, norm_mode: str, range_min: int, range_max: int, drive_mode: int = 0
) -> float:
    """Gripper reading -> 0..100 % open, whatever the norm mode."""
    if norm_mode == "range_0_100":
        return float(100 - value if drive_mode else value)
    deg = lerobot_to_degrees(value, norm_mode, range_min, range_max, drive_mode)
    half_span_deg = (range_max - range_min) / 2 * 360 / STS3215_MAX_RES
    return float(np.clip((deg / half_span_deg + 1) * 50, 0, 100))


def bus_reading_to_twin_inputs(device, reading: dict[str, float]) -> tuple[dict[str, float], float]:
    """Convert a LeRobot ``{motor}.pos`` dict from a leader/follower into (arm degrees, gripper %).

    ``device`` is a connected ``SO101Follower`` / ``SO101Leader`` (anything with ``.bus`` and ``.calibration``).
    """
    arm: dict[str, float] = {}
    for name in layout.ARM_JOINTS:
        cal = device.calibration[name]
        mode = device.bus.motors[name].norm_mode.value
        arm[name] = lerobot_to_degrees(
            reading[f"{name}.pos"], mode, cal.range_min, cal.range_max, cal.drive_mode
        )
    cal = device.calibration["gripper"]
    mode = device.bus.motors["gripper"].norm_mode.value
    pct = gripper_percent(reading["gripper.pos"], mode, cal.range_min, cal.range_max, cal.drive_mode)
    return arm, pct
