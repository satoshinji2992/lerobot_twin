"""Single source of truth for the Twin Depot cell: every position, size, colour and rule constant.

World frame == SO-101 base frame. The workbench top is z = 0, the robot faces +x, and a
positive ``shoulder_pan`` swings the arm towards -y (clockwise seen from above).
"""

from __future__ import annotations

from dataclasses import dataclass

# --- robot ----------------------------------------------------------------------------------
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
ALL_JOINTS = ARM_JOINTS + ("gripper",)
PAN_AXIS_XY = (0.0388, 0.0)  # shoulder_pan axis in world, from the MJCF

# LeRobot gripper: 0 = fully closed, 100 = fully open (RANGE_0_100). Map onto the MJCF hinge.
GRIPPER_CLOSED_RAD = -0.10
GRIPPER_OPEN_RAD = 1.70

# Grasp centre in the TCP ("gripperframe" site) frame: x = approach axis, +z = moving-jaw side.
GRASP_OFFSET_IN_TCP = (-0.012, 0.0, 0.016)

# --- workbench / hall -----------------------------------------------------------------------
TABLE_SIZE = (1.10, 1.30)  # x, y
TABLE_CENTER = (0.10, 0.12)
TABLE_HEIGHT = 0.76  # floor is at z = -TABLE_HEIGHT
FLOOR_Z = -TABLE_HEIGHT
HALL_SIZE = (26.0, 18.0)
HALL_HEIGHT = 7.0
PARK_ORIGIN = (0.0, 0.0, -6.0)  # hidden "basement" shelf for pooled parcels


# --- conveyor -------------------------------------------------------------------------------
@dataclass(frozen=True)
class Conveyor:
    """One long straight belt that runs past the robot towards -y and never stops.

    The inbound AGV unloads at the upstream end (``y_in``); parcels ride a fast infeed section to
    the metering gate, which releases them one at a time onto the slow pick section in front of
    the robot; whatever is not picked rides on (fast outfeed) and drops off the downstream end
    (``y_out``) into the recycling cage on the floor.
    """

    x: float = 0.145
    width: float = 0.07
    top_z: float = 0.030
    rail_height: float = 0.014
    y_in: float = 1.55  # upstream end (inbound AGV dock)
    y_out: float = -1.25  # downstream end: drops into the recycling cage
    gate_y: float = 0.34  # metering gate: the infeed queue waits here
    pick_hi: float = 0.32  # slow pick section between pick_lo and pick_hi
    pick_lo: float = -0.24
    reach_hi: float = 0.21  # the jaws reach across the belt between these y
    reach_lo: float = -0.19
    last_chance: float = 0.05  # red markings: the end of the reach window
    fast_speed: float = 0.24  # infeed / outfeed speed (m/s)
    queue_pitch: float = 0.05  # parcel spacing in the infeed queue
    queue_max: int = 6  # the AGV only unloads while the infeed queue is shorter
    tunnel_y: tuple[float, float] = (0.42, 0.70)  # metering tunnel over the infeed (on the table)


CONVEYOR = Conveyor()


@dataclass(frozen=True)
class Cage:
    """Recycling cage (roll container) on the floor under the downstream end of the belt."""

    center: tuple[float, float] = (0.145, -1.55)
    half: tuple[float, float] = (0.28, 0.28)
    height: float = 0.50
    capacity: int = 4  # when this many parcels are in it, the tug hauls it away


CAGE = Cage()


# --- bins -----------------------------------------------------------------------------------
@dataclass(frozen=True)
class BinSpec:
    key: str
    name_en: str
    name_zh: str
    rgba: tuple[float, float, float, float]
    center: tuple[float, float]  # a row on the far side of the belt
    inner: float = 0.110  # roomy: the SO-101 jaws open wide across it when letting go
    wall: float = 0.005
    height: float = 0.045


BINS: tuple[BinSpec, ...] = (
    BinSpec("express", "EXPRESS", "加急件", (0.90, 0.16, 0.14, 1.0), (0.275, 0.1875)),
    BinSpec("standard", "STANDARD", "普通件", (0.16, 0.42, 0.92, 1.0), (0.275, 0.0625)),
    BinSpec("fragile", "FRAGILE", "易碎件", (0.12, 0.72, 0.36, 1.0), (0.275, -0.0625)),
    BinSpec("qc", "QC / UNKNOWN", "异常件", (0.62, 0.30, 0.86, 1.0), (0.275, -0.1875)),
)
BIN_BY_KEY = {b.key: b for b in BINS}


# --- scanner gate (for mystery parcels) -----------------------------------------------------
@dataclass(frozen=True)
class Scanner:
    center: tuple[float, float] = (0.0, 0.19)  # on the robot's left, clear of the belt
    half: float = 0.040
    z_range: tuple[float, float] = (0.015, 0.140)
    dwell_s: float = 0.8  # hold a parcel inside the beam this long


SCANNER = Scanner()


# --- call buttons (pressed with the arm: the twin pushes a sprung cap down) ---------------------
@dataclass(frozen=True)
class ButtonSpec:
    key: str  # "agv" | "tug"
    label: str
    name_zh: str
    rgba: tuple[float, float, float, float]
    center: tuple[float, float]  # on the robot's right, opposite the scanner
    radius: float = 0.020
    base_h: float = 0.020  # housing height; the cap sits on top
    travel: float = 0.008
    trigger: float = 0.004  # pressed this deep = a press


BUTTONS: tuple[ButtonSpec, ...] = (
    ButtonSpec("agv", "CALL AGV", "呼叫供件车", (0.15, 0.85, 0.35, 1.0), (0.000, -0.200)),
    ButtonSpec("tug", "CALL TUG", "呼叫牵引车", (0.20, 0.55, 1.00, 1.0), (0.065, -0.200)),
)
BUTTON_BY_KEY = {b.key: b for b in BUTTONS}


# --- parcels --------------------------------------------------------------------------------
@dataclass(frozen=True)
class ParcelType:
    key: str
    name_en: str
    name_zh: str
    rgba: tuple[float, float, float, float]
    size: tuple[float, float, float]  # full extents (m)
    mass: float
    dest: str | None  # bin key; None = hidden until scanned (mystery)
    points: int
    pool: int  # how many instances exist in the model
    weight: float  # spawn probability weight
    gentle: bool = False  # must be lowered into the bin softly
    timer_s: float | None = None  # expires if not sorted in time


PARCEL_TYPES: tuple[ParcelType, ...] = (
    ParcelType(
        "express",
        "EXPRESS",
        "加急件",
        (0.93, 0.22, 0.18, 1),
        (0.030, 0.030, 0.030),
        0.030,
        "express",
        100,
        6,
        30,
    ),
    ParcelType(
        "standard",
        "STANDARD",
        "普通件",
        (0.20, 0.48, 0.95, 1),
        (0.032, 0.032, 0.028),
        0.035,
        "standard",
        60,
        6,
        30,
    ),
    ParcelType(
        "fragile",
        "FRAGILE",
        "易碎件",
        (0.30, 0.85, 0.50, 1),
        (0.028, 0.028, 0.032),
        0.025,
        "fragile",
        120,
        4,
        18,
        gentle=True,
    ),
    ParcelType(
        "vip",
        "VIP GOLD",
        "金色VIP",
        (1.00, 0.78, 0.10, 1),
        (0.028, 0.028, 0.028),
        0.030,
        "express",
        300,
        2,
        9,
        timer_s=25.0,
    ),
    ParcelType(
        "mystery", "MYSTERY", "神秘件", (0.45, 0.45, 0.50, 1), (0.031, 0.031, 0.031), 0.030, None, 150, 4, 13
    ),
)
PARCEL_TYPE_BY_KEY = {p.key: p for p in PARCEL_TYPES}
MYSTERY_TRUE_DESTS = ("express", "standard", "fragile")

# --- rules ----------------------------------------------------------------------------------
QC_SAFE_POINTS = 30  # an unscanned mystery parcel dropped in QC
WRONG_BIN_PENALTY = -40
DROP_PENALTY = -25
STRAY_PENALTY = -10
CRACK_PENALTY = -50
VIP_EXPIRED_PENALTY = -80
CRACK_SPEED = 0.70  # m/s impact speed that cracks a fragile parcel (~2.5 cm free fall)
MISS_PENALTY = -30  # rode off the end of the belt into the recycling cage
SPILL_PENALTY = -50  # ... while the cage was away: it hit the floor
CLUTCH_BONUS = 25  # picked inside the last-chance stretch
COMBO_STEP = 3  # every N clean sorts raises the multiplier
MAX_MULTIPLIER = 4

SYNC_TOLERANCE_DEG = 12.0
SYNC_HOLD_S = 1.0

DIFFICULTY = {
    # metering-gate release interval start->end (s), pick-section speed start->end (m/s)
    "chill": {"interval": (9.0, 7.0), "belt": (0.035, 0.045)},
    "normal": {"interval": (7.5, 5.5), "belt": (0.045, 0.060)},
    "hard": {"interval": (5.5, 4.0), "belt": (0.060, 0.080)},
}

RANKS = (  # (min score for a 150 s shift, letter, english title, chinese title)
    (5000, "S", "LEGENDARY SORTER", "传奇分拣官"),
    (3000, "A", "GOLD PICKER", "金牌拣货员"),
    (1500, "B", "SKILLED HAND", "熟练工"),
    (-(10**9), "C", "TRAINEE", "实习生"),
)


def rank_for(score: int, shift_s: float = 150.0) -> tuple[str, str, str]:
    scale = shift_s / 150.0
    for threshold, letter, en, zh in RANKS:
        if score >= threshold * scale:
            return letter, en, zh
    raise AssertionError("unreachable")


# --- Isaac Sim warehouse backdrop (--hall isaac) ----------------------------------------------
# Where the work cell stands inside Simple_Warehouse/full_warehouse.usd (USD metres): the cell
# origin goes to ``anchor`` and the robot faces along ``yaw_deg`` (USD frame). Props inside the
# keep-out box (twin frame) are dropped. Changing this needs a re-import (the meshes are baked).
# Default: on the open floor in front of the five rack rows, the operator looking down the aisles,
# the AGV loop running between the roof pillars (x = -10.5 / 0.5 in USD).
ISAAC_PLACEMENT = {"anchor": (-6.0, 1.5), "yaw_deg": -90.0, "keepout": (-5.2, 6.4, -4.4, 3.4)}


# --- vehicles ---------------------------------------------------------------------------------
AGV_SPEED, TUG_SPEED, FORKLIFT_SPEED = 1.1, 0.9, 1.3  # m/s
AGV_BATCH = 6  # parcels per supply trip
AGV_UNLOAD_S = 0.9
AGV_HALF_LEN = 0.55
FORKLIFT_DWELL_S = 2.0

# Stations in the twin frame (x, y, yaw deg): where the AGV loads, the tug empties the cage, where
# they park, and the forklift's patrol. Routes between them are planned around the scenery.
STATIONS = {
    "builtin": {
        "supply": (2.6, 3.2, 90.0),
        "returns": (-3.0, -3.3, -90.0),
        "agv_park": (-1.2, 3.0, 0.0),
        "tug_park": (2.4, -2.4, 180.0),
        "forklift_park": (-8.0, 3.0, 0.0),
        "forklift_route": [(7.5, 3.0, -90.0), (7.5, -3.2, 180.0), (-8.0, -3.2, 90.0), (-8.0, 3.0, 0.0)],
    },
    "isaac": {  # rack rows at x >= 7 (twin frame), aisles at y ~ 7 / 2 / -3 / -8
        "supply": (10.0, 7.1, 0.0),
        "returns": (9.5, 2.1, 0.0),
        "agv_park": (3.0, 4.0, 0.0),
        "tug_park": (2.5, -3.6, 180.0),
        "forklift_park": (9.3, -9.2, 90.0),  # where full_warehouse.usd parks it
        "forklift_route": [(5.5, -8.0, 90.0), (5.5, 11.5, -90.0), (5.5, -8.0, 0.0), (9.5, -8.0, 180.0)],
    },
}


def infeed_dock() -> tuple[float, float, float]:
    """Inbound AGV pose (x, y, yaw deg) with its deck against the upstream end of the belt."""
    return CONVEYOR.x, CONVEYOR.y_in + AGV_HALF_LEN + 0.03, -90.0
