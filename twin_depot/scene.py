"""Builds the Twin Depot MJCF: a night-shift warehouse hall, a workbench sorting cell and the SO-101 twin.

The official SO-101 MJCF is merged in (not copied by hand) so its kinematics and STS3215 actuator
tuning stay exactly as published. Everything else is generated from ``layout.py``.
"""

from __future__ import annotations

import copy
import math
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from . import layout
from .assets import ensure_so101_assets
from .textures import build_textures

BUILD_DIR = Path(__file__).resolve().parents[1] / "build"

N_SLATS = 11
SLAT_PITCH = 0.055  # N_SLATS * SLAT_PITCH covers the pick section, so the wrap-around is seamless
VEHICLES = ("agv", "tug", "forklift")


def _f(*vals: float) -> str:
    return " ".join(f"{v:.5g}" for v in vals)


def _quat_to_mat(q) -> np.ndarray:
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, dtype=float))
    return m.reshape(3, 3)


def _yaw_quat(deg: float) -> str:
    a = math.radians(deg) / 2
    return _f(math.cos(a), 0, 0, math.sin(a))


# --------------------------------------------------------------------------------------------
# robot
# --------------------------------------------------------------------------------------------
def _robot_parts(mjcf_path: Path) -> dict[str, list[ET.Element]]:
    root = ET.parse(mjcf_path).getroot()
    parts: dict[str, list[ET.Element]] = {"default": [], "asset": [], "body": [], "actuator": []}
    for d in root.findall("default"):
        parts["default"].extend(list(d))
    for a in root.findall("asset"):
        parts["asset"].extend(list(a))
    parts["body"] = [copy.deepcopy(b) for b in root.find("worldbody").findall("body")]
    for a in root.findall("actuator"):
        parts["actuator"].extend(list(a))

    gripper = next(b for b in parts["body"][0].iter("body") if b.get("name") == "gripper")
    tcp = next(s for s in gripper.findall("site") if s.get("name") == "gripperframe")
    tcp_pos = np.array([float(v) for v in tcp.get("pos").split()])
    tcp_rot = _quat_to_mat([float(v) for v in tcp.get("quat").split()])
    grasp = tcp_pos + tcp_rot @ np.array(layout.GRASP_OFFSET_IN_TCP)
    ET.SubElement(
        gripper, "site", name="grasp_center", pos=_f(*grasp), size="0.004", rgba="0 1 1 0.6", group="4"
    )
    # Wrist camera on the fixed-jaw side (like the SO-101 wrist-cam mount), looking down the jaws.
    approach, up = tcp_rot[:, 0], tcp_rot[:, 2]
    cam_pos = tcp_pos - 0.085 * approach - 0.030 * up
    view = (tcp_pos + 0.03 * up) - cam_pos
    view /= np.linalg.norm(view)
    cam_z = -view
    cam_y = up - np.dot(up, cam_z) * cam_z
    cam_y /= np.linalg.norm(cam_y)
    cam_x = np.cross(cam_y, cam_z)
    ET.SubElement(gripper, "camera", name="wrist_cam", pos=_f(*cam_pos), xyaxes=_f(*cam_x, *cam_y), fovy="70")

    # The convex hull of the printed gripper part fills the space between the jaws, so the jaws
    # could never close around a parcel. Swap the two hull colliders for finger pads + a palm block
    # (boxes measured from the mesh vertices, expressed in the TCP frame).
    jaw = next(b for b in gripper.findall("body") if b.get("name") == "moving_jaw_so101_v1")
    for body, mesh in ((gripper, "wrist_roll_follower_so101_v1"), (jaw, "moving_jaw_so101_v1")):
        for geom in body.findall("geom"):
            if geom.get("class") == "collision" and geom.get("mesh") == mesh:
                geom.set("contype", "0")
                geom.set("conaffinity", "0")
    jaw_pos = np.array([float(v) for v in jaw.get("pos").split()])
    jaw_rot = _quat_to_mat([float(v) for v in jaw.get("quat").split()])
    pad = 'class="collision" type="box" friction="1.6 0.02 0.002" condim="4" rgba="0.2 0.9 1 0.5"'
    for name, center, half, in_jaw in _GRIPPER_PADS:
        c_body = tcp_pos + tcp_rot @ np.array(center)  # gripper-body frame
        rot = tcp_rot
        if in_jaw:  # jaw frame at gripper = 0
            c_body = jaw_rot.T @ (c_body - jaw_pos)
            rot = jaw_rot.T @ tcp_rot
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, rot.flatten())
        (jaw if in_jaw else gripper).append(
            ET.fromstring(
                f'<geom name="{name}" pos="{_f(*c_body)}" quat="{_f(*quat)}" size="{_f(*half)}" {pad}/>'
            )
        )
    # Servo tuning for a *twin*: the published gains (kp=998, kv=2.7, +-3.35 Nm) cap joint speed at
    # ~1 rad/s (force limit / damping), so the twin would trail a real STS3215 (~4.7 rad/s no-load).
    # kp=200 / kv=1.5 keeps sag < 0.2 deg, damping ratio ~0.4 and lets the arm reach ~2 rad/s.
    for d in parts["default"]:
        if d.get("class") == "sts3215":
            d.find("joint").set("damping", "0.3")
            d.find("position").set("kp", "200")
            d.find("position").set("kv", "1.5")
    jaw.find("joint").set("damping", "0.05")
    jaw.find("joint").set("armature", "0.004")
    for act in parts["actuator"]:
        if act.get("joint") == "gripper":
            # LeRobot runs the real gripper at 50 % torque (Max_Torque_Limit=500).
            act.set("forcerange", "-1.5 1.5")
            act.set("kv", "0.6")
    return parts


# (name, centre in TCP frame, half extents, attached to moving jaw)
_GRIPPER_PADS = (
    ("pad_fixed", (-0.025, 0.0, -0.0062), (0.030, 0.0085, 0.0057), False),
    ("pad_palm", (-0.082, 0.0, 0.004), (0.017, 0.022, 0.028), False),
    ("pad_moving", (-0.022, 0.0, 0.0226), (0.028, 0.0085, 0.0046), True),
)


# --------------------------------------------------------------------------------------------
# environment helpers
# --------------------------------------------------------------------------------------------
# Untextured materials can be folded into merged meshes by colour.
_PLAIN_MATERIALS = {"mat_table": (0.52, 0.42, 0.31, 1.0), "mat_belt": (0.08, 0.08, 0.09, 1.0)}

_GLOW_MATERIALS = {"mat_lamp", "mat_glow"}  # emissive but untextured: mergeable, keep the material

# Face corners of a unit box (+-1), one quad per face so every face gets its own flat normal.
_BOX_FACES = (
    ((1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1)),
    ((-1, 1, -1), (-1, -1, -1), (-1, -1, 1), (-1, 1, 1)),
    ((1, 1, -1), (-1, 1, -1), (-1, 1, 1), (1, 1, 1)),
    ((-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)),
    ((-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)),
    ((-1, 1, -1), (1, 1, -1), (1, -1, -1), (-1, -1, -1)),
)


@dataclass
class _Xml:
    """XML fragments + a batch of plain-coloured boxes that get merged into one mesh per colour.

    Every geom costs a draw call (and on WSLg's D3D12 GL path ~1 ms each), so the ~900 boxes of
    racks, cartons and markings are baked into a handful of static meshes. Colliders stay
    individual (invisible, group 3) so the game can still look them up by name.
    """

    assets: list[str] = field(default_factory=list)
    world: list[str] = field(default_factory=list)
    equality: list[str] = field(default_factory=list)
    merge: dict[tuple, list[tuple]] = field(default_factory=dict)

    def box(self, name, pos, half, rgba=None, material=None, deco=True, quat=None, extra=""):
        q = f' quat="{quat}"' if quat else ""
        nm = f' name="{name}"' if name else ""
        glow = material in _GLOW_MATERIALS
        plain = rgba if material is None or glow else _PLAIN_MATERIALS.get(material)
        if not deco:
            grp = "" if "group=" in extra else ' group="3"'
            col = f' rgba="{_f(*plain)}"' if plain else ""
            self.world.append(
                f'<geom{nm} class="env" type="box" pos="{_f(*pos)}"{q} size="{_f(*half)}"{col}{grp} {extra}/>'
            )
            if plain is not None and plain[3] > 0 and "group=" not in extra:
                self._queue(plain, pos, half, quat)
            elif plain is None:  # textured collider: also needs a visible copy
                self.world.append(
                    f'<geom class="deco" type="box" pos="{_f(*pos)}"{q} size="{_f(*half)}" material="{material}"/>'
                )
        elif name is None and plain is not None and not extra:
            self._queue(plain, pos, half, quat, material if glow else None)
        else:
            look = f'material="{material}"' if material else f'rgba="{_f(*rgba)}"'
            self.world.append(
                f'<geom{nm} class="deco" type="box" pos="{_f(*pos)}"{q} size="{_f(*half)}" {look} {extra}/>'
            )

    def _queue(self, rgba, pos, half, quat, material: str | None = None) -> None:
        key = (*(round(float(c), 3) for c in rgba), material or "")
        q = [float(v) for v in quat.split()] if quat else [1.0, 0.0, 0.0, 0.0]
        self.merge.setdefault(key, []).append(
            (np.asarray(pos, float), np.asarray(half, float), _quat_to_mat(q))
        )

    def flush_merged(self, frame: str = "world") -> None:
        for k, (key, boxes) in enumerate(sorted(self.merge.items())):
            rgba, material = key[:4], key[4]
            verts, faces = [], []
            for pos, half, rot in boxes:
                for quad in _BOX_FACES:
                    base = len(verts)
                    verts += [pos + rot @ (np.array(c) * half) for c in quad]
                    faces += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
            name = f"merged_{frame}_{k}"
            vtxt = " ".join(f"{v:.5g}" for p in verts for v in p)
            ftxt = " ".join(str(i) for f in faces for i in f)
            self.assets.append(f'<mesh name="{name}" vertex="{vtxt}" face="{ftxt}" inertia="shell"/>')
            mat = f' material="{material}"' if material else ""
            self.world.append(f'<geom class="deco" type="mesh" mesh="{name}" rgba="{_f(*rgba)}"{mat}/>')
        self.merge.clear()


def _floor_collision(x: _Xml) -> None:
    hx, hy = layout.HALL_SIZE[0] / 2, layout.HALL_SIZE[1] / 2
    x.box(
        "floor_collision",
        (0, 0, layout.FLOOR_Z - 0.1),
        (hx, hy, 0.1),
        rgba=(0, 0, 0, 0),
        deco=False,
        extra='group="5"',
    )


def _isaac_hall(x: _Xml, verbose: bool) -> None:
    """NVIDIA Isaac Sim's warehouse (converted by import_isaac_scene.py) instead of the built-in hall."""
    from . import isaac_scene

    man = isaac_scene.load_manifest()
    if man is None:
        raise SystemExit(
            "The Isaac warehouse has not been imported yet (or was imported for another placement).\n"
            "Run once:  pip install usd-core  &&  python import_isaac_scene.py\n"
            "or play with the built-in hall:  --hall builtin"
        )
    assets, world = isaac_scene.mjcf(man)
    x.assets += assets
    x.world += world
    _floor_collision(x)
    if verbose:
        tris = sum(g["tris"] for g in man["groups"])
        print(f"[twin] Isaac warehouse: {len(man['groups'])} meshes, {tris / 1e3:.0f}k triangles", flush=True)


def _hall(x: _Xml, rng: random.Random) -> None:
    fz = layout.FLOOR_Z
    hx, hy = layout.HALL_SIZE[0] / 2, layout.HALL_SIZE[1] / 2
    x.world.append(
        f'<geom name="floor_visual" class="deco" type="plane" pos="0 0 {fz:.4f}" size="{hx} {hy} 0.1" material="mat_floor"/>'
    )
    _floor_collision(x)
    # safety walkways around the cell
    for cx, cy, sx, sy in (
        (0.1, 2.25, 1.6, 0.03),
        (0.1, -2.05, 1.6, 0.03),
        (1.65, 0.1, 0.03, 2.15),
        (-1.45, 0.1, 0.03, 2.15),
    ):
        x.box(None, (cx, cy, fz + 0.002), (sx, sy, 0.001), rgba=(1.0, 0.78, 0.05, 1))
    for i in range(12):  # hatched no-go zone in front of the racks
        x.box(
            None,
            (-4.6 + 0.5 * i, 3.2, fz + 0.002),
            (0.05, 0.6, 0.001),
            rgba=(0.95, 0.75, 0.1, 1),
            quat=_yaw_quat(35),
        )
    # walls + banner
    h = layout.HALL_HEIGHT
    wall = (0.22, 0.24, 0.29, 1)
    x.box(None, (hx, 0, fz + h / 2), (0.1, hy, h / 2), rgba=wall)
    x.box(None, (-hx, 0, fz + h / 2), (0.1, hy, h / 2), rgba=wall)
    x.box(None, (0, hy, fz + h / 2), (hx, 0.1, h / 2), rgba=wall)
    x.box(None, (0, -hy, fz + h / 2), (hx, 0.1, h / 2), rgba=wall)
    x.box("banner", (hx - 0.13, 0.0, fz + 3.6), (0.02, 3.0, 0.56), material="mat_banner")
    for k in range(5):  # windows with a cold night glow
        x.box(
            None,
            (-hx + 0.12, -6 + 3 * k, fz + 4.2),
            (0.02, 1.0, 1.2),
            rgba=(0.12, 0.2, 0.38, 1),
            material="mat_glow",
        )
    # ceiling light panels
    for i in range(4):
        for j in range(3):
            x.box(
                None,
                (-7.5 + 5 * i, -5 + 5 * j, fz + h - 0.4),
                (0.9, 0.12, 0.03),
                rgba=(0.95, 0.95, 0.9, 1),
                material="mat_lamp",
            )
    # pallet racks with random cartons
    carton = ((0.62, 0.45, 0.28, 1), (0.70, 0.52, 0.33, 1), (0.55, 0.40, 0.25, 1), (0.25, 0.42, 0.70, 1))
    for ry in (4.6, 7.2, -5.2, -7.6):
        for bay in range(6):
            bx = -7.5 + bay * 2.8
            for side in (-1.3, 1.3):
                for dy in (-0.5, 0.5):
                    x.box(None, (bx + side, ry + dy, fz + 2.4), (0.05, 0.05, 2.4), rgba=(0.95, 0.42, 0.08, 1))
            for level in range(4):
                z = fz + 0.15 + level * 1.4
                for dy in (-0.5, 0.5):
                    x.box(None, (bx, ry + dy, z), (1.3, 0.04, 0.06), rgba=(0.10, 0.30, 0.65, 1))
                for slot in (-0.7, 0.0, 0.7):
                    if rng.random() < 0.18:
                        continue
                    x.box(
                        None, (bx + slot, ry, z + 0.1), (0.38, 0.45, 0.04), rgba=(0.55, 0.42, 0.25, 1)
                    )  # pallet
                    for _ in range(rng.randint(1, 3)):
                        sx, sy, sz = rng.uniform(0.12, 0.3), rng.uniform(0.15, 0.35), rng.uniform(0.1, 0.25)
                        x.box(
                            None,
                            (
                                bx + slot + rng.uniform(-0.15, 0.15),
                                ry + rng.uniform(-0.15, 0.15),
                                z + 0.14 + sz,
                            ),
                            (sx, sy, sz),
                            rgba=rng.choice(carton),
                            quat=_yaw_quat(rng.uniform(-8, 8)),
                        )
    # a parked forklift-ish silhouette and a few floor pallets for depth
    for px, py in ((4.2, -1.3), (5.0, -1.3), (-4.6, 1.0)):
        x.box(None, (px, py, fz + 0.07), (0.6, 0.5, 0.07), rgba=(0.55, 0.42, 0.25, 1))
        x.box(None, (px, py, fz + 0.45), (0.5, 0.42, 0.3), rgba=rng.choice(carton))


def _workbench(x: _Xml) -> None:
    (cx, cy), (sx, sy) = layout.TABLE_CENTER, layout.TABLE_SIZE
    x.box("table_top", (cx, cy, -0.02), (sx / 2, sy / 2, 0.02), material="mat_table", deco=False)
    x.box(None, (cx, cy, -0.0395), (sx / 2 + 0.01, sy / 2 + 0.01, 0.0005), rgba=(0.15, 0.15, 0.16, 1))
    for lx in (-1, 1):
        for ly in (-1, 1):
            x.box(
                None,
                (cx + lx * (sx / 2 - 0.05), cy + ly * (sy / 2 - 0.05), layout.FLOOR_Z / 2 - 0.02),
                (0.03, 0.03, -layout.FLOOR_Z / 2 - 0.02),
                rgba=(0.3, 0.32, 0.36, 1),
            )
    # robot mounting plate (where the real follower is clamped)
    x.box(None, (0.01, 0.0, 0.0006), (0.075, 0.065, 0.0006), rgba=(0.12, 0.12, 0.13, 1))
    # beacon pole + lamp (blinks on jams / VIP alerts)
    x.world.append(
        '<geom class="deco" type="cylinder" pos="-0.30 0.55 0.15" size="0.008 0.15" rgba="0.3 0.3 0.32 1"/>'
    )
    x.world.append(
        '<geom name="beacon" class="deco" type="cylinder" pos="-0.30 0.55 0.32" size="0.025 0.025" rgba="0.35 0.25 0.05 1"/>'
    )


def _conveyor(x: _Xml) -> None:
    """One long belt past the robot: infeed rollers -> metering tunnel -> slat pick section -> outfeed
    rollers -> drop into the recycling cage. One slippery collider; the game drives the parcels."""
    c = layout.CONVEYOR
    tz = c.top_z
    length = c.y_in - c.y_out
    yc = (c.y_in + c.y_out) / 2
    x.box(
        "belt",
        (c.x, yc, tz / 2),
        (c.width / 2 + 0.006, length / 2, tz / 2),
        material="mat_belt",
        deco=False,
        extra='friction="0.05 0.005 0.0001" priority="2"',
    )
    frame = (0.55, 0.57, 0.6, 1)
    t_lo, t_hi = (
        layout.TABLE_CENTER[1] - layout.TABLE_SIZE[1] / 2,
        layout.TABLE_CENTER[1] + layout.TABLE_SIZE[1] / 2,
    )
    # under-frame + stands where the belt leaves the table
    for y0, y1 in ((t_hi, c.y_in), (c.y_out, t_lo)):
        x.box(None, (c.x, (y0 + y1) / 2, -0.025), (c.width / 2 + 0.016, abs(y1 - y0) / 2, 0.025), rgba=frame)
        for y in np.linspace(y0, y1, max(2, int(abs(y1 - y0) / 0.6) + 1))[1:]:
            for s in (-1, 1):
                x.box(
                    None,
                    (
                        c.x + s * (c.width / 2 + 0.01),
                        y - 0.04 * np.sign(y1 - y0),
                        (layout.FLOOR_Z - 0.05) / 2,
                    ),
                    (0.012, 0.012, (-layout.FLOOR_Z - 0.05) / 2),
                    rgba=(0.3, 0.32, 0.36, 1),
                )
    # side frames along infeed and outfeed (the open pick section in between)
    for y0, y1 in ((c.gate_y, c.y_in), (c.y_out, c.pick_lo)):
        for s in (-1, 1):
            x.box(
                None,
                (c.x + s * (c.width / 2 + 0.010), (y0 + y1) / 2, (tz + c.rail_height) / 2),
                (0.004, (y1 - y0) / 2, (tz + c.rail_height) / 2),
                rgba=(0.75, 0.77, 0.8, 1),
            )
        # rollers (static: on the fast sections the parcels glide over powered rollers)
        for y in np.arange(y0 + 0.02, y1, 0.045):
            x.box(
                None, (c.x, y, tz + 0.0006), (c.width / 2 + 0.004, 0.012, 0.0006), rgba=(0.62, 0.64, 0.68, 1)
            )
    for y in (c.y_out - 0.01, c.y_in + 0.01):  # end drums
        x.world.append(
            f'<geom class="deco" type="cylinder" pos="{_f(c.x, y, tz / 2)}" quat="0.7071 0 0.7071 0" '
            f'size="{_f(tz / 2 + 0.002, c.width / 2 + 0.008)}" rgba="0.4 0.4 0.42 1"/>'
        )
    # metering tunnel over the infeed (its light shows the gate state)
    t0, t1 = c.tunnel_y
    tl, ty = (t1 - t0) / 2, (t0 + t1) / 2
    for s in (-1, 1):
        x.box(
            None,
            (c.x + s * (c.width / 2 + 0.03), ty, 0.06),
            (0.012, tl, 0.06),
            rgba=(0.95, 0.75, 0.1, 1),
            deco=False,
        )
    x.box(None, (c.x, ty, 0.125), (c.width / 2 + 0.045, tl, 0.008), rgba=(0.15, 0.15, 0.17, 1))
    for k in range(6):
        x.box(
            None,
            (c.x, t0 + 0.002, 0.03 + 0.016 * k),
            (c.width / 2 + 0.042, 0.002, 0.004),
            rgba=(0.1, 0.1, 0.1, 1) if k % 2 else (0.95, 0.75, 0.1, 1),
        )
    x.world.append(
        f'<geom name="tunnel_light" class="deco" type="box" pos="{_f(c.x, t0 - 0.002, 0.118)}" '
        f'size="{_f(c.width / 2, 0.003, 0.004)}" rgba="0.2 1 0.4 1" material="mat_glow"/>'
    )
    # table markings beside the reach window: green = pick here, red = last chance
    zx = c.width / 2 + 0.024
    y_last = c.reach_lo + c.last_chance
    for name, y0, y1, rgba in (
        ("pick_zone", y_last, c.reach_hi, "0.1 0.9 0.35 1"),
        ("last_zone", c.reach_lo, y_last, "1 0.15 0.1 1"),
    ):
        for side, s in (("l", 1), ("r", -1)):
            x.world.append(
                f'<geom name="{name}_{side}" class="deco" type="box" pos="{_f(c.x + s * zx, (y0 + y1) / 2, 0.0009)}" '
                f'size="{_f(0.006, (y1 - y0) / 2, 0.0009)}" rgba="{rgba}" material="mat_glow"/>'
            )


def _bins(x: _Xml) -> None:
    for b in layout.BINS:
        cx, cy = b.center
        inner, w, h = b.inner / 2, b.wall, b.height
        rgba = b.rgba
        dark = tuple(v * 0.55 for v in rgba[:3]) + (1.0,)
        parts = [
            (f"bin_{b.key}_floor", (0, 0, 0.002), (inner + w, inner + w, 0.002), dark),
            (f"bin_{b.key}_wall_far", (inner + w / 2, 0, h / 2), (w / 2, inner + w, h / 2), rgba),
            (f"bin_{b.key}_wall_near", (-inner - w / 2, 0, h / 2), (w / 2, inner + w, h / 2), rgba),
            (f"bin_{b.key}_wall_l", (0, inner + w / 2, h / 2), (inner, w / 2, h / 2), rgba),
            (f"bin_{b.key}_wall_r", (0, -inner - w / 2, h / 2), (inner, w / 2, h / 2), rgba),
        ]
        for name, (lx, ly, lz), half, col in parts:
            x.box(name, (cx + lx, cy + ly, lz), half, rgba=col, deco=False)
        # sign on the far side, facing the robot, on a little post
        x.box(
            f"sign_{b.key}",
            (cx + inner + w + 0.006, cy, h + 0.026),
            (0.002, 0.045, 0.0225),
            material=f"mat_sign_{b.key}",
        )
        x.box(
            None, (cx + inner + w + 0.006, cy, h / 2 + 0.002), (0.0015, 0.004, h / 2), rgba=(0.3, 0.3, 0.3, 1)
        )


def _buttons(x: _Xml) -> None:
    """Two mushroom buttons the arm can push: a sprung cap on a slide joint over a visual housing."""
    for b in layout.BUTTONS:
        cx, cy = b.center
        dark = _f(*(v * 0.35 for v in b.rgba[:3]), 1)
        x.world += [
            f'<geom class="deco" type="cylinder" pos="{_f(cx, cy, b.base_h / 2)}" size="{_f(b.radius + 0.008, b.base_h / 2)}" '
            'rgba="0.16 0.16 0.18 1"/>',
            f'<geom name="button_{b.key}_light" class="deco" type="cylinder" pos="{_f(cx, cy, b.base_h + 0.0005)}" '
            f'size="{_f(b.radius + 0.007, 0.0012)}" rgba="{_f(*b.rgba)}" material="mat_glow"/>',
            f'<geom class="deco" type="box" pos="{_f(cx, cy - b.radius - 0.016, 0.0008)}" size="{_f(0.02, 0.006, 0.0008)}" '
            f'rgba="{_f(*b.rgba)}"/>',
            f'<body name="button_{b.key}" pos="{_f(cx, cy, b.base_h)}">'
            f'<joint name="button_{b.key}" type="slide" axis="0 0 1" range="{_f(-b.travel, 0.0)}" stiffness="150" '
            'damping="1.5" springref="0.0007" armature="0.002"/>'
            f'<geom name="button_{b.key}_cap" class="env" type="cylinder" pos="0 0 0.006" size="{_f(b.radius, 0.006)}" '
            f'mass="0.01" rgba="{dark}"/>'
            f'<geom class="deco" type="cylinder" pos="0 0 0.0125" size="{_f(b.radius * 0.8, 0.0006)}" rgba="{_f(*b.rgba)}" '
            'material="mat_glow"/></body>',
        ]


def _scanner(x: _Xml) -> None:
    s = layout.SCANNER
    (cx, cy), hf = s.center, s.half
    z1 = s.z_range[1]
    x.box("scan_pad", (cx, cy, 0.0008), (hf, hf, 0.0008), material="mat_scanner")
    for dx in (-1, 1):
        for dy in (-1, 1):
            x.box(
                None, (cx + dx * hf, cy + dy * hf, z1 / 2), (0.004, 0.004, z1 / 2), rgba=(0.2, 0.2, 0.22, 1)
            )
    for dz in (0.035, 0.07, 0.105):  # laser curtain
        x.box(None, (cx, cy + hf, dz), (hf, 0.0008, 0.0008), rgba=(1, 0.1, 0.1, 0.9), material="mat_glow")
    for dx, dy, sx, sy in (
        (0, hf, hf + 0.004, 0.004),
        (0, -hf, hf + 0.004, 0.004),
        (hf, 0, 0.004, hf),
        (-hf, 0, 0.004, hf),
    ):
        x.box(
            None,
            (cx + dx, cy + dy, z1),
            (sx, sy, 0.004),
            rgba=(0.95, 0.2, 0.2, 1) if dy else (0.2, 0.2, 0.22, 1),
        )


def _lights(x: _Xml, shadows: bool, hall: str = "builtin") -> None:
    fz, h = layout.FLOOR_Z, layout.HALL_HEIGHT
    if hall == "isaac":  # the Isaac warehouse's own area lights don't exist in MuJoCo: daylight shift
        x.world += [
            '<light name="isaac_sun" directional="true" pos="0 0 8" dir="-0.35 0.25 -1" diffuse="0.42 0.42 0.40" '
            'specular="0.1 0.1 0.1" castshadow="false"/>',
            f'<light name="isaac_aisles" pos="6 0 {fz + 7.5}" dir="0.3 0 -1" cutoff="70" exponent="1" '
            'diffuse="0.6 0.6 0.58" castshadow="false"/>',
        ]
    x.world += [
        f'<light name="moon" directional="true" pos="0 0 {fz + h}" dir="0.3 0.2 -1" diffuse="0.18 0.2 0.28" '
        'specular="0 0 0" castshadow="false"/>',
        '<light name="cell_spot" pos="0.05 -0.35 1.35" dir="0.05 0.25 -1" cutoff="40" exponent="8" '
        f'diffuse="{"0.6 0.59 0.55" if hall == "isaac" else "0.95 0.93 0.86"}" specular="0.3 0.3 0.3" '
        f'castshadow="{"true" if shadows else "false"}"/>',
        f'<light name="hall_a" pos="-6 4 {fz + h - 0.5}" dir="0 0 -1" cutoff="60" exponent="2" diffuse="0.45 0.45 0.5" castshadow="false"/>',
        f'<light name="hall_b" pos="6 -4 {fz + h - 0.5}" dir="0 0 -1" cutoff="60" exponent="2" diffuse="0.45 0.45 0.5" castshadow="false"/>',
        '<light name="beacon_light" pos="-0.30 0.55 0.40" dir="0.4 -0.4 -1" cutoff="70" exponent="1" '
        'diffuse="0 0 0" specular="0 0 0" castshadow="false"/>',
    ]


def _cameras(x: _Xml) -> None:
    x.world += [
        '<camera name="operator" pos="-0.42 0.02 0.46" xyaxes="0 -1 0 0.72 0 0.7" fovy="55"/>',
        '<camera name="top" pos="0.13 0.1 1.0" xyaxes="0 -1 0 1 0 0" fovy="55"/>',
        '<camera name="hall" pos="-2.8 -2.2 1.2" xyaxes="0.62 -0.78 0 0.25 0.2 0.95" fovy="60"/>',
        '<camera name="line" mode="targetbody" target="world" pos="-1.4 3.4 1.6" fovy="55"/>',
    ]


def _cargo_slots(prefix: str, origin, nx: int, ny: int, pitch: float, size: float = 0.015) -> list[str]:
    """Hidden parcel look-alikes (geom group 5) that a vehicle or the cage shows as its load."""
    out = []
    k = 0
    for j in range(ny):
        for i in range(nx):
            px = origin[0] + (i - (nx - 1) / 2) * pitch
            py = origin[1] + (j - (ny - 1) / 2) * pitch
            out.append(
                f'<geom name="{prefix}_cargo_{k}" class="deco" type="box" pos="{_f(px, py, origin[2] + size)}" '
                f'size="{_f(size, size, size)}" group="5" material="mat_parcel_standard"/>'
            )
            k += 1
    return out


def _forklift_builtin() -> list[str]:
    def b(pos, half, rgba):
        return f'<geom class="deco" type="box" pos="{_f(*pos)}" size="{_f(*half)}" rgba="{_f(*rgba)}"/>'

    y, dk = (1.0, 0.72, 0.08, 1), (0.12, 0.12, 0.13, 1)
    g = [
        b((-0.25, 0, 0.42), (0.6, 0.45, 0.32), y),
        b((-0.75, 0, 0.55), (0.12, 0.42, 0.35), dk),  # counterweight
        b((-0.15, 0, 1.25), (0.45, 0.42, 0.025), dk),  # overhead guard
        b((0.55, 0, 1.1), (0.05, 0.38, 1.1), dk),  # mast
        b((0.95, 0.2, 0.06), (0.38, 0.05, 0.02), dk),
        b((0.95, -0.2, 0.06), (0.38, 0.05, 0.02), dk),
        b((0.95, 0, 0.13), (0.4, 0.45, 0.05), (0.55, 0.42, 0.25, 1)),  # pallet
        b((0.95, 0, 0.38), (0.32, 0.38, 0.2), (0.62, 0.45, 0.28, 1)),
    ]
    for wx in (-0.55, 0.3):
        for wy in (-0.42, 0.42):
            g.append(
                f'<geom class="deco" type="cylinder" pos="{_f(wx, wy, 0.18)}" quat="0.7071 0.7071 0 0" '
                f'size="0.18 0.06" rgba="0.08 0.08 0.08 1"/>'
            )
    g.append(
        '<geom name="forklift_lamp" class="deco" type="sphere" pos="-0.3 0 1.32" size="0.05" rgba="1 0.55 0.05 1" material="mat_glow"/>'
    )
    return g


def _mocap_props(x: _Xml, hall: str) -> None:
    c, fz = layout.CONVEYOR, layout.FLOOR_Z
    # pick-section slats (the visible moving belt)
    for i in range(N_SLATS):
        x.world.append(
            f'<body name="slat_{i}" mocap="true" pos="{_f(c.x, c.pick_lo + SLAT_PITCH * i, c.top_z + 0.0004)}">'
            f'<geom class="deco" type="box" size="{_f(c.width / 2 + 0.004, 0.0035, 0.0004)}" rgba="0.22 0.22 0.24 1"/></body>'
        )
    # metering gate: a little yellow/black arm that drops when a parcel is released
    x.world.append(
        f'<body name="gate" mocap="true" pos="{_f(c.x, c.gate_y - 0.022, c.top_z + 0.012)}">'
        f'<geom class="deco" type="box" size="{_f(c.width / 2 + 0.012, 0.003, 0.006)}" rgba="0.95 0.75 0.1 1"/>'
        f'<geom class="deco" type="box" pos="0 0 0.0065" size="{_f(c.width / 2 + 0.012, 0.0031, 0.0015)}" rgba="0.1 0.1 0.1 1"/></body>'
    )
    # inbound AGV with a lift deck at belt height (body frame: +x forward, z = 0 on the floor)
    deck_z = -fz + c.top_z
    agv = [
        '<geom class="deco" type="box" pos="0 0 0.14" size="0.55 0.38 0.11" rgba="0.95 0.55 0.08 1"/>',
        '<geom class="deco" type="box" pos="0 0 0.255" size="0.5 0.34 0.006" rgba="0.15 0.15 0.15 1"/>',
        f'<geom class="deco" type="box" pos="0.3 0 {_f(deck_z / 2 + 0.12)}" size="{_f(0.05, 0.12, deck_z / 2 - 0.12)}" rgba="0.2 0.2 0.22 1"/>',
        f'<geom class="deco" type="box" pos="0.4 0 {_f(deck_z - 0.006)}" size="0.15 0.16 0.006" rgba="0.6 0.62 0.66 1"/>',
        f'<geom class="deco" type="box" pos="0.4 0.155 {_f(deck_z + 0.012)}" size="0.15 0.004 0.012" rgba="0.95 0.55 0.08 1"/>',
        f'<geom class="deco" type="box" pos="0.4 -0.155 {_f(deck_z + 0.012)}" size="0.15 0.004 0.012" rgba="0.95 0.55 0.08 1"/>',
        '<geom class="deco" type="box" pos="0.555 0 0.14" size="0.004 0.3 0.015" rgba="0.2 0.8 1 1" material="mat_glow"/>',
        '<geom name="agv_lamp" class="deco" type="sphere" pos="-0.45 0.3 0.3" size="0.03" rgba="1 0.7 0.1 1" material="mat_glow"/>',
    ] + _cargo_slots("agv", (0.4, 0.0, deck_z), 3, 2, 0.06)
    tug = [
        '<geom class="deco" type="box" pos="0 0 0.13" size="0.45 0.3 0.1" rgba="0.18 0.45 0.85 1"/>',
        '<geom class="deco" type="box" pos="0.1 0 0.36" size="0.18 0.26 0.13" rgba="0.12 0.12 0.14 1"/>',
        '<geom class="deco" type="box" pos="-0.52 0 0.12" size="0.08 0.03 0.03" rgba="0.3 0.3 0.3 1"/>',
        '<geom class="deco" type="box" pos="0.455 0 0.13" size="0.004 0.25 0.015" rgba="0.2 0.8 1 1" material="mat_glow"/>',
        '<geom name="tug_lamp" class="deco" type="sphere" pos="0.1 0 0.52" size="0.035" rgba="0.2 0.6 1 1" material="mat_glow"/>',
    ]
    if hall == "isaac" and _isaac_actor("forklift") is not None:
        man, actor = _isaac_actor("forklift")
        from . import isaac_scene

        assets, geoms = isaac_scene.mjcf(man, actor["groups"], prefix="forklift")
        x.assets += assets
        forklift = geoms
    else:
        forklift = _forklift_builtin()
    for name, geoms in (("agv", agv), ("tug", tug), ("forklift", forklift)):
        x.world.append(f'<body name="{name}" mocap="true" pos="0 0 {_f(fz)}">')
        x.world += geoms
        x.world.append("</body>")
    # recycling cage: real colliders (parcels land in it), wire-mesh look, load shown when towed
    cg = layout.CAGE
    hx, hy, h = cg.half[0], cg.half[1], cg.height
    cage = [
        f'<geom name="cage_floor" class="env" type="box" pos="0 0 0.03" size="{_f(hx, hy, 0.02)}" rgba="0.35 0.37 0.4 1"/>',
        f'<geom name="cage_wall_n" class="env" type="box" pos="0 {_f(hy)} {_f(h / 2)}" size="{_f(hx, 0.006, h / 2)}" rgba="0.7 0.74 0.8 0.35"/>',
        f'<geom name="cage_wall_s" class="env" type="box" pos="0 {_f(-hy)} {_f(h / 2)}" size="{_f(hx, 0.006, h / 2)}" rgba="0.7 0.74 0.8 0.35"/>',
        f'<geom name="cage_wall_e" class="env" type="box" pos="{_f(hx)} 0 {_f(h / 2)}" size="{_f(0.006, hy, h / 2)}" rgba="0.7 0.74 0.8 0.35"/>',
        f'<geom name="cage_wall_w" class="env" type="box" pos="{_f(-hx)} 0 {_f(h / 2)}" size="{_f(0.006, hy, h / 2)}" rgba="0.7 0.74 0.8 0.35"/>',
    ]
    for sx in (-1, 1):
        for sy in (-1, 1):
            cage.append(
                f'<geom class="deco" type="box" pos="{_f(sx * hx, sy * hy, h / 2)}" size="{_f(0.012, 0.012, h / 2)}" rgba="0.2 0.5 0.25 1"/>'
            )
    cage.append(
        f'<geom class="deco" type="box" pos="0 0 {_f(h)}" size="{_f(hx, hy, 0.004)}" rgba="0.2 0.5 0.25 0.0" group="5"/>'
    )
    for wx in (-1, 1):
        for wy in (-1, 1):
            cage.append(
                f'<geom class="deco" type="sphere" pos="{_f(wx * (hx - 0.05), wy * (hy - 0.05), 0.0)}" size="0.03" rgba="0.1 0.1 0.1 1"/>'
            )
    cage += _cargo_slots("cage", (0.0, 0.0, 0.05), 3, 2, 0.12, size=0.03)
    x.world.append(f'<body name="cage" mocap="true" pos="{_f(cg.center[0], cg.center[1], fz)}">')
    x.world += cage
    x.world.append("</body>")


def _isaac_actor(name: str):
    from . import isaac_scene

    man = isaac_scene.load_manifest()
    if man is None or name not in man.get("actors", {}):
        return None
    return man, man["actors"][name]


def _parcels(x: _Xml) -> list[tuple[str, str]]:
    """Pool of parcels parked in the hidden basement. Returns [(body_name, type_key)]."""
    pool: list[tuple[str, str]] = []
    px, py, pz = layout.PARK_ORIGIN
    x.box("park_shelf", (px, py, pz - 0.01), (1.0, 1.0, 0.01), rgba=(0.1, 0.1, 0.1, 1), deco=False)
    idx = 0
    for t in layout.PARCEL_TYPES:
        for _ in range(t.pool):
            name = f"p{idx:02d}_{t.key}"
            hx, hy, hz = (v / 2 for v in t.size)
            ix, iy = idx % 5, idx // 5
            x.world.append(
                f'<body name="{name}" pos="{_f(px - 0.4 + 0.2 * ix, py - 0.4 + 0.2 * iy, pz + hz + 0.001)}">'
                f'<freejoint name="{name}_free"/>'
                f'<geom name="{name}" class="parcel" type="box" size="{_f(hx, hy, hz)}" mass="{t.mass}" '
                f'material="mat_parcel_{t.key}"/></body>'
            )
            x.equality.append(
                f'<weld name="assist_{name}" body1="gripper" body2="{name}" active="false" solref="0.004 1"/>'
            )
            pool.append((name, t.key))
            idx += 1
    return pool


def _materials(x: _Xml, tex: dict[str, Path]) -> None:
    a = x.assets
    a.append(f'<texture name="tex_floor" type="2d" file="{tex["floor"].as_posix()}"/>')
    a.append('<material name="mat_floor" texture="tex_floor" texrepeat="1 1" texuniform="true"/>')
    a.append(
        '<texture name="tex_wood" type="2d" builtin="flat" rgb1="0.55 0.43 0.3" width="64" height="64"/>'
    )
    a.append('<material name="mat_table" rgba="0.52 0.42 0.31 1" specular="0.1"/>')
    a.append('<material name="mat_belt" rgba="0.08 0.08 0.09 1" specular="0.05"/>')
    a.append('<material name="mat_glow" emission="1" specular="0"/>')
    a.append('<material name="mat_lamp" rgba="0.95 0.95 0.9 1" emission="1"/>')
    for key in ("banner", "scanner") + tuple(f"sign_{b.key}" for b in layout.BINS):
        a.append(
            f'<texture name="tex_{key}" type="cube" file="{tex[key].as_posix()}" gridsize="3 4" gridlayout=".U..LFRB.D.."/>'
        )
        a.append(f'<material name="mat_{key}" texture="tex_{key}" emission="0.35" specular="0.1"/>')
    for t in layout.PARCEL_TYPES:
        a.append(
            f'<texture name="tex_parcel_{t.key}" type="cube" file="{tex[f"parcel_{t.key}"].as_posix()}" gridsize="3 4" '
            'gridlayout=".U..LFRB.D.."/>'
        )
        # no reflectance: every reflective geom costs a full extra render pass
        shine = 'specular="1" shininess="0.95" emission="0.15"' if t.key == "vip" else 'specular="0.2"'
        a.append(f'<material name="mat_parcel_{t.key}" texture="tex_parcel_{t.key}" {shine}/>')


# --------------------------------------------------------------------------------------------
@dataclass
class SceneIndex:
    """Resolved ids for everything the game touches each frame."""

    model: mujoco.MjModel
    arm_qpos: np.ndarray
    arm_dof: np.ndarray
    arm_act: np.ndarray
    joint_ranges: np.ndarray
    grasp_site: int
    tcp_site: int
    gripper_body: int
    jaw_body: int
    robot_geoms: set[int]
    jaw_geoms: set[int]
    parcels: list[str]
    parcel_types: list[str]
    parcel_body: np.ndarray
    parcel_geom: np.ndarray
    parcel_qpos: np.ndarray
    parcel_dof: np.ndarray
    parcel_weld: np.ndarray
    bin_geoms: dict[str, set[int]]
    bin_floor_geom: dict[str, int]
    belt_geom: int
    table_geom: int
    slat_mocap: np.ndarray
    vehicle_mocap: dict[str, int]
    mobile_bodies: list[int]  # ignored when baking the vehicles' occupancy grid
    agv_cargo: list[int]
    cage_mocap: int
    cage_cargo: list[int]
    gate_mocap: int
    button_qpos: dict[str, int]
    button_light: dict[str, int]
    parcel_material: dict[str, int]
    beacon_geom: int
    beacon_light: int
    tunnel_light_geom: int
    zone_geoms: dict[str, list[int]]

    @classmethod
    def build(cls, m: mujoco.MjModel, pool: list[tuple[str, str]]) -> SceneIndex:
        def jid(n):
            return m.joint(n).id

        joints = [jid(n) for n in layout.ALL_JOINTS]
        gripper = m.body("gripper").id
        robot_bodies = {
            m.body(n).id
            for n in ("shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "moving_jaw_so101_v1")
        }
        robot_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] in robot_bodies and m.geom_contype[g]}
        jaw_bodies = {gripper, m.body("moving_jaw_so101_v1").id}
        names = [p for p, _ in pool]
        bin_geoms = {
            b.key: {g for g in range(m.ngeom) if m.geom(g).name.startswith(f"bin_{b.key}_")}
            for b in layout.BINS
        }
        return cls(
            model=m,
            arm_qpos=np.array([m.jnt_qposadr[j] for j in joints]),
            arm_dof=np.array([m.jnt_dofadr[j] for j in joints]),
            arm_act=np.array([m.actuator(n).id for n in layout.ALL_JOINTS]),
            joint_ranges=np.array([m.jnt_range[j] for j in joints]),
            grasp_site=m.site("grasp_center").id,
            tcp_site=m.site("gripperframe").id,
            gripper_body=gripper,
            jaw_body=m.body("moving_jaw_so101_v1").id,
            robot_geoms=robot_geoms,
            jaw_geoms={g for g in robot_geoms if m.geom_bodyid[g] in jaw_bodies},
            parcels=names,
            parcel_types=[t for _, t in pool],
            parcel_body=np.array([m.body(n).id for n in names]),
            parcel_geom=np.array([m.geom(n).id for n in names]),
            parcel_qpos=np.array([m.jnt_qposadr[m.joint(f"{n}_free").id] for n in names]),
            parcel_dof=np.array([m.jnt_dofadr[m.joint(f"{n}_free").id] for n in names]),
            parcel_weld=np.array([m.equality(f"assist_{n}").id for n in names]),
            bin_geoms=bin_geoms,
            bin_floor_geom={b.key: m.geom(f"bin_{b.key}_floor").id for b in layout.BINS},
            belt_geom=m.geom("belt").id,
            table_geom=m.geom("table_top").id,
            slat_mocap=np.array([m.body_mocapid[m.body(f"slat_{i}").id] for i in range(N_SLATS)]),
            vehicle_mocap={v: int(m.body_mocapid[m.body(v).id]) for v in VEHICLES},
            mobile_bodies=list(range(1, m.nbody)),  # everything static lives in the world body
            agv_cargo=[m.geom(f"agv_cargo_{k}").id for k in range(6)],
            cage_mocap=int(m.body_mocapid[m.body("cage").id]),
            cage_cargo=[m.geom(f"cage_cargo_{k}").id for k in range(6)],
            gate_mocap=int(m.body_mocapid[m.body("gate").id]),
            button_qpos={b.key: int(m.jnt_qposadr[m.joint(f"button_{b.key}").id]) for b in layout.BUTTONS},
            button_light={b.key: m.geom(f"button_{b.key}_light").id for b in layout.BUTTONS},
            parcel_material={t.key: m.material(f"mat_parcel_{t.key}").id for t in layout.PARCEL_TYPES},
            beacon_geom=m.geom("beacon").id,
            beacon_light=m.light("beacon_light").id,
            tunnel_light_geom=m.geom("tunnel_light").id,
            zone_geoms={z: [m.geom(f"{z}_{s}").id for s in "lr"] for z in ("pick_zone", "last_zone")},
        )


HALLS = ("builtin", "isaac")


def build_scene(
    seed: int = 3, verbose: bool = True, shadows: bool = True, hall: str = "builtin"
) -> tuple[mujoco.MjModel, SceneIndex, Path]:
    mjcf = ensure_so101_assets(verbose=verbose)
    robot = _robot_parts(mjcf)
    tex = build_textures(BUILD_DIR / "textures")
    x = _Xml()
    rng = random.Random(seed)
    _materials(x, tex)
    if hall == "isaac":
        _isaac_hall(x, verbose)
    else:
        _hall(x, rng)
    _workbench(x)
    _conveyor(x)
    _bins(x)
    _scanner(x)
    _buttons(x)
    _lights(x, shadows, hall)
    _cameras(x)
    pool = _parcels(x)
    x.flush_merged()
    _mocap_props(x, hall)

    def dump(elems):
        return "\n".join(ET.tostring(e, encoding="unicode") for e in elems)

    xml = f"""<mujoco model="twin_depot">
  <compiler angle="radian" meshdir="{(mjcf.parent / "assets").as_posix()}" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="10"/>
  <statistic center="0.12 0 0.05" extent="0.9"/>
  <visual>
    <global offwidth="1600" offheight="900" azimuth="180" elevation="-32"/>
    <headlight ambient="{"0.28 0.28 0.29" if hall == "isaac" else "0.22 0.22 0.25"}" diffuse="0.25 0.25 0.27" specular="0.05 0.05 0.05"/>
    <quality shadowsize="2048"/>
    <map znear="0.004" zfar="{120 if hall == "isaac" else 60}" haze="{0.15 if hall == "isaac" else 0.4}"/>
    <rgba haze="0.05 0.07 0.12 1"/>
  </visual>
  <default>
{dump(robot["default"])}
    <default class="deco"><geom contype="0" conaffinity="0" group="1"/></default>
    <default class="env"><geom contype="1" conaffinity="1" group="0" friction="0.9 0.01 0.001"/></default>
    <default class="parcel"><geom contype="1" conaffinity="1" group="0" condim="4" friction="1.3 0.02 0.002"
      solref="0.004 1" priority="1"/></default>
  </default>
  <asset>
    <texture name="skybox" type="skybox" builtin="gradient" rgb1="0.05 0.07 0.12" rgb2="0 0 0" width="512" height="3072"/>
{dump(robot["asset"])}
{chr(10).join(x.assets)}
  </asset>
  <worldbody>
{chr(10).join(x.world)}
{dump(robot["body"])}
  </worldbody>
  <equality>
{chr(10).join(x.equality)}
  </equality>
  <actuator>
{dump(robot["actuator"])}
  </actuator>
</mujoco>
"""
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    out = BUILD_DIR / "twin_depot.xml"
    out.write_text(xml)
    model = mujoco.MjModel.from_xml_string(xml)
    return model, SceneIndex.build(model, pool), out
