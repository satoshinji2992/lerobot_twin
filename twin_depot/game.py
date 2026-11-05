"""Twin Depot game logic: conveyor, parcel lifecycle, vehicles, scoring, phases and digital-twin telemetry.

Phases (live operator)::

    lobby --signal--> sync --pose matched--> ready --double pinch / ENTER--> countdown --> shift --> results
      ^ (attract demo)                         ^------------------------------------------------------'
    shift --signal lost--> paused --signal back--> shift

In ``demo`` mode the autopilot plays shifts back to back.

The line never stops: the inbound AGV unloads batches at the head of a long belt, the metering
gate lets them onto the slow pick section in front of the robot one by one, and whatever is not
picked rides on and drops into the recycling cage - which the tug hauls away when it is full
(parcels that drop while it is away hit the floor).
"""

from __future__ import annotations

import json
import math
import random
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from . import layout
from .layout import CAGE, CONVEYOR as CONV
from .scene import N_SLATS, SLAT_PITCH, SceneIndex

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
ATTACH_Q, RELEASE_Q = 0.45, 0.62  # gripper target (rad) hysteresis for the magnet assist
ASSIST_RADIUS = 0.024
PARKED_GROUP = 5  # a geom group the viewer does not draw
BELT_FRICTION = 0.05  # must match the belt geom's friction in scene.py (the belt has contact priority)
SLAT_SPAN = N_SLATS * SLAT_PITCH


@dataclass
class Event:
    t: float
    kind: str  # "good" | "bad" | "info" | "big"
    text: str  # English (3D/HUD - MuJoCo fonts are ASCII only)
    zh: str  # Chinese for the terminal
    pos: tuple[float, float, float] | None = None
    points: int = 0


@dataclass
class Parcel:
    i: int
    ptype: layout.ParcelType
    state: str = "parked"  # parked | belt | loose | held | scored | caged
    spawn_t: float = 0.0
    true_dest: str | None = None
    scanned: bool = False
    scan_s: float = 0.0
    fall_speed: float = 0.0
    settle_s: float = 0.0
    recycle_at: float | None = None
    touched: bool = False
    clutched: bool = False
    released: bool = False  # past the metering gate
    off_end: bool = False  # rode off the downstream end of the belt

    @property
    def dest(self) -> str | None:
        if self.ptype.key == "mystery":
            return self.true_dest if self.scanned else None
        return self.ptype.dest


@dataclass
class Stats:
    score: int = 0
    combo: int = 0
    max_combo: int = 0
    sorted_ok: int = 0
    errors: int = 0
    vip_saved: int = 0
    gentle: int = 0
    cracked: int = 0
    scans: int = 0
    dropped: int = 0
    missed: int = 0
    spilled: int = 0
    clutch: int = 0

    @property
    def multiplier(self) -> int:
        return min(layout.MAX_MULTIPLIER, 1 + self.combo // layout.COMBO_STEP)


@dataclass
class Telemetry:
    source: str = "-"
    rate_hz: float = 0.0
    age_ms: float = 0.0
    sync_err_deg: float = 0.0
    blocked_s: float = 0.0
    extra: dict = field(default_factory=dict)


class Game:
    def __init__(
        self,
        m: mujoco.MjModel,
        d: mujoco.MjData,
        idx: SceneIndex,
        mode: str = "live",
        difficulty: str = "normal",
        shift_s: float = 150.0,
        assist: bool = True,
        hints: bool = True,
        player: str = "operator",
        seed: int | None = None,
        sync_pose: np.ndarray | None = None,
        hall: str = "builtin",
    ):
        from .fleet import Fleet

        self.m, self.d, self.idx = m, d, idx
        self.mode = mode
        self.diff = layout.DIFFICULTY[difficulty]
        self.difficulty = difficulty
        self.shift_s = shift_s
        self.assist = assist
        self.hints = hints
        self.player = player
        self.rng = random.Random(seed)
        self.sync_pose = sync_pose if sync_pose is not None else np.zeros(6)
        self.parcels = [Parcel(i, layout.PARCEL_TYPE_BY_KEY[k]) for i, k in enumerate(idx.parcel_types)]
        self.events: list[Event] = []
        self.event_log: deque[Event] = deque(maxlen=6)
        self.stats = Stats()
        self.tele = Telemetry()
        self.phase = "countdown" if mode == "demo" else "lobby"
        self.phase_t = 0.0
        self.shift_elapsed = 0.0
        self.unloaded = 0
        self.belt_speed = 0.0
        self.belt_offset = 0.0
        self.next_release = 0.0
        self.gate_open_until = -1.0
        self.welded: int | None = None
        self.held: int | None = None
        self.q_target = sync_pose.copy() if sync_pose is not None else np.zeros(6)
        self.sync_hold = 0.0
        self.pinch_times: deque[float] = deque(maxlen=4)
        self._grip_closed = False
        self.results: dict | None = None
        self.leaderboard = self._load_leaderboard()
        self.autopilot_skips: dict[int, float] = {}
        self.quit_requested = False
        # recycling cage
        self.cage_present = True
        self.cage_count = 0
        self.cage_pos = np.array(CAGE.center, float)
        for p in self.parcels:
            m.geom_group[idx.parcel_geom[p.i]] = PARKED_GROUP
        self.show_agv_cargo([])
        self._show_cage_cargo(0)
        self.button_latched = {b.key: False for b in layout.BUTTONS}
        self.button_flash: dict[str, float] = {b.key: -1.0 for b in layout.BUTTONS}
        self.button_presses = {b.key: 0 for b in layout.BUTTONS}
        self.fleet = Fleet(m, d, idx, hall, self)
        if self.phase == "countdown":
            self._prestock()

    # ------------------------------------------------------------------------------------------
    # inputs / commands
    # ------------------------------------------------------------------------------------------
    def command(self, cmd: str) -> None:
        if cmd == "start" and self.phase in ("ready", "results"):
            self._to("countdown")
        elif cmd == "start" and self.phase == "lobby" and self.mode == "live":
            self._emit("info", "Waiting for arm signal...", "还没收到机械臂信号，先连上主从臂")
        elif cmd == "assist":
            self.assist = not self.assist
            self._emit(
                "info",
                f"Magnet assist {'ON' if self.assist else 'OFF'}",
                f"磁吸辅助 {'开' if self.assist else '关'}",
            )
        elif cmd == "hints":
            self.hints = not self.hints
            self._emit(
                "info", f"Hints {'ON' if self.hints else 'OFF'}", f"提示 {'开' if self.hints else '关'}"
            )
        elif cmd == "reset":
            self._clear_field()
            self._to("ready" if self.mode == "live" and self.phase != "lobby" else self.phase)
        elif cmd in ("call_agv", "call_tug"):
            self.call(cmd[5:], "key")
        elif cmd == "quit":
            self.quit_requested = True

    def set_target(self, q: np.ndarray) -> None:
        self.q_target = np.clip(q, self.idx.joint_ranges[:, 0], self.idx.joint_ranges[:, 1])

    # ------------------------------------------------------------------------------------------
    # per-frame: controls before stepping, bookkeeping after
    # ------------------------------------------------------------------------------------------
    @property
    def belt_running(self) -> bool:
        return self.phase in ("shift", "lobby")

    def pre_step(self, dt: float) -> None:
        d, idx = self.d, self.idx
        d.ctrl[idx.arm_act] = self.q_target
        v_goal = self._belt_goal() if self.belt_running else 0.0
        self.belt_speed += np.clip(v_goal - self.belt_speed, -0.2 * dt, 0.2 * dt)
        self.belt_offset = (self.belt_offset + self.belt_speed * dt) % SLAT_SPAN
        fast = CONV.fast_speed * (self.belt_speed / max(1e-6, v_goal) if v_goal > 0 else 0.0)
        d.xfrc_applied[idx.parcel_body] = 0.0
        hold = self._queue_holds()
        for p in self.parcels:
            if p.state != "belt":
                continue
            b = idx.parcel_body[p.i]
            pos = d.xpos[b]
            vel = d.qvel[idx.parcel_dof[p.i] : idx.parcel_dof[p.i] + 3]
            mass = p.ptype.mass
            if p.i in hold:  # waiting in the infeed queue: creep up to its stop and stay there
                v_belt = float(np.clip(3.0 * (pos[1] - hold[p.i]), 0.0, fast))
            elif CONV.pick_lo < pos[1] < CONV.pick_hi:
                v_belt = self.belt_speed
            else:
                v_belt = fast
            # velocity servo + feed-forward for the sliding friction on the slippery belt, so parcels
            # really move at belt speed (the HUD countdowns and the autopilot rely on that)
            ff = -BELT_FRICTION * 9.81 if v_belt > 1e-4 else 0.0
            fy = np.clip(mass * (40.0 * (-v_belt - vel[1]) + ff), -0.6 * mass * 9.81, 0.6 * mass * 9.81)
            fx = mass * (40.0 * (CONV.x - pos[0]) - 6.0 * vel[0])
            d.xfrc_applied[b, :3] = (fx, fy, 0.0)
        for k, mid in enumerate(idx.slat_mocap):
            y = CONV.pick_lo + 0.004 + ((k * SLAT_PITCH - self.belt_offset) % SLAT_SPAN)
            d.mocap_pos[mid] = (CONV.x, y, CONV.top_z + 0.0004)
        gate_up = self.t >= self.gate_open_until
        d.mocap_pos[idx.gate_mocap] = (
            CONV.x,
            CONV.gate_y - 0.022,
            CONV.top_z + (0.012 if gate_up else -0.01),
        )
        self.fleet.step(dt)
        self._animate_lights()

    def post_step(self, dt: float) -> None:
        self._update_buttons()
        self._update_contacts()
        self._update_assist()
        self._update_parcels(dt)
        self._update_phase(dt)
        self._update_telemetry(dt)

    @property
    def t(self) -> float:
        return float(self.d.time)

    # ------------------------------------------------------------------------------------------
    # phases
    # ------------------------------------------------------------------------------------------
    def _to(self, phase: str) -> None:
        self.phase, self.phase_t = phase, 0.0
        if phase == "countdown":
            self._clear_field()
            self.stats = Stats()
            self.results = None
            self.shift_elapsed = 0.0
            self.unloaded = 0
            self._prestock()
            self._emit("big", "SHIFT STARTS IN 3", "夜班即将开始：3、2、1！")
        elif phase == "shift":
            self.next_release = self.t + 0.3
            self._emit("big", "GO! SORT THOSE PARCELS", "开工！把包裹分进对应颜色的箱子")
        elif phase == "sync":
            self._clear_field()
            self._emit("big", "MATCH THE HOLOGRAM", "信号已接入：把主臂摆成全息影子的姿态完成对齐")
        elif phase == "ready":
            self._emit("big", "TWIN LINKED", "孪生链路已建立 ✓ 双捏夹爪（或按 Enter）开始上班")
        elif phase == "paused":
            self._emit("bad", "SIGNAL LOST - SHIFT PAUSED", "机械臂信号中断，班次暂停")
        elif phase == "results":
            self._finish_shift()

    def _update_phase(self, dt: float) -> None:
        self.phase_t += dt
        live_ok = self.tele.age_ms < 500 and self.tele.rate_hz > 5
        if self.mode == "demo":
            if self.phase == "countdown" and self.phase_t >= 3.0:
                self._to("shift")
            elif self.phase == "shift":
                self._tick_shift(dt)
            elif self.phase == "results" and self.phase_t > 8.0:
                self._to("countdown")
            return
        if self.phase == "lobby":
            self._tick_lobby()
            if live_ok:
                self._to("sync")
        elif self.phase == "sync":
            err = np.degrees(np.max(np.abs(self.q_target[:5] - self.sync_pose[:5])))
            self.sync_hold = self.sync_hold + dt if err < layout.SYNC_TOLERANCE_DEG else 0.0
            if self.sync_hold >= layout.SYNC_HOLD_S:
                self._to("ready")
            elif not live_ok and self.phase_t > 2:
                self._to("lobby")
        elif self.phase in ("ready", "results"):
            if self._double_pinch():
                self._to("countdown")
        elif self.phase == "countdown":
            if self.phase_t >= 3.0:
                self._to("shift")
        elif self.phase == "shift":
            if not live_ok:
                self._to("paused")
                return
            self._tick_shift(dt)
        elif self.phase == "paused":
            if live_ok:
                self.phase = "shift"
                self._emit("info", "SIGNAL BACK - RESUME", "信号恢复，继续")

    def _tick_lobby(self) -> None:
        """Attract mode: the autopilot sorts a calm trickle of parcels, scores are not kept."""
        self._meter(6.0)

    def _tick_shift(self, dt: float) -> None:
        self.shift_elapsed += dt
        if self.shift_elapsed >= self.shift_s:
            self._to("results")
            return
        a, b = self.diff["interval"]
        self._meter(a + (b - a) * self.progress)

    @property
    def progress(self) -> float:
        return min(1.0, self.shift_elapsed / max(1.0, self.shift_s))

    @property
    def time_left(self) -> float:
        return max(0.0, self.shift_s - self.shift_elapsed)

    def _belt_goal(self) -> float:
        a, b = self.diff["belt"]
        if self.phase == "lobby":
            return a
        return a + (b - a) * self.progress

    # ------------------------------------------------------------------------------------------
    # the line: infeed queue -> metering gate -> pick section -> outfeed -> recycling cage
    # ------------------------------------------------------------------------------------------
    def queue(self) -> list[Parcel]:
        """Parcels waiting on the infeed (not yet through the gate), front first."""
        q = [p for p in self.parcels if p.state == "belt" and not p.released]
        return sorted(q, key=lambda p: self._pos(p)[1])

    def _queue_holds(self) -> dict[int, float]:
        """Stop line (y) for every queued parcel: the gate for the first, the one ahead + pitch after."""
        holds, stop = {}, CONV.gate_y
        for p in self.queue():
            holds[p.i] = stop
            stop = max(stop, self._pos(p)[1]) + CONV.queue_pitch
        return holds

    def _meter(self, interval: float) -> None:
        """Release the front parcel through the gate every ``interval`` seconds (a steady rhythm)."""
        q = self.queue()
        if not q or not self.belt_running:
            return
        front = q[0]
        at_gate = self._pos(front)[1] < CONV.gate_y + 0.01
        due = self.t >= self.next_release
        if at_gate and due:
            front.released = True
            front.spawn_t = self.t  # VIP clocks start when the parcel reaches the operator
            self.next_release = self.t + interval
            self.gate_open_until = self.t + 0.6
            if front.ptype.key == "vip" and self.phase == "shift":
                self._emit(
                    "big",
                    "GOLD VIP PARCEL INBOUND!",
                    f"金色 VIP 包裹来了！{front.ptype.timer_s:.0f} 秒内送进加急箱",
                )

    def next_batch(self, n: int) -> list[str]:
        """What the AGV picks up at the racks (the first parcels of a shift are a mini tutorial)."""
        tutorial = ("express", "standard", "fragile")
        out = []
        vip_busy = self._active("vip")
        for _ in range(n):
            if self.phase in ("countdown", "shift") and self.unloaded + len(out) < len(tutorial):
                out.append(tutorial[self.unloaded + len(out)])
                continue
            types = [t for t in layout.PARCEL_TYPES if not (t.key == "vip" and (vip_busy or "vip" in out))]
            out.append(self.rng.choices([t.key for t in types], weights=[t.weight for t in types])[0])
        return out

    def infeed_has_room(self) -> bool:
        if not self.belt_running:
            return False
        q = self.queue()
        if len(q) >= CONV.queue_max:
            return False
        return all(self._pos(p)[1] < CONV.y_in - 0.09 for p in q)

    def unload(self, key: str, y: float | None = None) -> bool:
        """Put a parcel of type ``key`` (or a stand-in) on the head of the belt."""
        free = [p for p in self.parcels if p.state == "parked" and p.ptype.key == key]
        if not free:
            free = [p for p in self.parcels if p.state == "parked" and p.ptype.key in ("express", "standard")]
        if not free:
            return False
        p = self.rng.choice(free)
        y = CONV.y_in - 0.045 if y is None else y
        # squared up, so it slides straight into jaws held across the belt
        self._teleport(
            p, (CONV.x, y, CONV.top_z + p.ptype.size[2] / 2 + 0.002), self.rng.uniform(-0.06, 0.06)
        )
        p.state, p.spawn_t, p.fall_speed, p.settle_s = "belt", self.t, 0.0, 0.0
        p.released, p.off_end = False, False
        self.m.geom_group[self.idx.parcel_geom[p.i]] = 0
        p.true_dest = self.rng.choice(layout.MYSTERY_TRUE_DESTS) if p.ptype.key == "mystery" else None
        self.unloaded += 1
        return True

    def _prestock(self) -> None:
        """A few parcels already wait at the gate when a shift starts (no idle opening)."""
        for k, key in enumerate(self.next_batch(3)):
            self.unload(key, y=CONV.gate_y + 0.002 + k * CONV.queue_pitch)

    def show_agv_cargo(self, keys: list[str]) -> None:
        for k, g in enumerate(self.idx.agv_cargo):
            if k < len(keys):
                self.m.geom_matid[g] = self.idx.parcel_material[keys[k]]
                self.m.geom_group[g] = 1
            else:
                self.m.geom_group[g] = PARKED_GROUP

    def _show_cage_cargo(self, n: int) -> None:
        for k, g in enumerate(self.idx.cage_cargo):
            self.m.geom_group[g] = 1 if k < n else PARKED_GROUP

    # -- recycling cage (driven by the fleet's tug) ----------------------------------------------
    def cage_needs_haul(self) -> bool:
        return self.cage_present and self.cage_count >= CAGE.capacity

    def cage_detach(self) -> None:
        """The tug hooks the cage: its parcels become its (visual) load."""
        n = 0
        for p in self.parcels:
            if p.state == "caged":
                self._park(p)
                n += 1
        self.cage_present = False
        self._show_cage_cargo(min(n, len(self.idx.cage_cargo)))
        self._emit(
            "info", "TUG HAULING THE RECYCLING CAGE", "牵引车拉走了回收笼：这段时间漏拣的包裹会直接摔在地上！"
        )

    def cage_emptied(self) -> None:
        self.cage_count = 0
        self._show_cage_cargo(0)

    def cage_attach(self) -> None:
        self.cage_present = True
        self.move_cage(np.array(CAGE.center), 0.0)
        self._emit("info", "RECYCLING CAGE BACK", "回收笼已归位")

    def move_cage(self, xy, yaw: float) -> None:
        self.cage_pos = np.asarray(xy, float)
        self.d.mocap_pos[self.idx.cage_mocap] = (xy[0], xy[1], layout.FLOOR_Z)
        self.d.mocap_quat[self.idx.cage_mocap] = (math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))

    def _in_cage(self, pos) -> bool:
        cx, cy = CAGE.center
        return (
            self.cage_present
            and abs(pos[0] - cx) < CAGE.half[0]
            and abs(pos[1] - cy) < CAGE.half[1]
            and pos[2] < layout.FLOOR_Z + CAGE.height
        )

    # -- call buttons -----------------------------------------------------------------------------
    def button_depth(self, key: str) -> float:
        return max(0.0, -float(self.d.qpos[self.idx.button_qpos[key]]))

    def _update_buttons(self) -> None:
        """A press = the cap pushed past its trigger depth (rising edge; let go to re-arm)."""
        for b in layout.BUTTONS:
            depth = self.button_depth(b.key)
            if not self.button_latched[b.key] and depth > b.trigger:
                self.button_latched[b.key] = True
                self.button_presses[b.key] += 1
                self.call(b.key, "button")
            elif self.button_latched[b.key] and depth < b.trigger * 0.4:
                self.button_latched[b.key] = False

    def call(self, key: str, how: str) -> bool:
        """CALL AGV / CALL TUG - from the physical button, a key or the terminal."""
        if self.t - self.button_flash[key] < 1.0:
            return False  # debounce (a wobbly press, or key repeat)
        self.button_flash[key] = self.t
        ok, en, zh = self.fleet.call(key)
        b = layout.BUTTON_BY_KEY[key]
        pos = (b.center[0], b.center[1], 0.08)
        self._emit("info" if ok else "bad", en, ("按下按钮：" if how == "button" else "") + zh, pos)
        return ok

    # -- vehicles -> game -------------------------------------------------------------------------
    def vehicle_event(self, kind: str, n: int) -> None:
        if kind == "agv_docked":
            self._emit("info", f"AGV DOCKED: {n} PARCELS", f"供件 AGV 到站，带来 {n} 件包裹")
        elif kind == "tug_dispatched":
            self._emit("info", "RECYCLING CAGE FULL - TUG ON ITS WAY", "回收笼满了，牵引车出发去拉")
        elif kind == "tug_called":
            self._emit("info", f"TUG ON ITS WAY ({n} in the cage)", f"牵引车出发：提前拉走回收笼（{n} 件）")
        elif kind == "agv_auto":
            self._emit(
                "bad",
                "LINE RAN DRY - AGV SENT AUTOMATICALLY",
                "排队区空了，供件车自动出发（下次早点按 CALL AGV）",
            )

    def vehicle_yields(self, who: str, to: str) -> None:
        names = {"agv": ("AGV", "供件 AGV"), "tug": ("TUG", "牵引车"), "forklift": ("FORKLIFT", "叉车")}
        self._emit(
            "info", f"{names[who][0]} GIVES WAY TO {names[to][0]}", f"{names[who][1]}停车让行{names[to][1]}"
        )

    # ------------------------------------------------------------------------------------------
    # parcels
    # ------------------------------------------------------------------------------------------
    def _pos(self, p: Parcel) -> np.ndarray:
        return self.d.xpos[self.idx.parcel_body[p.i]]

    def _vel(self, p: Parcel) -> np.ndarray:
        a = self.idx.parcel_dof[p.i]
        return self.d.qvel[a : a + 3]

    def _teleport(self, p: Parcel, pos, yaw: float = 0.0) -> None:
        a = self.idx.parcel_qpos[p.i]
        self.d.qpos[a : a + 3] = pos
        self.d.qpos[a + 3 : a + 7] = (math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))
        dof = self.idx.parcel_dof[p.i]
        self.d.qvel[dof : dof + 6] = 0.0

    def _park(self, p: Parcel) -> None:
        if self.welded == p.i:
            self._unweld()
        px, py, pz = layout.PARK_ORIGIN
        ix, iy = p.i % 5, p.i // 5
        self._teleport(p, (px - 0.4 + 0.2 * ix, py - 0.4 + 0.2 * iy, pz + p.ptype.size[2] / 2 + 0.002))
        p.state, p.recycle_at, p.scanned, p.scan_s, p.touched = "parked", None, False, 0.0, False
        p.clutched, p.released, p.off_end = False, False, False
        self.autopilot_skips.pop(p.i, None)
        self.m.geom_group[self.idx.parcel_geom[p.i]] = PARKED_GROUP  # not drawn while in the basement

    def _clear_field(self) -> None:
        for p in self.parcels:
            if p.state != "parked":
                self._park(p)
        self.cage_count = 0
        self.next_release = self.t + 1.0

    def _active(self, key: str) -> bool:
        return any(p.state not in ("parked", "scored", "caged") and p.ptype.key == key for p in self.parcels)

    def _on_belt(self, pos) -> bool:
        return (
            abs(pos[0] - CONV.x) < CONV.width / 2 + 0.012
            and CONV.y_out - 0.005 < pos[1] < CONV.y_in + 0.03
            and pos[2] < CONV.top_z + 0.045
        )

    @staticmethod
    def bin_at(pos) -> layout.BinSpec | None:
        for b in layout.BINS:
            cx, cy = b.center
            if (
                abs(pos[0] - cx) < b.inner / 2 + 0.004
                and abs(pos[1] - cy) < b.inner / 2 + 0.004
                and pos[2] < b.height + 0.02
            ):
                return b
        return None

    def _double_pinch(self) -> bool:
        now = self.t
        while self.pinch_times and now - self.pinch_times[0] > 2.0:
            self.pinch_times.popleft()
        if len(self.pinch_times) >= 2:
            self.pinch_times.clear()
            return True
        return False

    def _update_contacts(self) -> None:
        idx = self.idx
        geom_to_parcel = self._geom_to_parcel
        self._pinched: set[int] = set()
        fixed_side, moving_side = self._fixed_geoms, self._moving_geoms
        touch_fixed: set[int] = set()
        touch_moving: set[int] = set()
        for c in self.d.contact[: self.d.ncon]:
            for g, o in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                pi = geom_to_parcel.get(g)
                if pi is None:
                    continue
                if o in idx.robot_geoms:
                    self.parcels[pi].touched = True
                if o in fixed_side:
                    touch_fixed.add(pi)
                elif o in moving_side:
                    touch_moving.add(pi)
        self._pinched = touch_fixed & touch_moving

    @property
    def _geom_to_parcel(self) -> dict[int, int]:
        if not hasattr(self, "_g2p"):
            self._g2p = {int(g): i for i, g in enumerate(self.idx.parcel_geom)}
            m = self.m
            self._fixed_geoms = {m.geom("pad_fixed").id, m.geom("pad_palm").id}
            self._moving_geoms = {m.geom("pad_moving").id}
        return self._g2p

    def _update_assist(self) -> None:
        g = self.q_target[5]
        # pinch gesture bookkeeping (close -> open cycles)
        if g < 0.25:
            self._grip_closed = True
        elif g > 0.8 and self._grip_closed:
            self._grip_closed = False
            self.pinch_times.append(self.t)
        if self.welded is not None and (g > RELEASE_Q or not self.assist):
            self._unweld()
        if self.welded is None and self.assist and g < ATTACH_Q:
            gc = self.d.site_xpos[self.idx.grasp_site]
            best, best_d = None, ASSIST_RADIUS
            for p in self.parcels:
                if p.state in ("parked", "scored"):
                    continue
                dist = float(np.linalg.norm(self._pos(p) - gc))
                if dist < best_d:
                    best, best_d = p, dist
            if best is not None:
                self._weld(best)

    def _weld(self, p: Parcel) -> None:
        idx, d, m = self.idx, self.d, self.m
        pa, qa = d.xpos[idx.gripper_body], d.xquat[idx.gripper_body]
        pb, qb = d.xpos[idx.parcel_body[p.i]], d.xquat[idx.parcel_body[p.i]]
        qa_inv = np.zeros(4)
        mujoco.mju_negQuat(qa_inv, qa)
        rel_p, rel_q = np.zeros(3), np.zeros(4)
        mujoco.mju_rotVecQuat(rel_p, pb - pa, qa_inv)
        mujoco.mju_mulQuat(rel_q, qa_inv, qb)
        e = idx.parcel_weld[p.i]
        m.eq_data[e, 0:3] = 0.0
        m.eq_data[e, 3:6] = rel_p
        m.eq_data[e, 6:10] = rel_q
        m.eq_data[e, 10] = 1.0
        d.eq_active[e] = 1
        self.welded = p.i

    def _unweld(self) -> None:
        if self.welded is not None:
            self.d.eq_active[self.idx.parcel_weld[self.welded]] = 0
            self.welded = None

    def _update_parcels(self, dt: float) -> None:
        self.held = None
        for p in self.parcels:
            if p.state in ("parked", "caged"):
                continue
            pos, vel = self._pos(p), self._vel(p)
            speed = float(np.linalg.norm(vel))
            if p.state == "scored":
                if p.recycle_at is not None and self.t >= p.recycle_at:
                    self._park(p)
                continue
            held = self.welded == p.i or (
                p.i in self._pinched and pos[2] > CONV.top_z + p.ptype.size[2] / 2 + 0.006
            )
            if held:
                if p.state == "belt":
                    self._check_clutch(p, pos)
                self.held = p.i
                p.state, p.settle_s, p.fall_speed = "held", 0.0, 0.0
                p.released = True
                self._scan(p, pos, dt)
            elif self._on_belt(pos):
                p.state = "belt"
            else:
                if p.state == "belt" and pos[1] < CONV.y_out + 0.02:
                    p.off_end = True  # rode off the downstream end
                p.state = "loose"
            # VIP countdown (starts at the gate)
            if (
                p.ptype.timer_s
                and p.released
                and self.phase == "shift"
                and self.t - p.spawn_t > p.ptype.timer_s
            ):
                self._penalty(layout.VIP_EXPIRED_PENALTY, "VIP EXPIRED", "VIP 超时了，客户很生气", tuple(pos))
                self._park(p)
                continue
            if p.state != "loose":
                continue
            if self._in_cage(pos):
                self._caged(p, pos)
                continue
            if pos[2] < layout.FLOOR_Z + 0.2:
                if p.off_end:
                    self.stats.spilled += 1
                    self._penalty(
                        layout.SPILL_PENALTY,
                        "SPILLED - no cage!",
                        "回收笼不在，漏拣的包裹摔在了地上",
                        tuple(pos),
                    )
                else:
                    self.stats.dropped += 1
                    self._penalty(layout.DROP_PENALTY, "DROPPED!", "包裹掉地上了", (pos[0], pos[1], 0.05))
                self._park(p)
                continue
            b = self.bin_at(pos)
            if b is not None:
                p.fall_speed = max(p.fall_speed, -float(vel[2]))
            p.settle_s = p.settle_s + dt if speed < 0.03 else 0.0
            if b is not None and p.settle_s > 0.25:
                self._judge(p, b, pos)
            elif b is None and p.settle_s > 3.0 and -0.01 < pos[2] < 0.08:
                self._penalty(layout.STRAY_PENALTY, "STRAY - recycled", "包裹散落在台面，已回收", tuple(pos))
                self._park(p)

    def _caged(self, p: Parcel, pos) -> None:
        """Rode off the end into the recycling cage: a miss (it stays in the cage until hauled)."""
        p.state = "caged"
        self.cage_count += 1
        if self.phase == "shift":
            self.stats.missed += 1
            if p.ptype.key == "vip":
                self._penalty(
                    layout.VIP_EXPIRED_PENALTY, "VIP MISSED!", "VIP 包裹漏拣，掉进了回收笼", tuple(pos)
                )
            else:
                self._penalty(
                    layout.MISS_PENALTY,
                    f"MISSED {p.ptype.name_en}",
                    f"漏拣：{p.ptype.name_zh}掉进回收笼（{self.cage_count}/{CAGE.capacity}）",
                    tuple(pos),
                )

    def _check_clutch(self, p: Parcel, pos) -> None:
        """Snatched off the belt inside the red last-chance stretch."""
        if self.phase != "shift" or p.clutched or pos[1] > CONV.reach_lo + CONV.last_chance:
            return
        p.clutched = True
        self.stats.clutch += 1
        self.stats.score += layout.CLUTCH_BONUS
        self._emit(
            "good",
            f"+{layout.CLUTCH_BONUS} CLUTCH SAVE",
            f"+{layout.CLUTCH_BONUS} 极限救回！差一点就漏拣",
            (pos[0], pos[1], 0.09),
            layout.CLUTCH_BONUS,
        )

    def time_to_slot(self, p: Parcel) -> float | None:
        """Seconds until a released belt parcel leaves the robot's reach (None if not applicable)."""
        if p.state != "belt" or not p.released or self.belt_speed < 1e-3:
            return None
        y = self._pos(p)[1]
        if y < CONV.reach_lo or y > CONV.reach_hi + 0.05:
            return None
        return max(0.0, (y - CONV.reach_lo) / self.belt_speed)

    def _scan(self, p: Parcel, pos, dt: float) -> None:
        if p.ptype.key != "mystery" or p.scanned:
            return
        s = layout.SCANNER
        inside = (
            abs(pos[0] - s.center[0]) < s.half
            and abs(pos[1] - s.center[1]) < s.half
            and s.z_range[0] < pos[2] < s.z_range[1]
        )
        p.scan_s = p.scan_s + dt if inside else 0.0
        if p.scan_s >= s.dwell_s:
            p.scanned = True
            self.stats.scans += 1
            dest = layout.BIN_BY_KEY[p.true_dest]
            self._emit(
                "info",
                f"SCANNED -> {dest.name_en}",
                f"扫描完成：这是{dest.name_zh}，送去{dest.name_zh}箱",
                tuple(pos),
            )

    def _judge(self, p: Parcel, b: layout.BinSpec, pos) -> None:
        p.state, p.recycle_at = "scored", self.t + 1.4
        if self.phase != "shift":
            return
        pt, st = p.ptype, self.stats
        top = (pos[0], pos[1], b.height + 0.05)
        if pt.key == "mystery" and not p.scanned:
            ok = b.key == "qc"
            if ok:
                self._reward(layout.QC_SAFE_POINTS, "QC +SAFE", "异常件稳妥送检", top)
            else:
                self._penalty(
                    layout.WRONG_BIN_PENALTY, f"WRONG BIN ({b.name_en})", "没扫描就乱投，分错了", top
                )
            return
        dest = p.dest
        if b.key == "qc" and pt.key == "mystery":  # scanned but played safe
            self._reward(layout.QC_SAFE_POINTS, "QC +SAFE", "扫过了还送检？稳妥但分少", top)
            return
        if b.key != dest:
            self._penalty(
                layout.WRONG_BIN_PENALTY,
                f"WRONG BIN! {pt.name_en}",
                f"分错箱：{pt.name_zh}应该进{layout.BIN_BY_KEY[dest].name_zh}箱",
                top,
            )
            return
        if pt.gentle and p.fall_speed > layout.CRACK_SPEED:
            st.cracked += 1
            self._penalty(
                layout.CRACK_PENALTY,
                f"CRACK! {p.fall_speed:.1f} m/s",
                f"易碎件摔裂了（落地 {p.fall_speed:.1f} m/s），要轻放",
                top,
            )
            return
        bonus_txt, bonus_zh, pts = "", "", pt.points
        if pt.gentle:
            st.gentle += 1
            bonus_txt, bonus_zh = " GENTLE", "，轻拿轻放"
        if pt.key == "vip":
            st.vip_saved += 1
            left = pt.timer_s - (self.t - p.spawn_t)
            pts += int(10 * max(0.0, left))
            bonus_txt, bonus_zh = f" {left:.0f}s LEFT", f"，还剩 {left:.0f} 秒"
        if pt.key == "mystery":
            bonus_txt, bonus_zh = " DECODED", "，扫描破译"
        st.sorted_ok += 1
        st.combo += 1
        st.max_combo = max(st.max_combo, st.combo)
        mult = st.multiplier
        gained = pts * mult
        st.score += gained
        mult_txt = f" x{mult}" if mult > 1 else ""
        self._emit(
            "good",
            f"+{gained}{mult_txt}{bonus_txt}",
            f"+{gained} {pt.name_zh}入库{bonus_zh}（连击 {st.combo}）",
            top,
            gained,
        )
        if st.combo in (5, 10, 15, 20):
            self._emit("big", f"COMBO {st.combo}! x{mult}", f"{st.combo} 连击！倍率 x{mult}")

    def _reward(self, pts: int, text: str, zh: str, pos) -> None:
        st = self.stats
        st.sorted_ok += 1
        st.score += pts
        self._emit("good", f"+{pts} {text}", f"+{pts} {zh}", pos, pts)

    def _penalty(self, pts: int, text: str, zh: str, pos=None) -> None:
        if self.phase != "shift":
            return
        st = self.stats
        st.errors += 1
        st.combo = 0
        st.score += pts
        self._emit("bad", f"{pts} {text}", f"{pts} {zh}（连击清零）", pos, pts)

    def _emit(self, kind: str, text: str, zh: str, pos=None, points: int = 0) -> None:
        ev = Event(self.t, kind, text, zh, tuple(pos) if pos is not None else None, points)
        self.events.append(ev)
        self.event_log.append(ev)

    def pop_events(self) -> list[Event]:
        ev, self.events = self.events, []
        return ev

    # ------------------------------------------------------------------------------------------
    # results / leaderboard
    # ------------------------------------------------------------------------------------------
    def _finish_shift(self) -> None:
        st = self.stats
        letter, en, zh = layout.rank_for(st.score, self.shift_s)
        total = st.sorted_ok + st.errors
        self.results = {
            "player": self.player if self.mode == "live" else "AUTOPILOT",
            "score": st.score,
            "rank": letter,
            "title_en": en,
            "title_zh": zh,
            "accuracy": round(100.0 * st.sorted_ok / total, 1) if total else 0.0,
            "max_combo": st.max_combo,
            "sorted": st.sorted_ok,
            "errors": st.errors,
            "vip": st.vip_saved,
            "cracked": st.cracked,
            "missed": st.missed,
            "spilled": st.spilled,
            "clutch": st.clutch,
            "difficulty": self.difficulty,
            "assist": self.assist,
            "date": time.strftime("%Y-%m-%d %H:%M"),
        }
        if self.mode == "live":
            self.leaderboard.append(self.results)
            self.leaderboard.sort(key=lambda r: -r["score"])
            self.leaderboard = self.leaderboard[:10]
            self._save_leaderboard()
        place = next((i + 1 for i, r in enumerate(self.leaderboard) if r is self.results), None)
        self.results["place"] = place
        self._emit(
            "big",
            f"SHIFT OVER - RANK {letter}: {en}",
            f"下班！得分 {st.score}，评级 {letter}「{zh}」" + (f"，排行榜第 {place} 名" if place else ""),
        )

    def _load_leaderboard(self) -> list[dict]:
        try:
            return json.loads((DATA_DIR / "leaderboard.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []

    def _save_leaderboard(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        (DATA_DIR / "leaderboard.json").write_text(
            json.dumps(self.leaderboard, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    # ------------------------------------------------------------------------------------------
    # autopilot hooks
    # ------------------------------------------------------------------------------------------
    def autopilot_pick(self, lead_s: float = 0.0) -> int | None:
        """Parcel the autopilot should go for. ``lead_s``: how long it needs to get there."""
        best, best_key = None, None
        for p in self.parcels:
            if p.state not in ("belt", "loose") or self.autopilot_skips.get(p.i, -1) > self.t:
                continue
            pos = self._pos(p)
            if p.state == "belt":
                if not p.released:
                    continue
                y_then = pos[1] - self.belt_speed * lead_s  # where it will be when the jaws arrive
                if pos[1] > CONV.reach_hi + 0.12 or y_then < CONV.reach_lo + 0.03:
                    continue
            else:  # loose on the table, within reach and not moving
                r = math.hypot(pos[0] - layout.PAN_AXIS_XY[0], pos[1] - layout.PAN_AXIS_XY[1])
                if (
                    self.bin_at(pos) is not None
                    or not (-0.01 < pos[2] < 0.08)
                    or r > 0.3
                    or float(np.linalg.norm(self._vel(p))) > 0.05
                ):
                    continue
            key = (0 if p.ptype.key == "vip" else 1, pos[1])
            if best_key is None or key < best_key:
                best, best_key = p.i, key
        return best

    def _free_for(self, seconds: float) -> bool:
        """No parcel will need the arm within ``seconds`` (released ones, and the next release)."""
        v = max(self.belt_speed, 1e-3)
        for p in self.parcels:
            if p.state == "belt" and p.released:
                y = self._pos(p)[1]
                if y > CONV.reach_lo and (y - CONV.reach_hi) / v < seconds:
                    return False
        if self.queue():
            arrive = max(0.0, self.next_release - self.t) + (CONV.gate_y - CONV.reach_hi) / v
            if arrive < seconds:
                return False
        return True

    def autopilot_button(self) -> str | None:
        """Which CALL button the autopilot should press now (only when no parcel needs the arm)."""
        if self.held is not None or self.phase not in ("shift", "lobby") or not self._free_for(1.6):
            return None
        fl = self.fleet
        if fl.state["agv"] == "ready" and not fl.called["agv"] and len(self.queue()) <= 3:
            return "agv"
        if fl.state["tug"] == "parked" and not fl.called["tug"] and self.cage_count >= 2:
            return "tug"
        return None

    def autopilot_skip(self, i: int, seconds: float = 3.0) -> None:
        self.autopilot_skips[i] = self.t + seconds

    def needs_scan(self, i: int) -> bool:
        p = self.parcels[i]
        return p.ptype.key == "mystery" and not p.scanned

    def autopilot_dest(self, i: int) -> str:
        return self.parcels[i].dest or "qc"

    # ------------------------------------------------------------------------------------------
    # telemetry + eye candy
    # ------------------------------------------------------------------------------------------
    def _update_telemetry(self, dt: float) -> None:
        q = self.d.qpos[self.idx.arm_qpos[:5]]
        qd = self.d.qvel[self.idx.arm_dof[:5]]
        err = np.degrees(np.abs(q - self.q_target[:5]))
        self.tele.sync_err_deg = float(err.max())
        # "blocked" = a joint is far from where the real arm is AND not moving towards it (stalled),
        # i.e. the twin is pushing against something that only exists in the simulation.
        stalled = bool(np.any((err > 10.0) & (np.abs(qd) < 0.25)))
        self.tele.blocked_s = self.tele.blocked_s + dt if stalled else 0.0
        if self.tele.blocked_s > 0.4 and not getattr(self, "_blocked_warned", False):
            self._blocked_warned = True
            self._emit("bad", "VIRTUAL COLLISION - twin blocked", "孪生臂撞到虚拟物体了（真实臂感觉不到！）")
        elif self.tele.blocked_s == 0.0:
            self._blocked_warned = False

    def _animate_lights(self) -> None:
        m, idx = self.m, self.idx
        alarm = not self.cage_present and self.belt_running  # misses would hit the floor
        vip = self._active("vip") and self.phase == "shift"
        blink = (math.sin(self.t * 12) > 0) if (alarm or vip) else False
        col = (1.0, 0.15, 0.05) if alarm else (1.0, 0.7, 0.05)
        m.geom_rgba[idx.beacon_geom, :3] = col if blink else (0.35, 0.25, 0.05)
        m.light_diffuse[idx.beacon_light] = np.array(col) * 1.6 if blink else 0.0
        fl = self.fleet
        for b in layout.BUTTONS:
            g = idx.button_light[b.key]
            if self.t - self.button_flash[b.key] < 0.4:
                m.geom_rgba[g, :3] = (1.0, 1.0, 1.0)  # flash on press
                continue
            if b.key == "agv":
                lit = fl.state["agv"] == "ready" and not fl.called["agv"]
                pulse = lit and not self.queue()  # line dry: hurry
            else:
                lit = fl.state["tug"] == "parked" and self.cage_count > 0 and not fl.called["tug"]
                pulse = lit and self.cage_count >= CAGE.capacity - 1
            k = 1.0 if lit else 0.2
            if pulse:
                k = 0.5 + 0.5 * (math.sin(self.t * 10) > 0)
            m.geom_rgba[g, :3] = np.array(b.rgba[:3]) * k
        gate_open = self.t < self.gate_open_until
        m.geom_rgba[idx.tunnel_light_geom, :3] = (0.2, 1.0, 0.4) if gate_open else (1.0, 0.45, 0.1)
