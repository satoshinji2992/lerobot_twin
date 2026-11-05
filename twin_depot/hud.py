"""Everything drawn on top of the physics: overlay text, hologram ghost arm, glowing target bins,
floating score pop-ups and confetti. MuJoCo's built-in font is ASCII-only, so in-viewer text is
English; the terminal mirrors every event in Chinese."""

from __future__ import annotations

import math
import random
import sys

import mujoco
import numpy as np

from . import layout
from .game import Event, Game

FONT = mujoco.mjtFontScale
GRID = mujoco.mjtGridPos

KIND_RGBA = {
    "good": (0.3, 1.0, 0.45, 1.0),
    "bad": (1.0, 0.3, 0.25, 1.0),
    "info": (0.5, 0.85, 1.0, 1.0),
    "big": (1.0, 0.85, 0.2, 1.0),
}


class Hud:
    def __init__(self, m: mujoco.MjModel, game: Game, ghost_pose: np.ndarray):
        self.m, self.game = m, game
        self.ghost_d = mujoco.MjData(m)
        self.ghost_d.qpos[game.idx.arm_qpos] = ghost_pose
        # park ghost parcels out of sight; only robot visuals (geom group 2) are drawn anyway
        mujoco.mj_kinematics(m, self.ghost_d)
        self.ghost_opt = mujoco.MjvOption()
        self.ghost_opt.geomgroup[:] = 0
        self.ghost_opt.geomgroup[2] = 1
        self.pert = mujoco.MjvPerturb()
        self.popups: list[tuple[Event, float]] = []
        self.banner: tuple[Event, float] | None = None
        self.confetti: list[list] = []  # [pos(3), vel(3), rgba, t_end]
        self.rng = random.Random(1)
        self.color = sys.stdout.isatty()

    # ------------------------------------------------------------------------------------------
    def on_events(self, events: list[Event], now: float) -> None:
        for ev in events:
            self._print(ev)
            if ev.kind == "big":
                self.banner = (ev, now + 2.8)
            if ev.pos is not None:
                self.popups.append((ev, now))
            if ev.kind == "good" and ev.pos is not None:
                self._burst(ev.pos, big=ev.points >= 200)

    def _print(self, ev: Event) -> None:
        tag = {"good": "32", "bad": "31", "info": "36", "big": "33;1"}[ev.kind]
        line = f"[{ev.t:7.1f}s] {ev.zh}"
        print(f"\033[{tag}m{line}\033[0m" if self.color else line, flush=True)

    def _burst(self, pos, big: bool) -> None:
        palette = [(1, 0.3, 0.3, 1), (1, 0.85, 0.2, 1), (0.3, 1, 0.5, 1), (0.3, 0.6, 1, 1), (1, 0.4, 1, 1)]
        for _ in range(70 if big else 32):
            a = self.rng.uniform(0, 2 * math.pi)
            sp = self.rng.uniform(0.15, 0.45 if big else 0.32)
            vel = [sp * math.cos(a) * 0.6, sp * math.sin(a) * 0.6, self.rng.uniform(0.6, 1.2 if big else 0.9)]
            self.confetti.append(
                [np.array(pos, float), np.array(vel), self.rng.choice(palette), self.game.t + 1.6]
            )

    # ------------------------------------------------------------------------------------------
    def draw(self, viewer) -> None:
        g = self.game
        scn = viewer.user_scn
        scn.ngeom = 0
        now = g.t
        if g.phase == "sync":
            self._ghost(scn, alpha=0.28 + 0.1 * math.sin(now * 4))
        self._targets(scn, now)
        self._parcel_tags(scn)
        self._fleet_tags(scn)
        self._popups(scn, now)
        self._confetti(scn, now)
        if g.tele.blocked_s > 0.3:
            self._geom(
                scn,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                (0.02, 0, 0),
                g.d.site_xpos[g.idx.tcp_site],
                (1, 0.1, 0.1, 0.5),
                "BLOCKED",
            )

    def _geom(self, scn, gtype, size, pos, rgba, label: str | None = None, mat=None) -> None:
        if scn.ngeom >= scn.maxgeom:
            return
        geom = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            geom,
            gtype,
            np.asarray(size, float),
            np.asarray(pos, float),
            np.eye(3).flatten() if mat is None else np.asarray(mat, float).flatten(),
            np.asarray(rgba, np.float32),
        )
        if label:
            geom.label = label
        scn.ngeom += 1

    def _ghost(self, scn, alpha: float) -> None:
        start = scn.ngeom
        mujoco.mjv_addGeoms(
            self.m, self.ghost_d, self.ghost_opt, self.pert, mujoco.mjtCatBit.mjCAT_DYNAMIC, scn
        )
        for k in range(start, scn.ngeom):
            scn.geoms[k].rgba[:] = (0.25, 0.9, 1.0, alpha)
            scn.geoms[k].matid = -1

    def _targets(self, scn, now: float) -> None:
        g = self.game
        if not g.hints or g.held is None:
            return
        p = g.parcels[g.held]
        dest = p.dest
        pulse = 0.22 + 0.12 * math.sin(now * 8)
        if dest is None:  # mystery: point at the scanner (or QC as the safe option)
            s = layout.SCANNER
            self._geom(
                scn,
                mujoco.mjtGeom.mjGEOM_BOX,
                (s.half, s.half, 0.06),
                (*s.center, 0.07),
                (1, 0.2, 0.2, pulse),
                "SCAN ME (or QC)",
            )
            return
        b = layout.BIN_BY_KEY[dest]
        rot = np.eye(3)
        h = b.inner / 2 + b.wall
        self._geom(
            scn,
            mujoco.mjtGeom.mjGEOM_BOX,
            (h, h, 0.035),
            (*b.center, b.height + 0.035),
            (*b.rgba[:3], pulse),
            None,
            rot,
        )
        self._geom(
            scn,
            mujoco.mjtGeom.mjGEOM_ARROW,
            (0.006, 0.006, 0.04),
            (*b.center, b.height + 0.11),
            (*b.rgba[:3], 0.9),
            None,
            np.diag([1, -1, -1]),
        )

    def _parcel_tags(self, scn) -> None:
        g = self.game
        for p in g.parcels:
            if p.state in ("parked", "scored"):
                continue
            pos = g._pos(p) + (0, 0, 0.035)
            eta = g.time_to_slot(p)
            if eta is not None and eta < 2.5 and g.phase in ("shift", "lobby"):
                # flow mode: about to ride into the return slot
                self._geom(
                    scn,
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    (0.004, 0, 0),
                    pos + (0, 0, 0.012),
                    (1, 0.15 if eta < 1.0 else 0.6, 0.05, 0.9),
                    f"GRAB! {eta:3.1f}s",
                )
            elif p.ptype.timer_s and g.phase == "shift":
                left = p.ptype.timer_s - (g.t - p.spawn_t)
                label = f"VIP {left:4.1f}s" if p.released else "VIP (clock starts at the gate)"
                self._geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, (0.001, 0, 0), pos, (1, 1, 1, 0), label)
            elif p.ptype.key == "mystery":
                if p.scanned:
                    label = f"-> {layout.BIN_BY_KEY[p.true_dest].name_en}"
                elif p.scan_s > 0:
                    label = f"SCANNING {int(100 * p.scan_s / layout.SCANNER.dwell_s)}%"
                else:
                    label = "?"
                self._geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, (0.001, 0, 0), pos, (1, 1, 1, 0), label)

    def _fleet_tags(self, scn) -> None:
        g = self.game
        for v in g.fleet.vehicles.values():
            txt = v.label + (f"  (giving way to {v.yielding_to})" if v.yielding_to else "")
            pos = (v.pos[0], v.pos[1], layout.FLOOR_Z + (2.4 if v.name == "forklift" else 1.15))
            self._geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, (0.001, 0, 0), pos, (1, 1, 1, 0), txt)
        for b in layout.BUTTONS:
            st, called = g.fleet.state[b.key], g.fleet.called[b.key]
            if b.key == "agv":
                status = {"ready": "AGV READY", "loading": "AGV loading", "to_racks": "AGV at racks"}.get(
                    st, "AGV on its way"
                )
            else:
                status = "TUG BUSY" if st != "parked" else f"cage {g.cage_count}/{layout.CAGE.capacity}"
            txt = f"{b.label}: {'CALLED' if called else status}"
            self._geom(
                scn,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                (0.001, 0, 0),
                (*b.center, 0.095 if b.key == "agv" else 0.06),
                (1, 1, 1, 0),
                txt,
            )
        cg = layout.CAGE
        if g.cage_present:
            full = g.cage_count >= cg.capacity
            txt = f"RECYCLE {g.cage_count}/{cg.capacity}" + ("  FULL" if full else "")
        else:
            txt = "RECYCLE CAGE AWAY - misses hit the floor!"
        pos = (cg.center[0], cg.center[1], layout.FLOOR_Z + cg.height + 0.25)
        self._geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, (0.001, 0, 0), pos, (1, 1, 1, 0), txt)

    def _popups(self, scn, now: float) -> None:
        keep = []
        for ev, t0 in self.popups:
            age = now - t0
            if age > 1.8 or age < 0:
                continue
            keep.append((ev, t0))
            pos = np.array(ev.pos) + (0, 0, 0.02 + 0.05 * age)
            self._geom(
                scn,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                (0.004, 0, 0),
                pos,
                (*KIND_RGBA[ev.kind][:3], max(0.0, 0.9 - age / 2)),
                ev.text,
            )
        self.popups = keep

    def _confetti(self, scn, now: float) -> None:
        dt = 1 / 60
        alive = []
        for c in self.confetti:
            if now > c[3]:
                continue
            c[1][2] -= 9.81 * dt * 0.6
            c[0] = c[0] + c[1] * dt
            alive.append(c)
            self._geom(scn, mujoco.mjtGeom.mjGEOM_BOX, (0.006, 0.006, 0.0015), c[0], c[2])
        self.confetti = alive

    # ------------------------------------------------------------------------------------------
    def texts(self, now: float):
        """Overlay text. Call *outside* ``viewer.lock()`` - ``set_texts`` takes the same lock."""
        g, st, tele = self.game, self.game.stats, self.game.tele
        out = []
        mm, ss = divmod(int(math.ceil(g.time_left)), 60)
        phase_txt = {
            "lobby": "ATTRACT MODE - waiting for arm signal",
            "sync": "SYNC: move the leader arm into the cyan hologram",
            "ready": "READY: double-pinch the gripper (or ENTER) to start",
            "countdown": f"STARTING IN {max(1, 3 - int(g.phase_t))}",
            "shift": "ON SHIFT" + ("" if g.cage_present else "  -  RECYCLING CAGE AWAY!"),
            "paused": "PAUSED - arm signal lost",
            "results": "SHIFT OVER - double-pinch / ENTER to play again",
        }[g.phase]
        if g.mode == "demo":
            phase_txt = "AUTOPILOT DEMO  |  " + phase_txt
        cage = f"{g.cage_count}/{layout.CAGE.capacity}" if g.cage_present else "AWAY"
        line = f"{g.belt_speed * 100:3.1f} cm/s  queue {len(g.queue())}  missed {st.missed}  cage {cage}"
        left = "TWIN DEPOT\nPHASE\nSCORE\nCOMBO\nTIME\nSORTED / ERR\nLINE"
        right = (
            f"SO-101 digital twin sorting\n{phase_txt}\n{st.score}\n{st.combo}  (x{st.multiplier})\n"
            f"{mm:02d}:{ss:02d}\n{st.sorted_ok} / {st.errors}\n{line}"
        )
        out.append((FONT.mjFONTSCALE_150, GRID.mjGRID_TOPLEFT, left, right))
        blocked = "  << VIRTUAL CONTACT" if tele.blocked_s > 0.3 else ""
        held = "-"
        if g.held is not None:
            p = g.parcels[g.held]
            dest = layout.BIN_BY_KEY[p.dest].name_en if p.dest else "scan it / QC"
            held = f"{p.ptype.name_en} -> {dest}" + ("  [magnet]" if g.welded == g.held else "")
        tl = "TWIN LINK\nRATE\nAGE\nSYNC ERR\nHOLDING\nASSIST / HINTS"
        tr = (
            f"{tele.source}\n{tele.rate_hz:4.0f} Hz\n{tele.age_ms:4.0f} ms\n{tele.sync_err_deg:4.1f} deg{blocked}\n{held}\n"
            f"{'ON' if g.assist else 'OFF'} / {'ON' if g.hints else 'OFF'}"
        )
        out.append((FONT.mjFONTSCALE_150, GRID.mjGRID_TOPRIGHT, tl, tr))
        log = "\n".join(e.text for e in list(g.event_log)[-4:]) or " "
        out.append((FONT.mjFONTSCALE_150, GRID.mjGRID_BOTTOMLEFT, "EVENTS", log))
        keys = (
            "ENTER start  INS assist  DEL hints  END reset\n"
            "PgUp call AGV  PgDn call tug  (or press the buttons with the arm)\n[ ] cameras   double-pinch = start"
        )
        out.append((FONT.mjFONTSCALE_100, GRID.mjGRID_BOTTOMRIGHT, None, keys))
        if g.phase == "results" and g.results:
            r = g.results
            board = "\n".join(
                f"{i + 1:>2}. {e['player'][:10]:<10} {e['score']:>6}  {e['rank']}"
                for i, e in enumerate(g.leaderboard[:5])
            )
            summary = (
                f"RANK {r['rank']}  -  {r['title_en']}\nscore {r['score']}   accuracy {r['accuracy']}%   max combo {r['max_combo']}\n"
                f"VIP saved {r['vip']}   cracked {r['cracked']}   "
                + f"missed {r['missed']}   spilled {r['spilled']}   clutch saves {r['clutch']}"
                + f"\n\nLEADERBOARD\n{board or '-'}"
            )
            out.append((FONT.mjFONTSCALE_200, GRID.mjGRID_TOP, None, summary))
        elif self.banner and now < self.banner[1]:
            out.append((FONT.mjFONTSCALE_250, GRID.mjGRID_TOP, None, self.banner[0].text))
        return out
