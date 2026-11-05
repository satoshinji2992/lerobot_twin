"""IK-driven demo operator. Drives only the *twin* (never the real arm): used for the no-hardware
demo, the arcade-style attract mode and end-to-end tests."""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from . import layout
from .scene import SceneIndex

DOWN = np.array([0.0, 0.0, -1.0])
OPEN, CLOSED, RELEASE = 1.05, -0.12, 0.72  # RELEASE > game.RELEASE_Q so the magnet lets go


class ArmIK:
    """Damped-least-squares IK for the grasp centre with a 'jaws pointing down' constraint."""

    def __init__(self, idx: SceneIndex):
        self.idx = idx
        self.m = idx.model
        self.d = mujoco.MjData(self.m)
        self.qadr = idx.arm_qpos[:5]
        self.dof = idx.arm_dof[:5]
        self.lo, self.hi = idx.joint_ranges[:5, 0], idx.joint_ranges[:5, 1]
        self.jacp = np.zeros((3, self.m.nv))
        self.jacr = np.zeros((3, self.m.nv))
        self.last_approach = DOWN

    def fk(self, q5: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.d.qpos[self.qadr] = q5
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        pos = self.d.site_xpos[self.idx.grasp_site].copy()
        rot = self.d.site_xmat[self.idx.tcp_site].reshape(3, 3).copy()
        return pos, rot

    def seed(self, target, q0) -> np.ndarray:
        q = np.array(q0, dtype=float)[:5].copy()
        q[0] = np.clip(
            -math.atan2(target[1] - layout.PAN_AXIS_XY[1], target[0] - layout.PAN_AXIS_XY[0]),
            self.lo[0],
            self.hi[0],
        )
        return q

    def solve(
        self, target, q0, roll: float | None = None, approach=DOWN, strict: bool = True, iters: int = 150
    ):
        """``strict``: jaws must point along ``approach``; otherwise orientation is only a soft wish."""
        w = 0.25 if strict else 0.003
        target = np.asarray(target, dtype=float)
        best = None
        for q_init in (np.array(q0, dtype=float)[:5], self.seed(target, [0, 0.2, 0.2, 1.2, q0[4]])):
            q = q_init.copy()
            if roll is not None:
                q[4] = roll
            free = [0, 1, 2, 3] if roll is not None else [0, 1, 2, 3, 4]
            for _ in range(iters):
                pos, rot = self.fk(q)
                err_p = target - pos
                err_a = np.cross(rot[:, 0], approach)
                if np.linalg.norm(err_p) < 6e-4 and (not strict or np.linalg.norm(err_a) < 0.01):
                    break
                mujoco.mj_jacSite(self.m, self.d, self.jacp, self.jacr, self.idx.grasp_site)
                dofs = self.dof[free]
                jac = np.vstack([self.jacp[:, dofs], w * self.jacr[:, dofs]])
                e = np.concatenate([err_p, w * err_a])
                dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), e)
                q[free] += np.clip(dq, -0.25, 0.25)
                q = np.clip(q, self.lo, self.hi)
            pos, rot = self.fk(q)
            ep = np.linalg.norm(target - pos)
            ea = np.linalg.norm(np.cross(rot[:, 0], approach))
            ok = ep < 3e-3 and (not strict or ea < 0.08)
            if ok:
                return q, True
            if best is None or ep + 0.05 * ea < best[0]:
                best = (ep + 0.05 * ea, q)
        return best[1], False

    def solve_grasp(self, target, q0, parcel_yaw: float, max_tilt_deg: float = 0.0):
        """Grasp with the jaws closing on two opposite parcel faces.

        Top-down first. Targets beyond the top-down workspace (a moving belt runs out there) lean
        the approach outwards by up to ``max_tilt_deg``. The SO-101 has 5 DOF, so a tilted
        approach always lies in the vertical plane through the pan axis, and the jaws stay level
        only when they close perpendicular to that plane - the flow belt presents its parcels
        at ~45 deg so that one pair of faces lines up with it.
        ``self.last_approach`` holds the approach direction of the returned solution.
        """
        target = np.asarray(target, dtype=float)
        lean = math.atan2(target[1] - layout.PAN_AXIS_XY[1], target[0] - layout.PAN_AXIS_XY[0])
        lean_dir = np.array([math.cos(lean), math.sin(lean), 0.0])
        q, ok = np.asarray(q0, dtype=float)[:5], False
        for tilt in np.radians(np.arange(0.0, max_tilt_deg + 1e-6, 8.0)):
            approach = DOWN * math.cos(tilt) + lean_dir * math.sin(tilt)
            if tilt == 0.0:
                q, ok = self._solve_grasp(target, q0, parcel_yaw, approach, period=math.pi / 2)
            else:
                q, ok = self._solve_grasp(target, q0, lean + math.pi / 2, approach, period=math.pi)
            self.last_approach = approach
            if ok:
                break
        return q, ok

    def solve_receive(self, target, q0):
        """Top-down, jaws closing across the belt towards +x (fixed jaw on the robot's side of the
        parcel's path): the moving-belt receive grasp."""
        self.last_approach = DOWN
        return self._solve_grasp(np.asarray(target, dtype=float), q0, 0.0, DOWN, period=2 * math.pi)

    def solve_tilted(self, target, q0, max_tilt_deg: float = 40.0):
        """Top-down if possible, else leaning outwards (radially) a little more each try."""
        target = np.asarray(target, dtype=float)
        lean = math.atan2(target[1] - layout.PAN_AXIS_XY[1], target[0] - layout.PAN_AXIS_XY[0])
        lean_dir = np.array([math.cos(lean), math.sin(lean), 0.0])
        q, ok = np.asarray(q0, dtype=float)[:5], False
        for tilt in np.radians(np.arange(0.0, max_tilt_deg + 1e-6, 8.0)):
            q, ok = self.solve(
                target, q0, roll=q0[4], approach=DOWN * math.cos(tilt) + lean_dir * math.sin(tilt)
            )
            if ok:
                break
        return q, ok

    def _solve_grasp(self, target, q0, close_yaw: float, approach, period: float):
        """Pick the wrist roll whose closing direction matches ``close_yaw`` (mod ``period``)."""
        q, _ = self.solve(target, q0, approach=approach)
        _, rot = self.fk(q)
        close_dir = rot[:, 2]
        cur = math.atan2(close_dir[1], close_dir[0])
        best = None
        for k in range(-4, 5):
            want = close_yaw + k * math.pi / 2
            delta = (want - cur + math.pi) % (2 * math.pi) - math.pi
            for sgn in (1, -1):
                roll = q[4] + sgn * delta
                if not (self.lo[4] + 0.05 < roll < self.hi[4] - 0.05):
                    continue
                _, r2 = self.fk(np.r_[q[:4], roll])
                c2 = r2[:, 2]
                err = abs(math.sin(math.pi / period * (math.atan2(c2[1], c2[0]) - close_yaw)))
                score = err + 0.05 * abs(roll)
                if best is None or score < best[0]:
                    best = (score, roll)
        roll = best[1] if best else q[4]
        return self.solve(target, np.r_[q[:4], roll], roll=roll, approach=approach)


def _yaw_of(quat) -> float:
    w, x, y, z = quat
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


@dataclass
class _Segment:
    q_from: np.ndarray
    q_to: np.ndarray
    t0: float
    dur: float

    def at(self, t: float) -> np.ndarray:
        s = 1.0 if self.dur <= 0 else min(1.0, max(0.0, (t - self.t0) / self.dur))
        s = s * s * (3 - 2 * s)
        return self.q_from + s * (self.q_to - self.q_from)

    def done(self, t: float) -> bool:
        return t >= self.t0 + self.dur


DESCEND_S, CLOSE_S, SET_S = 0.45, 0.55, 0.3
# Flow belt "receive" grasp: the open jaws are set down straddling the belt ahead of the parcel,
# closing across the belt; the parcel slides in between the fixed jaw and the raised moving jaw.
INTERCEPT_Y = layout.CONVEYOR.reach_hi - 0.01  # furthest-upstream receive point
RECEIVE_X = layout.CONVEYOR.x - 0.004  # fixed jaw rides ~5 mm beside the parcel's path
RECEIVE_Z = 0.049  # grasp centre height: fixed pad ~2 mm above the belt
RECEIVE_GAP = 0.04  # set the jaws at least this far ahead of the parcel centre
BUTTON_PUSH_Z = 0.024  # grasp-centre height over the housing that pushes the cap ~6 mm down
PLACE_OFFSET_X = 0.012  # drop parcels nearer the bin's robot-side wall so the opening jaw clears
MAX_TILT_DEG = 40.0  # grasps further out than the top-down workspace lean the jaws outwards


class Autopilot:
    """A tidy, slightly showy operator. The game asks it for joint targets every frame."""

    SPEED = 1.6  # rad/s on the largest joint move (the twin's servos top out around 2 rad/s)

    def __init__(self, idx: SceneIndex, miss_rate: float = 0.0, seed: int | None = None):
        self.idx = idx
        self.miss_rate = miss_rate  # demo: now and then let a parcel ride by (so the cage/tug get work)
        self.rng = np.random.default_rng(seed)
        self.considered: set[int] = set()
        self.ik = ArmIK(idx)
        q, _ = self.ik.solve((0.19, 0.0, 0.10), np.array([0.0, -0.3, 0.3, 1.2, 0.0]), strict=False)
        self.home = np.r_[q, OPEN * 0.4]
        self.state = "idle"
        self.seg: _Segment | None = None
        self.q = self.home.copy()
        self.target: int | None = None
        self.wait_until = 0.0
        self.grasp_q: np.ndarray | None = None
        self.receive = False  # current pick is a flow-belt receive grasp
        self.above_q: np.ndarray | None = None  # pose above the destination bin
        self.button: str | None = None
        self.button_q: np.ndarray | None = None
        self.intercept_y = 0.0
        self.wait_deadline = 0.0
        # moving-target tracking (descend + close follow the parcel frame by frame)
        self.track_until = 0.0
        self.track_t = 0.0
        self.track_len = 1.0
        self.tracking: str | None = None  # which state the current tracking window belongs to

    # -- helpers ------------------------------------------------------------------------------
    def _go(self, q_to: np.ndarray, t: float, min_dur: float = 0.15) -> None:
        dist = float(np.max(np.abs(q_to[:5] - self.q[:5])))
        self.seg = _Segment(self.q.copy(), q_to.copy(), t, max(min_dur, dist / self.SPEED))

    def _ik(
        self, pos, grip: float, yaw: float | None = None, strict: bool = True, free_roll: bool = False
    ) -> np.ndarray | None:
        if yaw is None:
            q, ok = self.ik.solve(pos, self.q[:5], roll=None if free_roll else self.q[4], strict=strict)
        else:
            q, ok = self.ik.solve_grasp(pos, self.q[:5], yaw, max_tilt_deg=MAX_TILT_DEG)
        return np.r_[q, grip] if ok else None

    def _ik_receive(self, y: float, z: float, grip: float) -> np.ndarray | None:
        q, ok = self.ik.solve_receive((RECEIVE_X, y, z), self.q[:5])
        return np.r_[q, grip] if ok else None

    def reset(self, q_now: np.ndarray | None = None) -> None:
        """Abandon the current pick (e.g. the field was cleared) and continue from ``q_now``."""
        self.state, self.seg, self.target, self.wait_until = "idle", None, None, 0.0
        self.track_until, self.tracking, self.receive = 0.0, None, False
        if q_now is not None:
            self.q = np.array(q_now, dtype=float)

    @staticmethod
    def _parcel_vel(game, i: int) -> np.ndarray:
        """Flow-belt parcels move at belt speed towards -y; anything else is (nearly) still."""
        p = game.parcels[i]
        return np.array([0.0, -game.belt_speed, 0.0]) if p.state == "belt" and p.released else np.zeros(3)

    def _track(self, t: float, d: mujoco.MjData, game, grip_from: float, grip_to: float) -> bool:
        """Follow the target parcel until ``track_until``. Returns False once the window is over."""
        dt, self.track_t = max(0.0, t - self.track_t), t
        rem = self.track_until - t
        if rem <= 0:
            return False
        i = self.target
        pos = d.xpos[self.idx.parcel_body[i]] + self._parcel_vel(game, i) * min(rem, 0.12)
        if self.receive:  # ride along the belt with the parcel, jaws still across it
            q = self._ik_receive(pos[1], RECEIVE_Z, self.q[5])
        else:
            q = self._ik(pos + [0, 0, 0.004], self.q[5], _yaw_of(d.xquat[self.idx.parcel_body[i]]))
        if q is not None:
            self.q[:5] += (q[:5] - self.q[:5]) * min(1.0, dt / max(rem, dt, 1e-3) * 2.0)
        frac = 1.0 - rem / self.track_len
        self.q[5] = grip_from + (grip_to - grip_from) * min(1.0, max(0.0, frac))
        return True

    def _start_track(self, t: float, dur: float, state: str) -> None:
        self.track_until, self.track_t, self.track_len, self.tracking = t + dur, t, dur, state

    def _press_button(self, t: float, key: str) -> bool:
        b = layout.BUTTON_BY_KEY[key]
        q = self._ik((*b.center, b.base_h + 0.06), CLOSED, strict=True, free_roll=True)
        if q is None:
            return False
        self.button, self.button_q = key, q
        self._go(q, t)
        self.state = "push"
        return True

    def _plan_receive(self, t: float, p: np.ndarray, speed: float) -> bool:
        """Pick a receive point the jaws can reach before the parcel does; start moving there."""
        c = layout.CONVEYOR
        dur, q, y_set = 0.6, None, INTERCEPT_Y
        y_min = c.reach_lo + speed * CLOSE_S + 0.005  # still closed before it leaves our reach
        for _ in range(3):
            y_ready = p[1] - speed * (dur + SET_S)  # parcel position once the jaws are down
            y_set = min(INTERCEPT_Y, y_ready - RECEIVE_GAP)
            if y_set < y_min:
                # late: drop the jaws right onto it - the fixed jaw passes beside the parcel and the
                # open moving jaw is above it, so this works on a moving parcel too
                y_set = min(INTERCEPT_Y, y_ready)
            if y_set < y_min:
                return False  # it would reach the slot before we could close
            q = self._ik_receive(y_set, RECEIVE_Z + 0.03, OPEN)
            if q is None or self._ik_receive(y_set, RECEIVE_Z, OPEN) is None:
                return False
            dur = max(0.15, float(np.max(np.abs(q[:5] - self.q[:5]))) / self.SPEED)
        self.intercept_y = y_set
        self.wait_deadline = t + dur + SET_S + (p[1] - y_set) / max(speed, 1e-3) + 2.5
        self._go(q, t)
        self.state = "set"
        return True

    # -- main ---------------------------------------------------------------------------------
    def step(self, t: float, d: mujoco.MjData, game) -> np.ndarray:
        if self.seg is not None:
            self.q = self.seg.at(t)
            if not self.seg.done(t):
                return self.q
            self.seg = None
        if t < self.wait_until:
            return self.q
        idx = self.idx
        st = self.state

        def ppos(i):
            return d.xpos[idx.parcel_body[i]].copy()

        if st == "idle":
            key = game.autopilot_button()  # a CALL button first, if no parcel needs us for a while
            if key is not None and self._press_button(t, key):
                self.target = None
                return self.q
            self.target = game.autopilot_pick(lead_s=1.6)
            if self.target is not None and self.target not in self.considered:
                self.considered.add(self.target)
                if len(self.considered) > 64:
                    self.considered = {self.target}
                if self.rng.random() < self.miss_rate and game.parcels[self.target].state == "belt":
                    game.autopilot_skip(self.target, 12.0)  # "oops, too slow"
                    self.target = None
            if self.target is None:
                if float(np.max(np.abs(self.q - self.home))) > 0.02:
                    self._go(self.home, t)
                return self.q
            v = self._parcel_vel(game, self.target)
            self.receive = bool(v[1] < 0)
            if self.receive:
                if not self._plan_receive(t, ppos(self.target), -float(v[1])):
                    self.receive = False
                    game.autopilot_skip(self.target)
                return self.q
            yaw = _yaw_of(d.xquat[idx.parcel_body[self.target]])
            grasp = ppos(self.target) + [0, 0, 0.004]
            q = None
            if self._ik(grasp, OPEN, yaw) is not None:
                # hover back along the approach axis, so the descent slides straight onto the parcel
                q = self._ik(grasp - self.ik.last_approach * 0.045, OPEN, yaw)
            if q is None:
                game.autopilot_skip(self.target)
                return self.q
            self.grasp_q = q
            self._go(q, t)
            self.state = "descend"
        elif st == "push":  # above the button: press it down with the closed jaws
            b = layout.BUTTON_BY_KEY[self.button]
            q = self._ik((*b.center, b.base_h + BUTTON_PUSH_Z), CLOSED, strict=True, free_roll=True)
            if q is not None:
                self._go(q, t, 0.35)
            self.state = "unpush"
        elif st == "unpush":
            self.wait_until = t + 0.15
            self._go(self.button_q, t, 0.3)
            self.state = "idle"
        elif st == "set":  # lower the open jaws onto the belt at the receive point
            q = self._ik_receive(self.intercept_y, RECEIVE_Z, OPEN)
            if q is None:
                self.state = "idle"
                return self.q
            self._go(q, t, SET_S)
            self.state = "wait"
        elif st == "wait":  # jaws straddle the belt; close once the parcel is between them
            i = self.target
            p = ppos(i)
            if game.parcels[i].state != "belt" or t > self.wait_deadline:
                self._go(np.r_[self.q[:2], self.q[2] - 0.1, self.q[3:5], OPEN], t, 0.25)  # back off
                self.state, self.receive = "idle", False
                return self.q
            if p[1] <= self.intercept_y + 0.003:
                self._start_track(t, CLOSE_S, "close")
                self.state = "close"
            return self.q
        elif st == "descend":
            if self.tracking != "descend":
                self._start_track(t, DESCEND_S, "descend")
            if not self._track(t, d, game, OPEN, OPEN):
                self._start_track(t, CLOSE_S, "close")
                self.state = "close"
            return self.q
        elif st == "close":
            if not self._track(t, d, game, OPEN, CLOSED):
                self.tracking = None
                self.state = "lift"
            return self.q
        elif st == "lift":
            p = ppos(self.target)
            if self.receive:  # up and out over the belt, towards the bins (straight up is out of reach)
                q = self._ik((layout.CONVEYOR.x + 0.045, p[1], 0.10), CLOSED, strict=False, free_roll=True)
            else:
                q = self._ik(p + [0, 0, 0.07], CLOSED, strict=False)
                if q is None:
                    q = self._ik(p + [0, 0, 0.05], CLOSED, strict=False, free_roll=True)
            if q is None:
                self.state = "release"
                return self.q
            self._go(q, t, 0.4)
            self.state = "check"
        elif st == "check":
            if ppos(self.target)[2] < 0.06:  # missed it - try again
                self._go(np.r_[self.q[:5], OPEN], t, 0.25)
                self.state = "idle"
                return self.q
            if game.needs_scan(self.target):
                sx, sy = layout.SCANNER.center
                q = self._ik((sx, sy, 0.07), CLOSED, strict=False)
                if q is not None:
                    self._go(q, t)
                    self.state = "scan"
                    return self.q
            self.state = "carry"
        elif st == "scan":
            if game.needs_scan(self.target):
                self.wait_until = t + 0.1
                return self.q
            self.state = "carry"
        elif st == "carry":
            dest = layout.BIN_BY_KEY[game.autopilot_dest(self.target)]
            cx, cy = dest.center
            cx -= PLACE_OFFSET_X  # moving jaw (+x side) needs room to open
            q = self._ik((cx, cy, dest.height + 0.045), CLOSED, strict=False)
            if q is None:
                self.state = "release"
                return self.q
            self.above_q = q  # retreat through the same pose: straight up and out of the bin
            self._go(q, t)
            self.state = "lower"
        elif st == "lower":
            dest = layout.BIN_BY_KEY[game.autopilot_dest(self.target)]
            cx, cy = dest.center
            cx -= PLACE_OFFSET_X  # moving jaw (+x side) needs room to open
            half_h = layout.PARCEL_TYPE_BY_KEY[idx.parcel_types[self.target]].size[2] / 2
            z = 0.004 + half_h + 0.012  # a ~1 cm drop: gentle enough for fragile parcels
            q = self._ik((cx, cy, z), CLOSED)
            if q is None:  # the bins at the ends of the row need the wrist to lean out a bit
                qt, ok = self.ik.solve_tilted((cx, cy, z), self.q[:5])
                q = np.r_[qt, CLOSED] if ok else None
            if q is not None:
                self._go(q, t, 0.5)
            self.state = "release"
        elif st == "release":
            # open just enough to let go - a wide-open jaw wedges against the bin walls
            self._go(np.r_[self.q[:5], RELEASE], t, 0.3)
            self.state = "retreat"
        elif st == "retreat":
            if self.above_q is not None:
                self._go(np.r_[self.above_q[:5], RELEASE], t, 0.45)
            else:
                pos, _ = self.ik.fk(self.q[:5])
                q = self._ik(pos + [0, 0, 0.07], RELEASE, strict=False)
                if q is not None:
                    self._go(q, t, 0.45)
            self.above_q = None
            self.state = "idle"
        return self.q
