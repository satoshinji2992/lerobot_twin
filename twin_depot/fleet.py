"""Warehouse vehicles: an inbound AGV that feeds the belt, a tug that hauls the recycling cage and a
forklift on patrol. Routes are planned with A* on an occupancy grid baked from the scene geometry,
so the same missions work in the built-in hall and in the Isaac Sim warehouse.

Missions are generators: each ``yield`` is one simulation frame. Vehicles give way to vehicles of
higher priority (forklift > tug > AGV) that are close in front of them.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import mujoco
import numpy as np

from . import layout

CELL = 0.2  # m
BAND = (0.03, 2.0)  # obstacle heights above the floor that block a vehicle


# ------------------------------------------------------------------------------------------
# occupancy grid + A*
# ------------------------------------------------------------------------------------------
class Grid:
    def __init__(self, occ: np.ndarray, origin: tuple[float, float], inflate: float):
        self.raw = occ
        self.origin = np.asarray(origin, dtype=float)
        self.occ = _dilate(occ, int(math.ceil(inflate / CELL)))

    @classmethod
    def from_model(cls, m: mujoco.MjModel, d: mujoco.MjData, skip_bodies: set[int], inflate: float = 0.55):
        """Rasterise every static geom that pokes into the vehicle height band."""
        z0, z1 = layout.FLOOR_Z + BAND[0], layout.FLOOR_Z + BAND[1]
        pts = []
        for g in range(m.ngeom):
            if m.geom_bodyid[g] in skip_bodies or m.geom_group[g] == 5:
                continue
            gt = m.geom_type[g]
            pos, rot = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3)
            if gt == mujoco.mjtGeom.mjGEOM_MESH:
                mid = m.geom_dataid[g]
                v = m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
                f = m.mesh_face[m.mesh_faceadr[mid] : m.mesh_faceadr[mid] + m.mesh_facenum[mid]]
                pts.append(_sample_triangles(v @ rot.T + pos, f, z0, z1))
            elif gt in (
                mujoco.mjtGeom.mjGEOM_BOX,
                mujoco.mjtGeom.mjGEOM_CYLINDER,
                mujoco.mjtGeom.mjGEOM_SPHERE,
            ):
                s = m.geom_size[g]
                half = (
                    s
                    if gt == mujoco.mjtGeom.mjGEOM_BOX
                    else np.array([s[0], s[0], s[1] if gt == 5 else s[0]])
                )
                ext = np.abs(rot) @ half
                lo, hi = pos - ext, pos + ext
                if hi[2] < z0 or lo[2] > z1:
                    continue
                xs = np.arange(lo[0], hi[0] + CELL / 2, CELL / 2)
                ys = np.arange(lo[1], hi[1] + CELL / 2, CELL / 2)
                gx, gy = np.meshgrid(xs, ys)
                pts.append(np.stack([gx.ravel(), gy.ravel()], axis=1))
        pts = np.concatenate([p for p in pts if len(p)]) if pts else np.zeros((0, 2))
        lo = np.maximum(pts.min(axis=0) - 2.0, -60.0) if len(pts) else np.array([-15.0, -15.0])
        hi = np.minimum(pts.max(axis=0) + 2.0, 60.0) if len(pts) else np.array([15.0, 15.0])
        shape = tuple(np.ceil((hi - lo) / CELL).astype(int) + 1)
        occ = np.zeros(shape, dtype=bool)
        ij = np.floor((pts - lo) / CELL).astype(int)
        ok = (ij >= 0).all(axis=1) & (ij[:, 0] < shape[0]) & (ij[:, 1] < shape[1])
        occ[ij[ok, 0], ij[ok, 1]] = True
        return cls(occ, tuple(lo), inflate)

    def cell(self, p) -> tuple[int, int]:
        ij = np.floor((np.asarray(p[:2]) - self.origin) / CELL).astype(int)
        return int(np.clip(ij[0], 0, self.occ.shape[0] - 1)), int(np.clip(ij[1], 0, self.occ.shape[1] - 1))

    def center(self, ij) -> np.ndarray:
        return self.origin + (np.asarray(ij) + 0.5) * CELL

    def free(self, ij) -> bool:
        return not self.occ[ij]

    def nearest_free(self, ij, max_r: int = 15) -> tuple[int, int]:
        if self.free(ij):
            return ij
        for r in range(1, max_r):
            ring = [
                (ij[0] + dx, ij[1] + dy)
                for dx in range(-r, r + 1)
                for dy in range(-r, r + 1)
                if max(abs(dx), abs(dy)) == r
            ]
            ring = [
                c
                for c in ring
                if 0 <= c[0] < self.occ.shape[0] and 0 <= c[1] < self.occ.shape[1] and self.free(c)
            ]
            if ring:
                return min(ring, key=lambda c: (c[0] - ij[0]) ** 2 + (c[1] - ij[1]) ** 2)
        return ij

    def line_free(self, a, b) -> bool:
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = int(np.linalg.norm(b - a) / (CELL * 0.4)) + 1
        return all(self.free(self.cell(a + (b - a) * s)) for s in np.linspace(0.0, 1.0, n + 1))

    def plan(self, start, goal) -> list[np.ndarray]:
        """Smoothed A* path (world xy points, start excluded, goal included)."""
        s, g = self.nearest_free(self.cell(start)), self.nearest_free(self.cell(goal))
        nbrs = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0)] + [
            (dx, dy, 1.4142) for dx in (-1, 1) for dy in (-1, 1)
        ]
        openq = [(0.0, s)]
        came: dict = {s: None}
        cost = {s: 0.0}
        while openq:
            _, cur = heapq.heappop(openq)
            if cur == g:
                break
            for dx, dy, w in nbrs:
                nx, ny = cur[0] + dx, cur[1] + dy
                if not (0 <= nx < self.occ.shape[0] and 0 <= ny < self.occ.shape[1]) or self.occ[nx, ny]:
                    continue
                nc = cost[cur] + w
                if nc < cost.get((nx, ny), 1e18):
                    cost[(nx, ny)] = nc
                    came[(nx, ny)] = cur
                    heapq.heappush(openq, (nc + math.hypot(g[0] - nx, g[1] - ny), (nx, ny)))
        if g not in came:
            return [np.asarray(goal[:2], float)]  # no route: drive straight (and hope)
        cells = []
        c = g
        while c is not None:
            cells.append(c)
            c = came[c]
        pts = [np.asarray(start[:2], float)] + [self.center(c) for c in reversed(cells[:-1])]
        pts[-1] = np.asarray(goal[:2], float) if self.free(self.cell(goal)) else pts[-1]
        if not self.free(self.cell(goal)):
            pts.append(np.asarray(goal[:2], float))
        # string pulling
        out, i = [], 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not self.line_free(pts[i], pts[j]):
                j -= 1
            out.append(pts[j])
            i = j
        return out


def _sample_triangles(v: np.ndarray, f: np.ndarray, z0: float, z1: float) -> np.ndarray:
    """XY sample points covering every triangle that reaches into [z0, z1]."""
    tri = v[f]  # (n, 3, 3)
    zmin, zmax = tri[:, :, 2].min(axis=1), tri[:, :, 2].max(axis=1)
    tri = tri[(zmax >= z0) & (zmin <= z1)]
    if len(tri) == 0:
        return np.zeros((0, 2))
    edge = np.max(np.linalg.norm(tri - np.roll(tri, 1, axis=1), axis=2), axis=1)
    steps = np.clip(np.ceil(edge / (CELL * 0.5)).astype(int), 1, 60)
    out = [tri[:, :, :2].reshape(-1, 2)]
    for n in np.unique(steps):
        sel = tri[steps == n]
        if n <= 1:
            out.append(sel[:, :, :2].mean(axis=1))
            continue
        a, b = np.meshgrid(np.arange(n + 1), np.arange(n + 1))
        keep = a + b <= n
        w = np.stack([a[keep], b[keep]], axis=1) / n  # barycentric (u, v)
        p = sel[:, None, 0, :2] + w[None, :, :1] * (sel[:, None, 1, :2] - sel[:, None, 0, :2])
        p = p + w[None, :, 1:] * (sel[:, None, 2, :2] - sel[:, None, 0, :2])
        out.append(p.reshape(-1, 2))
    return np.concatenate(out)


def _dilate(occ: np.ndarray, r: int) -> np.ndarray:
    out = occ.copy()
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            if dx * dx + dy * dy > r * r or (dx == 0 and dy == 0):
                continue
            sx = slice(max(dx, 0), occ.shape[0] + min(dx, 0))
            tx = slice(max(-dx, 0), occ.shape[0] + min(-dx, 0))
            sy = slice(max(dy, 0), occ.shape[1] + min(dy, 0))
            ty = slice(max(-dy, 0), occ.shape[1] + min(-dy, 0))
            out[tx, ty] |= occ[sx, sy]
    return out


# ------------------------------------------------------------------------------------------
# vehicles
# ------------------------------------------------------------------------------------------
@dataclass
class Vehicle:
    name: str
    mocap: int
    pos: np.ndarray
    yaw: float
    speed: float  # m/s
    priority: int  # higher = others give way to it
    turn_rate: float = 1.6  # rad/s
    half_len: float = 0.55
    label: str = ""
    moving: bool = False
    yielding_to: str | None = None
    vel: float = 0.0
    trail: list[np.ndarray] = field(default_factory=list)

    def heading(self) -> np.ndarray:
        return np.array([math.cos(self.yaw), math.sin(self.yaw)])


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class Fleet:
    """Owns the vehicles and runs their missions. The game provides the hooks (see ``Game``)."""

    def __init__(self, m: mujoco.MjModel, d: mujoco.MjData, idx, hall: str, game):
        self.m, self.d, self.idx, self.game = m, d, idx, game
        self.st = layout.STATIONS[hall]
        skip = {int(b) for b in idx.mobile_bodies}
        self.grid = Grid.from_model(m, d, skip)
        self.vehicles: dict[str, Vehicle] = {}
        spec = {
            "agv": (layout.AGV_SPEED, 1),
            "tug": (layout.TUG_SPEED, 2),
            "forklift": (layout.FORKLIFT_SPEED, 3),
        }
        for name, (speed, prio) in spec.items():
            x, y, yaw = self.st[f"{name}_park"]
            v = Vehicle(
                name, int(idx.vehicle_mocap[name]), np.array([x, y], float), math.radians(yaw), speed, prio
            )
            if name == "forklift":
                v.half_len, v.turn_rate = 1.3, 1.0
            self.vehicles[name] = v
            self._place(v)
        self.missions: dict[str, Iterator] = {
            "agv": self._supply_mission(self.vehicles["agv"]),
            "tug": self._recycle_mission(self.vehicles["tug"]),
            "forklift": self._patrol_mission(self.vehicles["forklift"]),
        }
        self.dt = 1 / 60
        self.cage_attached = False
        # call buttons: what each vehicle is doing, and whether it has been called
        self.called = {"agv": False, "tug": False}
        self.state = {"agv": "to_racks", "tug": "parked"}

    def call(self, key: str) -> tuple[bool, str, str]:
        """A CALL button was pressed. Returns (accepted, english, chinese) for the event log."""
        g, st = self.game, self.state[key]
        if key == "agv":
            if st in ("delivering", "unloading"):
                return False, "AGV ALREADY ON ITS WAY", "供件车已经在路上了"
            self.called["agv"] = True
            if st == "ready":
                return True, "AGV DISPATCHED", "供件车出发！"
            return True, "AGV CALLED - leaves once loaded", "已呼叫供件车：装完货马上出发"
        if st != "parked":
            return False, "TUG IS BUSY", "牵引车正在忙"
        if g.cage_count == 0:
            return False, "CAGE IS EMPTY - nothing to haul", "回收笼是空的，不用拉"
        self.called["tug"] = True
        return True, "TUG CALLED", "已呼叫牵引车：提前清空回收笼"

    # -- per frame ------------------------------------------------------------------------------
    def step(self, dt: float) -> None:
        self.dt = dt
        for name, mission in self.missions.items():
            try:
                next(mission)
            except StopIteration:
                self.missions[name] = iter(())
        for v in self.vehicles.values():
            self._place(v)
        if self.cage_attached:
            t = self.vehicles["tug"]
            back = t.pos - t.heading() * (t.half_len + layout.CAGE.half[1] + 0.12)
            self.game.move_cage(back, t.yaw)

    def _place(self, v: Vehicle) -> None:
        self.d.mocap_pos[v.mocap] = (v.pos[0], v.pos[1], layout.FLOOR_Z)
        self.d.mocap_quat[v.mocap] = (math.cos(v.yaw / 2), 0, 0, math.sin(v.yaw / 2))

    # -- motion primitives (generators) ---------------------------------------------------------
    def _wait(self, seconds: float) -> Iterator:
        t_end = self.game.t + seconds
        while self.game.t < t_end:
            yield

    def _blocked(self, v: Vehicle, reverse: bool) -> str | None:
        ahead = -v.heading() if reverse else v.heading()
        for o in self.vehicles.values():
            if o is v:
                continue
            rel = o.pos - v.pos
            dist = float(np.linalg.norm(rel))
            if dist < 1e-6:
                continue
            front = float(rel @ ahead) / dist
            reach = v.half_len + o.half_len + 1.0
            # strict right of way (forklift > tug > AGV): nobody waits for a lower-ranked vehicle,
            # so two vehicles can never wait for each other
            if front > 0.55 and dist < reach and o.priority > v.priority:
                return o.name
        return None

    def _turn_to(self, v: Vehicle, yaw: float) -> Iterator:
        while abs(_wrap(yaw - v.yaw)) > 1e-3:
            step = v.turn_rate * self.dt
            dy = _wrap(yaw - v.yaw)
            v.yaw = _wrap(v.yaw + max(-step, min(step, dy)))
            yield

    def _straight(self, v: Vehicle, goal, reverse: bool = False) -> Iterator:
        goal = np.asarray(goal, float)
        v.moving, v.vel = True, 0.0
        while True:
            rel = goal - v.pos
            dist = float(np.linalg.norm(rel))
            if dist < 0.01:
                break
            other = self._blocked(v, reverse)
            if other is not None:
                if v.yielding_to != other:
                    v.yielding_to = other
                    self.game.vehicle_yields(v.name, other)
                v.vel = max(0.0, v.vel - 2.0 * self.dt)
            else:
                v.yielding_to = None
                v.vel = min(
                    v.speed * (0.4 if reverse else 1.0), v.vel + 0.8 * self.dt, math.sqrt(1.6 * dist) + 0.05
                )
            v.pos = v.pos + rel / dist * min(dist, v.vel * self.dt)
            yield
        v.moving, v.vel = False, 0.0

    def _drive(self, v: Vehicle, goal_xy, goal_yaw_deg: float | None = None) -> Iterator:
        """Plan around obstacles, turn on the spot towards each waypoint, then face ``goal_yaw``."""
        for wp in self.grid.plan(v.pos, goal_xy):
            rel = wp - v.pos
            if np.linalg.norm(rel) < 0.02:
                continue
            yield from self._turn_to(v, math.atan2(rel[1], rel[0]))
            yield from self._straight(v, wp)
        if goal_yaw_deg is not None:
            yield from self._turn_to(v, math.radians(goal_yaw_deg))

    # -- missions -------------------------------------------------------------------------------
    def _supply_mission(self, v: Vehicle) -> Iterator:
        """Rack aisle -> load a batch -> infeed dock at the head of the belt -> unload one by one."""
        g = self.game
        dock_x, dock_y, dock_yaw = layout.infeed_dock()
        pre = (dock_x, dock_y + 1.4)
        while True:
            sx, sy, syaw = self.st["supply"]
            self.state["agv"], v.label = "to_racks", "AGV -> racks"
            yield from self._drive(v, (sx, sy), syaw)
            self.state["agv"], v.label = "loading", "AGV loading"
            yield from self._wait(2.0)
            batch = g.next_batch(layout.AGV_BATCH)
            g.show_agv_cargo(batch)
            # loaded: wait for the CALL AGV button - or go anyway once the infeed has run dry
            self.state["agv"], v.label = "ready", f"AGV READY ({len(batch)}) - press CALL AGV"
            while not (self.called["agv"] or (g.belt_running and not g.queue())):
                yield
            if not self.called["agv"]:
                g.vehicle_event("agv_auto", len(batch))
            self.called["agv"] = False
            self.state["agv"], v.label = "delivering", f"AGV  {len(batch)} parcels"
            yield from self._drive(v, pre, dock_yaw)
            yield from self._straight(v, (dock_x, dock_y))
            g.vehicle_event("agv_docked", len(batch))
            self.state["agv"] = "unloading"
            while batch:
                v.label = f"AGV unloading {len(batch)}"
                if g.infeed_has_room() and g.unload(batch[0]):
                    batch.pop(0)
                    g.show_agv_cargo(batch)
                    yield from self._wait(layout.AGV_UNLOAD_S)
                else:
                    yield
            self.state["agv"] = "returning"
            yield from self._straight(v, pre, reverse=True)

    def _recycle_mission(self, v: Vehicle) -> Iterator:
        """Waits for a full recycling cage, tows it to the returns area and brings it back empty."""
        g = self.game
        cx, cy = layout.CAGE.center
        dock = np.array([cx, cy - layout.CAGE.half[1] - 0.12 - v.half_len])
        pre = dock + (0.0, -1.4)
        while True:
            px, py, pyaw = self.st["tug_park"]
            v.label = "TUG"
            yield from self._drive(v, (px, py), pyaw)
            self.state["tug"] = "parked"
            while not (g.cage_needs_haul() or (self.called["tug"] and g.cage_count > 0)):
                yield
            self.state["tug"], v.label = "busy", "TUG -> cage"
            g.vehicle_event("tug_called" if self.called["tug"] else "tug_dispatched", g.cage_count)
            self.called["tug"] = False
            yield from self._drive(v, pre, -90.0)  # tow hitch faces the cage
            yield from self._straight(v, dock, reverse=True)
            g.cage_detach()
            self.cage_attached = True
            rx, ry, ryaw = self.st["returns"]
            v.label = "TUG towing cage"
            yield from self._straight(v, pre)
            yield from self._drive(v, (rx, ry), ryaw)
            v.label = "TUG emptying"
            yield from self._wait(2.5)
            g.cage_emptied()
            v.label = "TUG returning cage"
            yield from self._drive(v, pre, -90.0)
            yield from self._straight(v, dock, reverse=True)
            self.cage_attached = False
            g.cage_attach()
            yield from self._straight(v, pre)

    def _patrol_mission(self, v: Vehicle) -> Iterator:
        route = self.st["forklift_route"]
        k = 0
        while True:
            x, y, yaw = route[k % len(route)]
            v.label = "FORKLIFT"
            yield from self._drive(v, (x, y), yaw)
            yield from self._wait(layout.FORKLIFT_DWELL_S)
            k += 1


Hook = Callable[..., None]
