"""Where the twin's joint targets come from.

* ``RealTeleopSource``  - leader + follower on this machine. Runs the leader->follower teleop loop
  (like ``lerobot-teleoperate``) in a thread and feeds the *measured* follower joints to the twin.
* ``LeaderOnlySource``  - only the leader arm: the twin is a purely virtual follower.
* ``UdpSource``         - joints streamed from another machine by ``stream_teleop.py``.
* ``ReplaySource``      - plays back a session recorded with ``--record``.

All sources are one-way: nothing here ever sends twin/sim state to the real robot.
"""

from __future__ import annotations

import contextlib
import json
import logging
import socket
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import layout
from .twin import TwinMapping, bus_reading_to_twin_inputs

log = logging.getLogger(__name__)

PROTOCOL_VERSION = 1
DEFAULT_UDP_PORT = 5566


@dataclass
class JointSample:
    q: np.ndarray  # 6 twin joint targets (rad)
    stamp: float  # time.monotonic() when the reading was taken
    meta: dict = field(default_factory=dict)


class JointSource:
    name = "none"
    live = True  # a human operator is behind it

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: JointSample | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._rate = _RateMeter()
        self.error: str | None = None

    # -- lifecycle ----------------------------------------------------------------------------
    def connect(self) -> None:  # blocking, may prompt (LeRobot calibration)
        pass

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run_guarded, name=f"src-{self.name}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.disconnect()

    def disconnect(self) -> None:
        pass

    # -- data ---------------------------------------------------------------------------------
    def latest(self) -> JointSample | None:
        with self._lock:
            return self._latest

    def rate_hz(self) -> float:
        return self._rate.hz()

    def send_event(self, event: dict) -> None:
        """Game feedback for the operator (only the UDP source forwards it anywhere)."""

    def _publish(self, q: np.ndarray, **meta) -> None:
        sample = JointSample(q=q, stamp=time.monotonic(), meta=meta)
        with self._lock:
            self._latest = sample
        self._rate.tick()

    def _run_guarded(self) -> None:
        try:
            self._run()
        except Exception as e:  # surfaced in the HUD instead of killing the sim
            log.exception("joint source crashed")
            self.error = f"{type(e).__name__}: {e}"

    def _run(self) -> None:
        raise NotImplementedError


class _RateMeter:
    def __init__(self, window: int = 60) -> None:
        self.stamps: list[float] = []
        self.window = window

    def tick(self) -> None:
        self.stamps.append(time.monotonic())
        if len(self.stamps) > self.window:
            self.stamps.pop(0)

    def hz(self) -> float:
        if len(self.stamps) < 2 or time.monotonic() - self.stamps[-1] > 1.0:
            return 0.0
        return (len(self.stamps) - 1) / max(1e-6, self.stamps[-1] - self.stamps[0])


# ------------------------------------------------------------------------------------------------
def _make_arm_configs(arm: str, leader_port, leader_id, follower_port=None, follower_id=None, max_rel=None):
    if arm == "so101":
        from lerobot.robots.so_follower import SO101FollowerConfig as FCfg
        from lerobot.teleoperators.so_leader import SO101LeaderConfig as LCfg
    elif arm == "so100":
        from lerobot.robots.so_follower import SO100FollowerConfig as FCfg
        from lerobot.teleoperators.so_leader import SO100LeaderConfig as LCfg
    else:
        raise ValueError(arm)
    leader = LCfg(port=leader_port, id=leader_id)
    follower = None
    if follower_port:
        follower = FCfg(port=follower_port, id=follower_id, max_relative_target=max_rel)
    return leader, follower


class LeaderOnlySource(JointSource):
    name = "leader"

    def __init__(self, port: str, arm_id: str, mapping: TwinMapping, arm: str = "so101", fps: int = 60):
        super().__init__()
        self.port, self.arm_id, self.mapping, self.arm, self.fps = port, arm_id, mapping, arm, fps
        self.leader = None

    def connect(self) -> None:
        from lerobot.teleoperators.utils import make_teleoperator_from_config

        cfg, _ = _make_arm_configs(self.arm, self.port, self.arm_id)
        self.leader = make_teleoperator_from_config(cfg)
        self.leader.connect()

    def disconnect(self) -> None:
        if self.leader is not None and self.leader.is_connected:
            self.leader.disconnect()

    def _run(self) -> None:
        period = 1.0 / self.fps
        while not self._stop.is_set():
            t0 = time.perf_counter()
            action = self.leader.get_action()
            arm, pct = bus_reading_to_twin_inputs(self.leader, action)
            self._publish(self.mapping.to_twin(arm, pct), leader_q=None)
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))


class RealTeleopSource(JointSource):
    """Leader drives the real follower; the twin mirrors what the follower *actually* did."""

    name = "real"

    def __init__(
        self,
        leader_port: str,
        leader_id: str,
        follower_port: str,
        follower_id: str,
        mapping: TwinMapping,
        arm: str = "so101",
        fps: int = 60,
        max_relative_target: float | None = None,
        soft_start_s: float = 2.0,
    ):
        super().__init__()
        self.mapping, self.arm, self.fps, self.soft_start_s = mapping, arm, fps, soft_start_s
        self.cfgs = _make_arm_configs(
            arm, leader_port, leader_id, follower_port, follower_id, max_relative_target
        )
        self.leader = None
        self.follower = None
        self.leader_q: np.ndarray | None = None

    def connect(self) -> None:
        from lerobot.robots.utils import make_robot_from_config
        from lerobot.teleoperators.utils import make_teleoperator_from_config

        leader_cfg, follower_cfg = self.cfgs
        self.leader = make_teleoperator_from_config(leader_cfg)
        self.follower = make_robot_from_config(follower_cfg)
        self.leader.connect()
        self.follower.connect()

    def disconnect(self) -> None:
        for dev in (self.leader, self.follower):
            if dev is not None and dev.is_connected:
                dev.disconnect()

    def _run(self) -> None:
        period = 1.0 / self.fps
        start_obs = self.follower.get_observation()
        start = {k: v for k, v in start_obs.items() if k.endswith(".pos")}
        t_start = time.perf_counter()
        while not self._stop.is_set():
            t0 = time.perf_counter()
            action = self.leader.get_action()
            # Soft start: glide from where the follower is to where the leader is, instead of jumping.
            s = min(1.0, (t0 - t_start) / self.soft_start_s) if self.soft_start_s > 0 else 1.0
            s = s * s * (3 - 2 * s)
            cmd = {k: start[k] + s * (v - start[k]) for k, v in action.items()} if s < 1.0 else action
            self.follower.send_action(cmd)
            obs = self.follower.get_observation()
            f_arm, f_pct = bus_reading_to_twin_inputs(self.follower, obs)
            l_arm, l_pct = bus_reading_to_twin_inputs(self.leader, action)
            self.leader_q = self.mapping.to_twin(l_arm, l_pct)
            self._publish(self.mapping.to_twin(f_arm, f_pct), soft_start=s < 1.0)
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))


class UdpSource(JointSource):
    """Receives ``{"v":1,"seq":..,"t":..,"arm_deg":{..},"gripper_pct":..}`` datagrams."""

    name = "udp"

    def __init__(self, mapping: TwinMapping, host: str = "127.0.0.1", port: int = DEFAULT_UDP_PORT):
        super().__init__()
        self.mapping = mapping
        self.addr = (host, port)
        self.sock: socket.socket | None = None
        self.peer = None
        self.rtt_ms: float | None = None

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(self.addr)
        self.sock.settimeout(0.2)
        log.info("listening for joint stream on udp://%s:%d", *self.addr)

    def disconnect(self) -> None:
        if self.sock is not None:
            self.sock.close()

    def send_event(self, event: dict) -> None:
        if self.sock is not None and self.peer is not None:
            with contextlib.suppress(OSError):
                self.sock.sendto(json.dumps({"v": PROTOCOL_VERSION, "event": event}).encode(), self.peer)

    def _run(self) -> None:
        last_seq, last_t = -1, 0.0
        while not self._stop.is_set():
            try:
                payload, peer = self.sock.recvfrom(65535)
            except TimeoutError:
                continue
            except OSError:
                return
            try:
                msg = json.loads(payload)
                if msg.get("v") != PROTOCOL_VERSION:
                    continue
                seq = int(msg["seq"])
                now = time.monotonic()
                # drop reordered datagrams; a sequence reset after a quiet second is a restarted streamer
                if seq <= last_seq and now - last_t < 1.0:
                    continue
                last_seq, last_t = seq, now
                arm = {n: float(msg["arm_deg"][n]) for n in layout.ARM_JOINTS}
                q = self.mapping.to_twin(arm, float(msg["gripper_pct"]))
            except (KeyError, ValueError, TypeError):
                continue
            self.peer = peer
            self.rtt_ms = msg.get("rtt_ms")
            self._publish(q, seq=seq)
            # echo so the streamer can measure round-trip latency
            self.sock.sendto(
                json.dumps({"v": PROTOCOL_VERSION, "ack": seq, "t": msg.get("t")}).encode(), peer
            )


class ReplaySource(JointSource):
    name = "replay"
    live = True

    def __init__(self, path: Path, loop: bool = True):
        super().__init__()
        self.frames = [
            json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        if not self.frames:
            raise ValueError(f"{path} is empty")
        self.loop = loop

    def _run(self) -> None:
        while not self._stop.is_set():
            t0 = time.monotonic()
            for fr in self.frames:
                while time.monotonic() - t0 < fr["t"]:
                    if self._stop.is_set():
                        return
                    time.sleep(0.002)
                self._publish(np.asarray(fr["q"], dtype=float))
            if not self.loop:
                return


class SessionRecorder:
    """Writes the twin's joint targets as JSON lines (replayable with ``--source replay``)."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = path.open("w", encoding="utf-8")
        self.t0: float | None = None

    def write(self, sample: JointSample) -> None:
        if self.t0 is None:
            self.t0 = sample.stamp
        self.f.write(
            json.dumps({"t": round(sample.stamp - self.t0, 4), "q": [round(v, 5) for v in sample.q]}) + "\n"
        )

    def close(self) -> None:
        self.f.close()
