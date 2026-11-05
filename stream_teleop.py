#!/usr/bin/env python
"""Run the SO-101 leader->follower teleop on the machine the arms are plugged into and stream the
follower's measured joints to a Twin Depot running elsewhere (``play.py --source udp``).

    python twin.py stream --twin-host 192.168.1.50      # ports/ids from config.ini
    python stream_teleop.py --twin-host 192.168.1.50 \
        --leader-port COM3 --leader-id my_leader --follower-port COM4 --follower-id my_follower

Ports and ids default to config.ini (``twin ports``). ``--no-follower`` streams the leader alone
(the twin becomes a virtual follower).
``--fake`` streams a synthetic waving motion (no hardware) to test the network path.
Only joint angles travel over the wire; the twin never sends commands back to the robot.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from twin_depot import layout, system  # noqa: E402
from twin_depot.sources import DEFAULT_UDP_PORT, PROTOCOL_VERSION, _make_arm_configs  # noqa: E402
from twin_depot.twin import bus_reading_to_twin_inputs  # noqa: E402


def parse_args():
    cfg = system.load_config()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--twin-host", default="127.0.0.1")
    p.add_argument("--twin-port", type=int, default=DEFAULT_UDP_PORT)
    p.add_argument("--arm", choices=["so101", "so100"], default=cfg["arm"])
    p.add_argument("--leader-port", default=cfg["leader_port"] or None)
    p.add_argument("--leader-id", default=cfg["leader_id"])
    p.add_argument("--follower-port", default=cfg["follower_port"] or None)
    p.add_argument("--follower-id", default=cfg["follower_id"])
    p.add_argument("--no-follower", action="store_true", help="stream the leader alone")
    p.add_argument("--max-relative-target", type=float, default=None)
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--fake", action="store_true", help="synthetic motion, no hardware")
    p.add_argument("--duration", type=float, default=None)
    args = p.parse_args()
    if args.no_follower:
        args.follower_port = None
    return args


def fake_reading(t: float) -> tuple[dict[str, float], float]:
    arm = {
        "shoulder_pan": 35 * math.sin(0.5 * t),
        "shoulder_lift": -20 + 15 * math.sin(0.7 * t),
        "elbow_flex": 20 + 15 * math.sin(0.9 * t + 1),
        "wrist_flex": 70 + 10 * math.sin(1.1 * t),
        "wrist_roll": 30 * math.sin(0.3 * t),
    }
    return arm, 50 + 50 * math.sin(1.3 * t)


def main() -> None:
    args = parse_args()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setblocking(False)
    target = (args.twin_host, args.twin_port)
    leader = follower = None
    if not args.fake:
        if not args.leader_port:
            raise SystemExit("--leader-port is required (or use --fake)")
        from lerobot.robots.utils import make_robot_from_config
        from lerobot.teleoperators.utils import make_teleoperator_from_config

        lcfg, fcfg = _make_arm_configs(
            args.arm,
            args.leader_port,
            args.leader_id,
            args.follower_port,
            args.follower_id,
            args.max_relative_target,
        )
        leader = make_teleoperator_from_config(lcfg)
        leader.connect()
        if fcfg is not None:
            follower = make_robot_from_config(fcfg)
            follower.connect()

    print(f"streaming to udp://{target[0]}:{target[1]} at {args.fps} Hz (Ctrl+C to stop)")
    period, seq, t0, rtt_ms = 1.0 / args.fps, 0, time.monotonic(), None
    sent_at: dict[int, float] = {}
    try:
        while args.duration is None or time.monotonic() - t0 < args.duration:
            tick = time.perf_counter()
            if args.fake:
                arm, pct = fake_reading(time.monotonic() - t0)
            else:
                action = leader.get_action()
                if follower is not None:
                    follower.send_action(action)
                    arm, pct = bus_reading_to_twin_inputs(follower, follower.get_observation())
                else:
                    arm, pct = bus_reading_to_twin_inputs(leader, action)
            msg = {
                "v": PROTOCOL_VERSION,
                "seq": seq,
                "t": time.time(),
                "arm_deg": arm,
                "gripper_pct": pct,
                "rtt_ms": rtt_ms,
                "src": "follower" if follower is not None else ("fake" if args.fake else "leader"),
            }
            sock.sendto(json.dumps(msg).encode(), target)
            sent_at[seq] = time.monotonic()
            seq += 1
            while True:  # drain acks + game events
                try:
                    payload, _ = sock.recvfrom(65535)
                except (BlockingIOError, OSError):
                    break
                reply = json.loads(payload)
                if "ack" in reply and reply["ack"] in sent_at:
                    rtt_ms = round((time.monotonic() - sent_at.pop(reply["ack"])) * 1000, 1)
                elif "event" in reply:
                    print(f"  [twin] {reply['event'].get('zh', reply['event'].get('text'))}", flush=True)
            for k in [k for k in sent_at if k < seq - 200]:
                sent_at.pop(k)
            time.sleep(max(0.0, period - (time.perf_counter() - tick)))
    except KeyboardInterrupt:
        pass
    finally:
        for dev in (follower, leader):
            if dev is not None and dev.is_connected:
                dev.disconnect()
        print(f"sent {seq} frames, last rtt {rtt_ms} ms; joints: {', '.join(layout.ARM_JOINTS)} + gripper")


if __name__ == "__main__":
    main()
