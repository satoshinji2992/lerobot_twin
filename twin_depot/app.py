"""Twin Depot entry point: wires a joint source, the MuJoCo twin, the game and the HUD together."""

from __future__ import annotations

import argparse
import getpass
import logging
import os
import queue
import sys
import threading
import time
from pathlib import Path

from . import system
from .system import IS_WINDOWS, IS_WSL

if IS_WSL:  # WSLg: render on the discrete GPU, not the iGPU
    os.environ.setdefault("MESA_D3D12_DEFAULT_ADAPTER_NAME", "NVIDIA")

import mujoco

from . import layout
from .autopilot import Autopilot
from .game import Game
from .scene import build_scene
from .sources import (
    DEFAULT_UDP_PORT,
    JointSource,
    LeaderOnlySource,
    RealTeleopSource,
    ReplaySource,
    SessionRecorder,
    UdpSource,
)
from .twin import TwinMapping

FRAME_HZ = 60
# GLFW key codes. Mac keyboards have no Insert/PgUp/PgDn/End keys: Home (fn+Left) toggles assist,
# fn+Up / fn+Down / fn+Right / fn+Delete give PgUp / PgDn / End / Delete.
KEY_ENTER, KEY_KP_ENTER, KEY_INSERT, KEY_DELETE, KEY_END = 257, 335, 260, 261, 269
KEY_PAGE_UP, KEY_PAGE_DOWN, KEY_HOME = 266, 267, 268


def _default_player(cfg: dict) -> str:
    if cfg["player"]:
        return cfg["player"]
    try:
        return getpass.getuser()
    except Exception:  # no USER/USERNAME in the environment
        return "operator"


def _setup_console() -> None:
    """Windows consoles: enable ANSI colours and never crash on characters the code page lacks."""
    if IS_WINDOWS:
        os.system("")  # turns on virtual-terminal processing for the colour codes
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")


def parse_args(argv=None) -> argparse.Namespace:
    cfg = system.load_config()  # config.ini (written by `twin ports` / `twin config`) sets the defaults
    leader_port, follower_port, _ = system.arm_ports(cfg)
    p = argparse.ArgumentParser(
        prog="play.py",
        description="SO-101 Twin Depot - a digital-twin parcel sorting game (MuJoCo + LeRobot).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--source",
        choices=["demo", "real", "leader", "udp", "replay"],
        default="demo",
        help="demo: autopilot plays; real: leader drives the real follower and the twin mirrors the "
        "follower; leader: only a leader arm; udp: joints streamed by stream_teleop.py; replay: a --record file",
    )
    p.add_argument("--arm", choices=["so101", "so100"], default=cfg["arm"])
    p.add_argument("--leader-port", default=leader_port)
    p.add_argument("--leader-id", default=cfg["leader_id"])
    p.add_argument("--follower-port", default=follower_port)
    p.add_argument("--follower-id", default=cfg["follower_id"])
    p.add_argument(
        "--max-relative-target",
        type=float,
        default=None,
        help="LeRobot safety clamp for the real follower (normalised units per step)",
    )
    p.add_argument("--fps", type=int, default=60, help="teleop loop rate")
    p.add_argument(
        "--udp-host", default="0.0.0.0", help="interface to listen on for --source udp (twin-only data)"
    )
    p.add_argument("--udp-port", type=int, default=DEFAULT_UDP_PORT)
    p.add_argument("--replay", type=Path, help="JSONL file from --record")
    p.add_argument("--record", type=Path, help="record the twin's joint stream to this JSONL file")
    p.add_argument(
        "--offsets",
        default=None,
        help='per-joint calibration trim in degrees, e.g. "shoulder_lift=3,elbow_flex=-2"',
    )
    p.add_argument("--difficulty", choices=list(layout.DIFFICULTY), default="normal")
    p.add_argument("--shift", type=float, default=150.0, help="shift length in seconds")
    p.add_argument(
        "--hall",
        choices=["auto", "isaac", "builtin"],
        default="auto",
        help="isaac: NVIDIA Isaac Sim's warehouse (import once with import_isaac_scene.py); "
        "builtin: the generated night-shift hall; auto: isaac when imported (builtin on WSL)",
    )
    p.add_argument(
        "--autopilot-miss",
        type=float,
        default=0.18,
        help="demo: share of parcels the autopilot lets ride by (so the recycling cage and tug get work)",
    )
    p.add_argument("--no-assist", action="store_true", help="pure physics grasping (no magnet assist)")
    p.add_argument("--no-hints", action="store_true", help="no glowing target bins")
    p.add_argument("--player", default=_default_player(cfg))
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--shadows",
        action=argparse.BooleanOptionalAction,
        default=not IS_WSL,
        help="shadow-casting spotlight (off by default on WSL, where each extra render pass is costly)",
    )
    p.add_argument("--list-ports", action="store_true", help="list serial ports and exit")
    p.add_argument(
        "--find-ports",
        action="store_true",
        help="unplug/replug each arm to identify its serial port, save to config.ini and exit",
    )
    p.add_argument("--headless", action="store_true", help="no window (tests / benchmarks)")
    p.add_argument("--duration", type=float, default=None, help="stop after this many simulated seconds")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def _comports():
    try:
        from serial.tools import list_ports as lp
    except ImportError:
        raise SystemExit(
            'pyserial missing: install the real-arm extras (`twin setup`, or pip install -e ".[feetech]")'
        ) from None
    return sorted(lp.comports(), key=lambda p: p.device)


def _is_usb(port) -> bool:
    return port.hwid.upper().startswith("USB")  # servo boards are USB-serial; COM1/Bluetooth are not


def _usb_ports() -> set[str]:
    return {p.device for p in _comports() if _is_usb(p)}


def list_ports() -> None:
    """Print serial ports (* = USB). `--find-ports` tells leader from follower automatically."""
    ports = _comports()
    if not ports:
        print("no serial ports found - is the arm's USB cable plugged in and powered?")
    for port in ports:
        print(f"{'*' if _is_usb(port) else ' '} {port.device:<22} {port.description}  [{port.hwid}]")
        hint = system.serial_permission_hint(port.device)
        if hint:
            print(f"  ! {hint}")
    if not any(_is_usb(p) for p in ports):
        print(
            "no USB serial port (*) found - plug in the arms' servo boards (and power them), then run again"
        )


def _identify(role: str) -> str:
    """Unplug the arm, see which port vanished, replug and take the port that comes back."""
    present = _usb_ports()
    if not present:
        raise SystemExit("no USB serial ports found - plug in and power both arms first")
    input(f"\n  Unplug the {role.upper()} arm's USB cable, then press Enter ... ")
    time.sleep(0.5)
    remaining = _usb_ports()
    gone = present - remaining
    if len(gone) != 1:
        raise SystemExit(f"expected exactly one port to disappear, saw {sorted(gone) or 'none'} - try again")
    input(f"  {role} was {gone.pop()}. Plug it back in, then press Enter ... ")
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:  # macOS/Linux may rename the port on re-plug: take the new one
        back = _usb_ports() - remaining
        if len(back) == 1:
            return back.pop()
        time.sleep(0.25)
    raise SystemExit(f"the {role} port did not come back - check the cable and run again")


def find_ports() -> None:
    print(f"[twin] finding the arms' serial ports ({system.OS_NAME}). Current USB serial ports:")
    for dev in sorted(_usb_ports()) or ["(none)"]:
        print(f"    {dev}")
    leader = _identify("leader")
    others = _usb_ports() - {leader}
    follower = others.pop() if len(others) == 1 else _identify("follower")
    cfg = system.save_config({"leader_port": leader, "follower_port": follower})
    print(
        f"\n[twin] leader = {cfg['leader_port']}, follower = {cfg['follower_port']} -> {system.CONFIG_PATH}"
    )
    for port in (leader, follower):
        hint = system.serial_permission_hint(port)
        if hint:
            print(f"  ! {hint}")


def _warn_unconfigured_ports(args) -> None:
    if args.source in ("real", "leader") and not system.arm_ports()[2]:
        cli = {a.split("=")[0] for a in (args.argv or [])}
        if not {"--leader-port", "--follower-port"} & cli:
            print(
                f"[twin] ports not configured, guessing leader={args.leader_port} follower={args.follower_port}"
                " - run `twin ports` (or pass --leader-port/--follower-port)",
                flush=True,
            )


def make_source(args, mapping: TwinMapping) -> JointSource | None:
    if args.source == "real":
        return RealTeleopSource(
            args.leader_port,
            args.leader_id,
            args.follower_port,
            args.follower_id,
            mapping,
            arm=args.arm,
            fps=args.fps,
            max_relative_target=args.max_relative_target,
        )
    if args.source == "leader":
        return LeaderOnlySource(args.leader_port, args.leader_id, mapping, arm=args.arm, fps=args.fps)
    if args.source == "udp":
        return UdpSource(mapping, args.udp_host, args.udp_port)
    if args.source == "replay":
        if not args.replay:
            raise SystemExit("--source replay needs --replay FILE")
        return ReplaySource(args.replay)
    return None


class Session:
    """One running twin: physics + game + (optional) live source. Shared by the viewer and headless loops."""

    def __init__(self, args):
        self.args = args
        hall = args.hall
        if hall == "auto":
            from .isaac_scene import available

            # WSLg's D3D12 GL path chokes on the 680k-triangle warehouse: keep the light hall there
            hall = "isaac" if available() and not IS_WSL else "builtin"
        self.hall = hall
        print(f"[twin] building scene ({hall} hall) ...", flush=True)
        self.m, self.idx, xml_path = build_scene(verbose=True, shadows=args.shadows, hall=hall)
        print(f"[twin] scene ready ({self.m.ngeom} geoms) -> {xml_path}", flush=True)
        self.d = mujoco.MjData(self.m)
        self.autopilot = Autopilot(self.idx, miss_rate=args.autopilot_miss, seed=args.seed)
        self.d.qpos[self.idx.arm_qpos] = self.autopilot.home
        self.d.ctrl[self.idx.arm_act] = self.autopilot.home
        mujoco.mj_forward(self.m, self.d)
        self.mapping = TwinMapping(offset_deg=TwinMapping.parse_offsets(args.offsets))
        self.source = make_source(args, self.mapping)
        mode = "demo" if self.source is None else "live"
        self.game = Game(
            self.m,
            self.d,
            self.idx,
            mode=mode,
            difficulty=args.difficulty,
            shift_s=args.shift,
            assist=not args.no_assist,
            hints=not args.no_hints,
            player=args.player,
            seed=args.seed,
            sync_pose=self.autopilot.home,
            hall=hall,
        )
        self.recorder = SessionRecorder(args.record) if args.record else None
        self.nsub = max(1, round(1.0 / FRAME_HZ / self.m.opt.timestep))
        self.frame_dt = self.nsub * self.m.opt.timestep
        self._last_sample_stamp = None
        self._last_phase = self.game.phase
        self.commands: queue.SimpleQueue[str] = queue.SimpleQueue()  # from key/stdin threads

    def start(self) -> None:
        if self.source is not None:
            print(f"[twin] connecting source '{self.source.name}' ...", flush=True)
            self.source.connect()
            self.source.start()
            self.game.tele.source = self.source.name

    def stop(self) -> None:
        if self.source is not None:
            self.source.stop()
        if self.recorder:
            self.recorder.close()

    def frame(self) -> list:
        g, src = self.game, self.source
        while not self.commands.empty():
            g.command(self.commands.get())
        if g.phase != self._last_phase:  # field gets cleared on most transitions
            self.autopilot.reset(g.q_target)
            self._last_phase = g.phase
        sample = src.latest() if src is not None else None
        if src is None or g.phase == "lobby":
            g.set_target(self.autopilot.step(g.t, self.d, g))
            if src is None:
                g.tele.source, g.tele.rate_hz, g.tele.age_ms = "autopilot", FRAME_HZ, 0.0
        if src is not None:
            g.tele.rate_hz = src.rate_hz()
            g.tele.age_ms = (time.monotonic() - sample.stamp) * 1000 if sample else 1e9
            if src.error:
                g.tele.source = f"{src.name} ERROR: {src.error[:40]}"
            if sample is not None and g.phase != "lobby":
                g.set_target(sample.q)
            if sample is not None and self.recorder and sample.stamp != self._last_sample_stamp:
                self.recorder.write(sample)
            self._last_sample_stamp = sample.stamp if sample else None
        g.pre_step(self.frame_dt)
        for _ in range(self.nsub):
            mujoco.mj_step(self.m, self.d)
        g.post_step(self.frame_dt)
        events = g.pop_events()
        if src is not None:
            for ev in events:
                src.send_event({"kind": ev.kind, "text": ev.text, "zh": ev.zh, "points": ev.points})
        return events


def _stdin_commands(commands: queue.SimpleQueue) -> None:
    aliases = {
        "": "start",
        "s": "start",
        "start": "start",
        "a": "assist",
        "h": "hints",
        "r": "reset",
        "c": "call_agv",
        "agv": "call_agv",
        "t": "call_tug",
        "tug": "call_tug",
        "q": "quit",
    }
    for line in sys.stdin:
        cmd = aliases.get(line.strip().lower())
        if cmd:
            commands.put(cmd)
        if cmd == "quit":
            return


def run_viewer(sess: Session) -> None:
    import mujoco.viewer

    from .hud import Hud

    hud = Hud(sess.m, sess.game, sess.autopilot.home)
    keymap = {
        KEY_ENTER: "start",
        KEY_KP_ENTER: "start",
        KEY_INSERT: "assist",
        KEY_HOME: "assist",
        KEY_DELETE: "hints",
        KEY_END: "reset",
        KEY_PAGE_UP: "call_agv",
        KEY_PAGE_DOWN: "call_tug",
    }

    def on_key(key: int) -> None:
        if key in keymap:
            sess.commands.put(keymap[key])

    if sys.stdin.isatty():
        threading.Thread(target=_stdin_commands, args=(sess.commands,), daemon=True).start()
    with mujoco.viewer.launch_passive(
        sess.m, sess.d, key_callback=on_key, show_left_ui=False, show_right_ui=False
    ) as v:
        with v.lock():
            v.cam.lookat[:] = (0.12, 0.02, 0.02)
            v.cam.distance, v.cam.azimuth, v.cam.elevation = 0.95, 0.0, -38.0
            v.opt.sitegroup[:] = 0
        print(
            "[twin] viewer up. Terminal: Enter=start a=assist h=hints r=reset c=call AGV t=call tug q=quit",
            flush=True,
        )
        while v.is_running() and not sess.game.quit_requested:
            t0 = time.perf_counter()
            with v.lock():
                events = sess.frame()
                hud.on_events(events, sess.game.t)
                hud.draw(v)
            v.sync()
            # after sync(): set_texts blocks until the render thread has consumed the previous texts
            v.set_texts(hud.texts(sess.game.t))
            if sess.args.duration and sess.game.t >= sess.args.duration:
                break
            time.sleep(max(0.0, sess.frame_dt - (time.perf_counter() - t0)))


def run_headless(sess: Session, realtime: bool) -> None:
    from .hud import Hud

    hud = Hud(sess.m, sess.game, sess.autopilot.home)
    duration = sess.args.duration or 60.0
    while sess.game.t < duration and not sess.game.quit_requested:
        t0 = time.perf_counter()
        hud.on_events(sess.frame(), sess.game.t)
        if realtime:
            time.sleep(max(0.0, sess.frame_dt - (time.perf_counter() - t0)))


def main(argv=None) -> None:
    _setup_console()
    args = parse_args(argv)
    args.argv = list(sys.argv[1:] if argv is None else argv)
    if args.list_ports:
        list_ports()
        return
    if args.find_ports:
        find_ports()
        return
    _warn_unconfigured_ports(args)
    if not args.headless:
        system.ensure_viewer_process()  # macOS: hop into mjpython before building anything
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s"
    )
    sess = Session(args)
    try:
        sess.start()
        if args.headless:
            run_headless(sess, realtime=sess.source is not None)
        else:
            run_viewer(sess)
    except KeyboardInterrupt:
        pass
    finally:
        sess.stop()
        r = sess.game.results
        if r:
            print(
                f"[twin] last shift: score {r['score']} rank {r['rank']} ({r['title_zh']}), accuracy {r['accuracy']}%"
            )
        if not args.headless:
            # GLFW's atexit teardown can segfault under WSLg/Wayland; everything is cleaned up by now.
            sys.stdout.flush()
            os._exit(0)
