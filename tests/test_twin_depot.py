"""Headless tests for Twin Depot (no window, no hardware).

conda activate lerobot
python -m pytest tests -q   (or: python twin.py tests)
"""

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from twin_depot import layout  # noqa: E402
from twin_depot.app import Session, parse_args  # noqa: E402
from twin_depot.twin import (  # noqa: E402
    STS3215_MAX_RES,
    TwinMapping,
    bus_reading_to_twin_inputs,
    gripper_percent,
    lerobot_to_degrees,
)


# --------------------------------------------------------------------------------- mapping
def _lerobot_normalize(raw, mode, lo, hi):
    """Re-implementation of MotorsBus._normalize for the round trip."""
    if mode == "range_m100_100":
        return ((raw - lo) / (hi - lo)) * 200 - 100
    if mode == "range_0_100":
        return ((raw - lo) / (hi - lo)) * 100
    return (raw - (lo + hi) / 2) * 360 / STS3215_MAX_RES


@pytest.mark.parametrize("mode", ["range_m100_100", "range_0_100", "degrees"])
def test_every_norm_mode_maps_to_the_same_degrees(mode):
    lo, hi = 900, 3100
    for raw in (900, 1500, 2000, 2600, 3100):
        expected = (raw - (lo + hi) / 2) * 360 / STS3215_MAX_RES
        got = lerobot_to_degrees(_lerobot_normalize(raw, mode, lo, hi), mode, lo, hi)
        assert got == pytest.approx(expected, abs=1e-9)


def test_gripper_percent_and_twin_mapping_roundtrip():
    assert gripper_percent(37.0, "range_0_100", 1000, 3000) == 37.0
    mapping = TwinMapping(offset_deg=TwinMapping.parse_offsets("shoulder_lift=3,elbow_flex=-2"))
    arm = {"shoulder_pan": 10, "shoulder_lift": -20, "elbow_flex": 30, "wrist_flex": 45, "wrist_roll": -90}
    q = mapping.to_twin(arm, 25.0)
    assert q[1] == pytest.approx(math.radians(-17))
    assert q[5] == pytest.approx(
        layout.GRIPPER_CLOSED_RAD + 0.25 * (layout.GRIPPER_OPEN_RAD - layout.GRIPPER_CLOSED_RAD)
    )
    back, pct = mapping.from_twin(q)
    assert pct == pytest.approx(25.0)
    assert all(back[k] == pytest.approx(v) for k, v in arm.items())
    with pytest.raises(ValueError):
        TwinMapping.parse_offsets("elbow=3")


def test_bus_reading_uses_each_motors_calibration():
    cal = {n: SimpleNamespace(range_min=1000, range_max=3000, drive_mode=0) for n in layout.ALL_JOINTS}
    motors = {
        n: SimpleNamespace(norm_mode=SimpleNamespace(value="range_m100_100")) for n in layout.ARM_JOINTS
    }
    motors["gripper"] = SimpleNamespace(norm_mode=SimpleNamespace(value="range_0_100"))
    dev = SimpleNamespace(calibration=cal, bus=SimpleNamespace(motors=motors))
    reading = {f"{n}.pos": 0.0 for n in layout.ARM_JOINTS} | {"shoulder_pan.pos": 100.0, "gripper.pos": 80.0}
    arm, pct = bus_reading_to_twin_inputs(dev, reading)
    assert arm["elbow_flex"] == pytest.approx(0.0)  # middle of range -> URDF/MJCF zero
    assert arm["shoulder_pan"] == pytest.approx(1000 * 360 / STS3215_MAX_RES)
    assert pct == 80.0


# --------------------------------------------------------------------------------- sim
@pytest.fixture(scope="module")
def session():
    return Session(
        parse_args(
            ["--source", "demo", "--seed", "3", "--shift", "60", "--hall", "builtin", "--autopilot-miss", "0"]
        )
    )


def test_scene_has_everything_the_game_needs(session):
    idx = session.idx
    assert len(idx.parcels) == sum(t.pool for t in layout.PARCEL_TYPES)
    assert set(idx.bin_geoms) == {b.key for b in layout.BINS}
    lo, hi = idx.joint_ranges[:5].T
    assert np.all(session.autopilot.home[:5] >= lo) and np.all(session.autopilot.home[:5] <= hi)


def test_magnet_assist_weld_keeps_relative_pose(session):
    m, idx = session.m, session.idx
    d = mujoco.MjData(m)
    d.qpos[idx.arm_qpos] = session.autopilot.home
    d.ctrl[idx.arm_act] = session.autopilot.home
    mujoco.mj_forward(m, d)
    game = session.game.__class__(m, d, idx, mode="demo")
    p = game.parcels[0]
    gc = d.site_xpos[idx.grasp_site].copy()
    game._teleport(p, gc)
    mujoco.mj_forward(m, d)
    game._weld(p)
    rel0 = d.xpos[idx.parcel_body[0]] - d.xpos[idx.gripper_body]
    for _ in range(500):
        mujoco.mj_step(m, d)
    rel1 = d.xpos[idx.parcel_body[0]] - d.xpos[idx.gripper_body]
    assert np.linalg.norm(rel1 - rel0) < 0.004, "welded parcel should ride along with the gripper"
    game._unweld()
    for _ in range(500):
        mujoco.mj_step(m, d)
    assert d.xpos[idx.parcel_body[0]][2] < gc[2] - 0.01, "released parcel should fall"


def test_autopilot_shift_sorts_parcels_without_mistakes(session):
    game = session.game
    while game.t < 66 and game.results is None:
        session.frame()
    assert game.results is not None, "shift should have finished"
    assert game.results["sorted"] >= 6
    assert game.results["errors"] == 0


def test_double_pinch_starts_a_shift(session):
    game = session.game
    game.mode, game.phase = "live", "ready"
    open_q, closed_q = session.autopilot.home.copy(), session.autopilot.home.copy()
    open_q[5], closed_q[5] = 1.2, 0.0
    for q in (closed_q, open_q, closed_q, open_q):
        for _ in range(10):  # drive the game directly: in demo the autopilot would own the target
            game.set_target(q)
            game.pre_step(session.frame_dt)
            for _ in range(session.nsub):
                mujoco.mj_step(session.m, session.d)
            game.post_step(session.frame_dt)
    assert game.phase == "countdown"


def _drive(session, game, seconds: float) -> None:
    """Step the game without the autopilot (the arm just holds its pose)."""
    for _ in range(int(seconds / session.frame_dt)):
        game.pre_step(session.frame_dt)
        for _ in range(session.nsub):
            mujoco.mj_step(session.m, session.d)
        game.post_step(session.frame_dt)


@pytest.fixture(scope="module")
def line_session():
    return Session(
        parse_args(
            ["--source", "demo", "--seed", "4", "--shift", "60", "--hall", "builtin", "--autopilot-miss", "0"]
        )
    )


def _park_everything(game):
    for p in game.parcels:
        if p.state != "parked":
            game._park(p)
    game.cage_count = 0


def test_unpicked_parcel_rides_past_the_robot_into_the_recycling_cage(line_session):
    s = line_session
    game = s.game
    game.phase = "shift"
    _park_everything(game)
    _drive(s, game, 2.0)  # belt spins up
    c = layout.CONVEYOR
    assert game.unload("standard", y=c.reach_hi)
    p = next(p for p in game.parcels if p.state == "belt")
    p.released = True  # already through the metering gate
    _drive(s, game, 1.0)
    assert abs(game._vel(p)[1] + game.belt_speed) < 0.15 * game.belt_speed, "rides at pick-section speed"
    assert game.time_to_slot(p) is not None
    score0 = game.stats.score
    _drive(s, game, 25.0)  # never stops in front of the robot: off the end, into the cage
    assert p.state == "caged"
    assert game.stats.missed == 1 and game.cage_count == 1
    assert game.stats.score == score0 + layout.MISS_PENALTY


def test_metering_gate_holds_the_queue_and_releases_one_at_a_time(line_session):
    s = line_session
    game = s.game
    game.phase = "shift"
    _park_everything(game)
    game.next_release = game.t + 100.0  # gate closed
    for k in range(3):
        assert game.unload("express", y=layout.CONVEYOR.y_in - 0.05 - 0.1 * k)
    _drive(s, game, 12.0)
    q = game.queue()
    ys = [float(game._pos(p)[1]) for p in q]
    assert len(q) == 3 and abs(ys[0] - layout.CONVEYOR.gate_y) < 0.01, ys
    assert all(b - a > layout.CONVEYOR.queue_pitch * 0.8 for a, b in zip(ys, ys[1:], strict=False)), (
        "queued, not piled up"
    )
    game.next_release = game.t
    _drive(s, game, 0.5)
    assert len(game.queue()) == 2, "exactly one parcel released"


def test_tug_hauls_a_full_cage_away_and_brings_it_back():
    s = Session(
        parse_args(
            [
                "--source",
                "demo",
                "--seed",
                "5",
                "--shift",
                "120",
                "--hall",
                "builtin",
                "--autopilot-miss",
                "0",
            ]
        )
    )
    game = s.game
    while game.t < 2.0:
        s.frame()
    game.cage_count = layout.CAGE.capacity
    seen_away = False
    while game.t < 90.0 and not (seen_away and game.cage_present):
        s.frame()
        seen_away |= not game.cage_present
    assert seen_away, "the tug should have hooked the cage"
    assert game.cage_present and game.cage_count == 0
    assert np.allclose(game.cage_pos, layout.CAGE.center, atol=0.02), "cage back in place"


def test_route_planner_goes_around_obstacles(line_session):
    grid = line_session.game.fleet.grid
    start, goal = np.array([0.15, 2.9]), np.array([0.15, -3.8])  # opposite ends of the long conveyor
    path = grid.plan(start, goal)
    pts = [start] + path
    assert np.allclose(path[-1], goal)
    assert all(grid.line_free(a, b) for a, b in zip(pts, pts[1:], strict=False))
    assert max(abs(p[0] - 0.15) for p in path) > 0.6, "must detour around the table and the belt"


def _press(s, game, key: str, seconds: float = 0.6) -> None:
    """Push a CALL button's cap down (like a fingertip would) and let go."""
    b = layout.BUTTON_BY_KEY[key]
    dof = s.m.jnt_dofadr[s.m.joint(f"button_{key}").id]
    for _ in range(int(seconds / s.frame_dt)):
        game.pre_step(s.frame_dt)
        for _ in range(s.nsub):
            s.d.qfrc_applied[dof] = -3.0  # ~3 N down, well past the spring
            mujoco.mj_step(s.m, s.d)
        game.post_step(s.frame_dt)
    s.d.qfrc_applied[dof] = 0.0
    _drive(s, game, 0.5)
    assert game.button_depth(key) < b.trigger, "spring pushes the cap back up"


def test_call_buttons_dispatch_the_vehicles():
    s = Session(
        parse_args(
            [
                "--source",
                "demo",
                "--seed",
                "6",
                "--shift",
                "120",
                "--hall",
                "builtin",
                "--autopilot-miss",
                "0",
            ]
        )
    )
    game, fl = s.game, s.game.fleet
    game.phase = "ready"  # belt stopped: nothing dispatches by itself
    _drive(s, game, 9.0)  # the AGV drives to the racks and loads
    assert fl.state["agv"] == "ready"
    _press(s, game, "agv")
    assert game.button_presses["agv"] == 1
    _drive(s, game, 1.0)
    assert fl.state["agv"] in ("delivering", "unloading"), "CALL AGV sends the loaded AGV"
    # tug: refuses an empty cage, comes when there is something in it
    _press(s, game, "tug")
    assert not fl.called["tug"] and fl.state["tug"] == "parked"
    game.cage_count = 1
    _drive(s, game, 1.2)  # debounce
    game.command("call_tug")  # same as PageDown / terminal "t"
    _drive(s, game, 1.0)
    assert fl.state["tug"] == "busy"


def test_autopilot_presses_the_call_buttons_itself():
    s = Session(
        parse_args(
            ["--source", "demo", "--seed", "1", "--shift", "60", "--hall", "builtin", "--autopilot-miss", "0"]
        )
    )
    while s.game.t < 50:
        s.frame()
    assert s.game.button_presses["agv"] >= 1, "it should call the AGV before the line runs dry"


def test_isaac_warehouse_scene_builds_when_imported():
    from twin_depot import isaac_scene
    from twin_depot.scene import build_scene

    if not isaac_scene.available():
        pytest.skip("Isaac warehouse not imported (python import_isaac_scene.py)")
    m, idx, _ = build_scene(verbose=False, shadows=False, hall="isaac")
    names = [m.geom(g).name for g in range(m.ngeom)]
    isaac = [g for g, n in enumerate(names) if n.startswith("isaac_")]
    assert len(isaac) > 50
    assert all(m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0 for g in isaac), (
        "backdrop must be visual only"
    )
    assert "floor_collision" in names and "belt" in names


# --------------------------------------------------------------------------------- cross-platform
@pytest.mark.parametrize(
    "os_flags, devices, expected",
    [
        ({"IS_WINDOWS": True, "IS_MAC": False}, [], ("COM3", "COM4")),
        ({"IS_WINDOWS": False, "IS_MAC": True}, ["/dev/tty.usbmodem5A7A1", "/dev/tty.usbmodem5A4B2"],
         ("/dev/tty.usbmodem5A4B2", "/dev/tty.usbmodem5A7A1")),
        ({"IS_WINDOWS": False, "IS_MAC": True}, [], ("/dev/tty.usbmodem1", "/dev/tty.usbmodem2")),
        ({"IS_WINDOWS": False, "IS_MAC": False}, ["/dev/ttyACM1"], ("/dev/ttyACM1", "/dev/ttyACM0")),
        ({"IS_WINDOWS": False, "IS_MAC": False}, [], ("/dev/ttyACM0", "/dev/ttyACM1")),
    ],
)  # fmt: skip
def test_port_guesses_follow_each_os_naming(monkeypatch, os_flags, devices, expected):
    from twin_depot import system

    for k, v in os_flags.items():
        monkeypatch.setattr(system, k, v)
    monkeypatch.setattr(system.glob, "glob", lambda pat: [d for d in devices if d.startswith(pat[:-1])])
    assert system.guess_ports() == expected  # boards found come first, fallbacks fill the rest


def test_config_file_sets_the_play_defaults(monkeypatch, tmp_path):
    from twin_depot import system

    monkeypatch.setattr(system, "CONFIG_PATH", tmp_path / "config.ini")
    args = parse_args([])
    assert args.leader_id == "my_awesome_leader_arm" and not system.arm_ports()[2]
    system.save_config({"leader_port": "/dev/cu.usbmodemA", "follower_port": "COM7", "player": "Sam"})
    with pytest.raises(KeyError):
        system.save_config({"leder_port": "typo"})
    args = parse_args(["--follower-port", "COM9"])
    assert (args.leader_port, args.follower_port, args.player) == ("/dev/cu.usbmodemA", "COM9", "Sam")
    assert system.arm_ports()[2]


def test_launcher_runs_commands_inside_the_environment(tmp_path):
    import os
    import subprocess

    sim = Path(__file__).resolve().parents[1]
    env = {**os.environ, "TWIN_PYTHON": sys.executable}
    env.pop("TWIN_IN_ENV", None)
    out = subprocess.run(
        [sys.executable, str(sim / "twin.py"), "demo", "--headless", "--duration", "1", "--hall", "builtin"],
        capture_output=True, text=True, env=env, timeout=300,
    )  # fmt: skip
    assert out.returncode == 0, out.stdout + out.stderr
    assert "scene ready" in out.stdout
    helped = subprocess.run([sys.executable, str(sim / "twin.py"), "help"], capture_output=True, text=True)
    assert helped.returncode == 0 and "calibrate" in helped.stdout
    bad = subprocess.run(
        [sys.executable, str(sim / "twin.py"), "dmeo"], capture_output=True, text=True, env=env
    )
    assert bad.returncode != 0


def test_macos_viewer_relaunches_under_mjpython(monkeypatch, tmp_path):
    import mujoco.viewer

    from twin_depot import system

    calls = []

    class ExecError(Exception):
        pass

    def fake_execv(exe, argv):
        calls.append((exe, argv))
        raise ExecError

    fake = tmp_path / "mjpython"
    fake.write_text("")
    monkeypatch.setattr(system, "IS_MAC", True)
    monkeypatch.setattr(system, "mjpython_path", lambda: str(fake))
    monkeypatch.setattr(system.os, "execv", fake_execv)
    monkeypatch.setattr(system.sys, "orig_argv", ["python3", "sorting_sim/twin.py", "demo"])
    monkeypatch.setattr(mujoco.viewer, "_MJPYTHON", None)
    monkeypatch.setenv("TWIN_MJPYTHON", "")  # restored afterwards (the hop sets it to 1)
    with pytest.raises(ExecError):
        system.ensure_viewer_process()
    assert calls == [(str(fake), [str(fake), "sorting_sim/twin.py", "demo"])]
    with pytest.raises(SystemExit):  # already under mjpython but still no viewer: stop, don't loop
        system.ensure_viewer_process()
    monkeypatch.setattr(mujoco.viewer, "_MJPYTHON", object())
    system.ensure_viewer_process()  # under mjpython: nothing to do
    monkeypatch.setattr(system, "IS_MAC", False)
    system.ensure_viewer_process()


def test_saved_serial_numbers_follow_the_boards_to_new_ports():
    from twin_depot import system

    def port(dev, sn):
        return SimpleNamespace(device=dev, serial_number=sn, hwid=f"USB VID:PID=1A86:55D3 SER={sn}")

    cfg = {**system.DEFAULTS, "leader_port": "COM5", "follower_port": "COM6",
           "leader_serial": "5A7A0123", "follower_serial": "5A7A0456"}  # fmt: skip
    same = system.resolve_ports(cfg, [port("COM5", "5A7A0123"), port("COM6", "5A7A0456")])
    assert same == ("COM5", "COM6", True, [])
    # re-plugged into other sockets and swapped COM numbers: still the right arms
    moved = system.resolve_ports(cfg, [port("COM9", "5A7A0456"), port("COM8", "5A7A0123")])
    assert (moved.leader, moved.follower) == ("COM8", "COM9") and len(moved.notes) == 2
    # follower unplugged: keep its last known port and say so
    gone = system.resolve_ports(cfg, [port("COM8", "5A7A0123")])
    assert (gone.leader, gone.follower) == ("COM8", "COM6") and "not plugged in" in gone.notes[-1]
    # boards without serial numbers fall back to the saved names; pyserial missing changes nothing
    plain = {**cfg, "leader_serial": "", "follower_serial": ""}
    assert system.resolve_ports(plain, [port("COM8", None)])[:3] == ("COM5", "COM6", True)
    assert system.resolve_ports(cfg, None)[:3] == ("COM5", "COM6", True)
    # two clones sharing one serial number are ambiguous: don't trust it
    clones = system.resolve_ports(cfg, [port("COM8", "5A7A0123"), port("COM9", "5A7A0123")])
    assert clones.leader == "COM5"


def test_find_ports_records_serial_numbers(monkeypatch, tmp_path):
    from twin_depot import app, system

    monkeypatch.setattr(system, "CONFIG_PATH", tmp_path / "config.ini")
    boards = {"leader": SimpleNamespace(device="/dev/ttyACM0", serial_number="AAA", hwid="USB VID:PID=1A86:55D3",
                                        description="USB Single Serial"),
              "follower": SimpleNamespace(device="/dev/ttyACM1", serial_number="BBB", hwid="USB VID:PID=1A86:55D3",
                                          description="USB Single Serial")}  # fmt: skip
    plugged = dict(boards)
    script = iter([lambda: plugged.pop("leader"), lambda: plugged.update(leader=boards["leader"])])
    monkeypatch.setattr(app, "_comports", lambda: list(plugged.values()))
    monkeypatch.setattr("builtins.input", lambda *_: next(script)())
    monkeypatch.setattr(app.time, "sleep", lambda s: None)
    app.find_ports()
    cfg = system.load_config()
    assert (cfg["leader_port"], cfg["leader_serial"]) == ("/dev/ttyACM0", "AAA")
    assert (cfg["follower_port"], cfg["follower_serial"]) == ("/dev/ttyACM1", "BBB")
