#!/usr/bin/env python3
"""Twin Depot launcher: the same commands on Windows, macOS and Linux.

    python twin.py setup       # once: Python env + LeRobot + MuJoCo + Isaac warehouse
    python twin.py demo        # autopilot plays, no hardware
    python twin.py ports       # unplug/replug each arm -> config.ini
    python twin.py calibrate   # LeRobot calibration of both arms
    python twin.py real        # leader drives the real follower, the twin mirrors it

Wrappers: twin.bat (Windows; double-click for a menu) and twin.sh (macOS / Linux). Arguments after
the command go straight to play.py, e.g. `twin demo --difficulty hard`. This file only needs the
standard library: any Python 3 can run `setup`, and every other command re-runs itself inside the
environment that setup created (recorded in .twin-python).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

SIM = Path(__file__).resolve().parent
sys.path.insert(0, str(SIM))

from twin_depot import system  # noqa: E402  (stdlib only)

ENV_FILE = SIM / ".twin-python"  # absolute path of the environment's interpreter
VENV_DIR = SIM / ".venv"
IN_ENV = "TWIN_IN_ENV"
PY_VERSION = "3.12"
PACKAGES = ["mujoco", "pillow", "pytest", "usd-core"]
LEROBOT = "lerobot[feetech]>=0.4.4"  # PyPI; the first release with lerobot.robots.so_follower


def lerobot_spec() -> list[str]:
    """Editable install when this folder sits inside a lerobot source checkout (dev), PyPI otherwise."""
    parent = SIM.parent / "pyproject.toml"
    if parent.exists() and 'name = "lerobot"' in parent.read_text(encoding="utf-8", errors="replace"):
        return ["-e", f"{SIM.parent}[feetech]"]
    return [LEROBOT]


PLAY = {
    "demo": "autopilot demo, no hardware",
    "real": "leader drives the real follower; the twin mirrors the follower",
    "leader": "leader arm only; the twin is the follower",
    "udp": "joints streamed from another computer (`twin stream` there)",
    "replay": "replay a --record file: twin replay --replay runs/x.jsonl",
}
COMMANDS = {
    "setup": "one-time install (conda, uv or venv) + Isaac warehouse import + self-test",
    **PLAY,
    "ports": "identify the leader/follower serial ports (unplug/replug) -> config.ini; --list to only list",
    "calibrate": "LeRobot calibration: twin calibrate [leader|follower]  (default: both)",
    "stream": "on the computer with the arms: stream joints to a twin elsewhere (--twin-host IP)",
    "import-isaac": "(re)import the Isaac Sim warehouse backdrop",
    "tests": "run the test suite",
    "config": "show settings, or set them: twin config leader_id=alice player=Sam",
    "doctor": "check this machine: Python, MuJoCo, OpenGL window, serial ports, Isaac assets",
}
MENU = ["demo", "real", "leader", "ports", "calibrate", "setup", "doctor", "tests"]


def say(msg: str) -> None:
    print(f"[twin] {msg}", flush=True)


def run(cmd: list, check: bool = True, **kw) -> int:
    print("  $ " + " ".join(str(c) for c in cmd), flush=True)
    rc = subprocess.call([str(c) for c in cmd], **kw)
    if check and rc != 0:
        raise SystemExit(f"[twin] command failed (exit {rc})")
    return rc


# ------------------------------------------------------------------------------ environments
def _bin(env: Path, name: str) -> Path:
    return env / (f"Scripts/{name}.exe" if system.IS_WINDOWS else f"bin/{name}")


def _env_python_of(env: Path) -> Path:
    return env / "python.exe" if system.IS_WINDOWS else env / "bin" / "python"


def find_conda() -> str | None:
    for c in (os.environ.get("CONDA_EXE"), shutil.which("conda"), shutil.which("mamba")):
        if c and Path(c).exists():
            return c
    home = Path.home()
    roots = [home / n for n in ("miniconda3", "anaconda3", "miniforge3", "mambaforge")]
    if system.IS_WINDOWS:
        for base in (os.environ.get("PROGRAMDATA"), os.environ.get("LOCALAPPDATA")):
            if base:
                roots += [Path(base) / "miniconda3", Path(base) / "anaconda3"]
    else:
        roots += [Path("/opt") / n for n in ("miniconda3", "anaconda3", "miniforge3")]
        roots += [Path("/opt/homebrew/Caskroom/miniconda/base"), Path("/usr/local/Caskroom/miniconda/base")]
    for root in roots:
        exe = _bin(root, "conda")
        if exe.exists():
            return str(exe)
    return None


def conda_env_path(name: str) -> Path | None:
    conda = find_conda()
    if not conda:
        return None
    try:
        out = subprocess.run([conda, "env", "list", "--json"], capture_output=True, text=True, timeout=60)
        envs = json.loads(out.stdout)["envs"]
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired):
        return None
    for env in envs:
        if Path(env).name == name and _env_python_of(Path(env)).exists():
            return Path(env)
    return None


def _has_mujoco(py: str) -> bool:
    return subprocess.call([py, "-c", "import mujoco"], stderr=subprocess.DEVNULL) == 0


def env_python(allow_current: bool = True) -> str | None:
    """The interpreter commands run under: TWIN_PYTHON, .twin-python, .venv, the conda env, or this one."""
    candidates = [os.environ.get("TWIN_PYTHON")]
    if ENV_FILE.exists():
        candidates.append(ENV_FILE.read_text(encoding="utf-8").strip())
    candidates.append(str(_env_python_of(VENV_DIR)))
    for c in candidates:
        if c and Path(c).exists():
            return c
    env = conda_env_path(system.load_config()["conda_env"])
    if env:
        return str(_env_python_of(env))
    if allow_current and _has_mujoco(sys.executable):
        return sys.executable  # e.g. an already-activated dev environment
    return None


def _same(a: str, b: str) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:
        return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def enter_env(argv: list[str]) -> None:
    """Re-run this command inside the twin environment (no-op when already there)."""
    if os.environ.get(IN_ENV):
        return
    py = env_python()
    if py is None:
        raise SystemExit("[twin] no environment yet - run `twin setup` first")
    os.environ[IN_ENV] = "1"
    os.environ.setdefault("PYTHONUTF8", "1")
    if _same(py, sys.executable):
        return
    try:
        rc = subprocess.call([py, str(Path(__file__).resolve()), *argv])
    except KeyboardInterrupt:  # Ctrl+C reaches the child too; let it finish cleanly
        rc = 130
    raise SystemExit(rc)


# ------------------------------------------------------------------------------ setup
def cmd_setup(rest: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="twin setup")
    ap.add_argument("--env", choices=["auto", "conda", "uv", "venv"], default="auto")
    ap.add_argument("--no-isaac", action="store_true", help="skip the ~380 MB Isaac warehouse download")
    ap.add_argument("--demo-only", action="store_true", help="only MuJoCo (no LeRobot, no real arms)")
    a = ap.parse_args(rest)
    cfg = system.load_config()
    say(f"setting up on {system.OS_NAME} ({sys.platform}, Python {sys.version.split()[0]})")

    py = env_python(allow_current=False) if a.env == "auto" else None
    uv = shutil.which("uv")
    if py:
        say(f"using the existing environment {py}")
    else:
        kind = a.env
        if kind == "auto":
            kind = "conda" if find_conda() else "uv" if uv else "venv"
        if kind == "conda":
            conda = find_conda() or sys.exit(
                "[twin] conda not found (https://docs.conda.io/en/latest/miniconda.html)"
            )
            name = cfg["conda_env"]
            env = conda_env_path(name)
            if env is None:
                say(f"creating conda env {name} (Python {PY_VERSION}) ...")
                run(
                    [
                        conda,
                        "create",
                        "-y",
                        "-n",
                        name,
                        "--override-channels",
                        "-c",
                        "conda-forge",
                        f"python={PY_VERSION}",
                    ]
                )
                env = conda_env_path(name) or sys.exit(f"[twin] conda env {name} not found after creating it")
            py = str(_env_python_of(env))
        elif kind == "uv":
            uv = uv or sys.exit("[twin] uv not found (https://docs.astral.sh/uv/)")
            say(f"creating {VENV_DIR} with uv (Python {PY_VERSION}) ...")
            run([uv, "venv", "--seed", "--python", PY_VERSION, VENV_DIR])
            py = str(_env_python_of(VENV_DIR))
        else:
            if not ((3, 10) <= sys.version_info[:2] <= (3, 12)):
                sys.exit(
                    f"[twin] Python {sys.version.split()[0]} can't host LeRobot here: install Miniconda or uv, "
                    f"or run setup with Python 3.10-3.12"
                )
            say(f"creating {VENV_DIR} ...")
            run([sys.executable, "-m", "venv", VENV_DIR])
            py = str(_env_python_of(VENV_DIR))
    ENV_FILE.write_text(py + "\n", encoding="utf-8")

    pip = [py, "-m", "pip", "install"]
    run([*pip, "--upgrade", "pip"])
    if a.demo_only:
        run([*pip, "numpy", *PACKAGES])
    else:
        run([*pip, *lerobot_spec(), *PACKAGES])
    if not a.no_isaac and run([py, SIM / "import_isaac_scene.py"], check=False) != 0:
        say("Isaac warehouse import failed - the built-in hall will be used. Retry: twin import-isaac")
    run([py, SIM / "play.py", "--headless", "--duration", "1"])
    say("setup done. Next: twin demo   |   real arms: twin ports -> twin calibrate -> twin real")


# ------------------------------------------------------------------------------ commands (in env)
def cmd_play(source: str, rest: list[str]) -> None:
    from twin_depot.app import main

    main(["--source", source, *rest])


def cmd_ports(rest: list[str]) -> None:
    from twin_depot.app import find_ports, list_ports

    list_ports() if "--list" in rest else find_ports()


def cmd_calibrate(rest: list[str]) -> None:
    cfg = system.load_config()
    leader, follower, configured, notes = system.arm_ports(cfg)
    for note in notes:
        say(note)
    which = [r for r in rest if r in ("leader", "follower")] or ["follower", "leader"]
    extra = [r for r in rest if r not in ("leader", "follower")]
    if not configured:
        say(
            f"ports not configured (guessing leader={leader} follower={follower}) - `twin ports` first is safer"
        )
    for role in which:
        port, arm_id = (leader, cfg["leader_id"]) if role == "leader" else (follower, cfg["follower_id"])
        kind = "teleop" if role == "leader" else "robot"
        say(f"calibrating {role} {arm_id} on {port}: middle pose, then sweep every joint through its range")
        run(
            [sys.executable, "-m", "lerobot.scripts.lerobot_calibrate",
             f"--{kind}.type={cfg['arm']}_{role}", f"--{kind}.port={port}", f"--{kind}.id={arm_id}", *extra]
        )  # fmt: skip
    try:
        from lerobot.utils.constants import HF_LEROBOT_CALIBRATION

        say(f"calibration saved under {HF_LEROBOT_CALIBRATION}")
    except ImportError:
        pass


def cmd_import_isaac(rest: list[str]) -> None:
    try:
        import pxr  # noqa: F401
    except ImportError:
        run([sys.executable, "-m", "pip", "install", "usd-core"])
    run([sys.executable, SIM / "import_isaac_scene.py", *rest])


def cmd_config(rest: list[str]) -> None:
    updates = {}
    for item in rest:
        key, sep, value = item.partition("=")
        if not sep:
            sys.exit(f"[twin] expected key=value, got {item!r}")
        updates[key.strip().replace("-", "_")] = value.strip()
    try:
        cfg = system.save_config(updates) if updates else system.load_config()
    except KeyError as e:
        sys.exit(f"[twin] {e.args[0]}")
    leader, follower, _, _ = system.arm_ports(cfg)
    print(f"{system.CONFIG_PATH}{'' if system.CONFIG_PATH.exists() else '  (not written yet: defaults)'}")
    for k, v in cfg.items():
        shown = v or {"leader_port": f"(guess: {leader})", "follower_port": f"(guess: {follower})"}.get(
            k, "(unset)"
        )
        print(f"  {k:<14} {shown}")


def cmd_doctor(rest: list[str]) -> None:
    import platform

    ok = True

    def line(good: bool, what: str, detail: str = "") -> None:
        nonlocal ok
        ok &= good
        print(f"  [{'ok' if good else '!!'}] {what:<14} {detail}")

    print(f"Twin Depot doctor - {system.OS_NAME} {platform.release()} {platform.machine()}")
    line(True, "python", f"{sys.version.split()[0]}  {sys.executable}")
    try:
        import mujoco

        line(True, "mujoco", mujoco.__version__)
    except ImportError as e:
        line(False, "mujoco", f"{e} - run `twin setup`")
    try:
        import lerobot  # noqa: F401

        line(True, "lerobot", "installed (real arms supported)")
    except ImportError:
        line(False, "lerobot", "missing - real arms need `twin setup` (without --demo-only)")
    if system.IS_MAC:
        mj = system.mjpython_path()
        line(bool(mj), "mjpython", mj or "missing - the viewer needs it (pip install mujoco)")
    if not rest or "--no-window" not in rest:
        line(*_probe_window())
    from twin_depot.isaac_scene import available

    line(
        True, "isaac hall", "imported" if available() else "not imported (built-in hall; `twin import-isaac`)"
    )
    leader, follower, configured, notes = system.arm_ports()
    line(
        configured,
        "arm ports",
        f"leader={leader} follower={follower}" + ("" if configured else "  (guess: `twin ports`)"),
    )
    for note in notes:
        line(False, "arm ports", note)
    try:
        from twin_depot.app import _usb_ports

        usb = sorted(_usb_ports())
        line(bool(usb), "usb serial", ", ".join(usb) or "none (arms unplugged? fine for the demo)")
    except SystemExit as e:
        line(False, "usb serial", str(e))
    for port in (leader, follower):
        hint = system.serial_permission_hint(port)
        if hint:
            line(False, "permission", hint)
    if system.IS_WSL:
        print("  note: WSL renders slowly and needs usbipd for USB arms - native Windows/Linux is smoother")
    print("all good" if ok else "some checks need attention (demo still works without arms)")


def _probe_window() -> tuple[bool, str, str]:
    """Open and close an invisible GLFW window: the viewer needs a working OpenGL desktop session."""
    code = (
        "import glfw\n"
        "ok = glfw.init()\n"
        "if ok:\n"
        "    glfw.window_hint(glfw.VISIBLE, 0)\n"
        "    ok = bool(glfw.create_window(64, 64, 'probe', None, None))\n"
        "    glfw.terminate()\n"
        "raise SystemExit(0 if ok else 1)\n"
    )
    try:
        rc = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=30).returncode
    except subprocess.TimeoutExpired:
        return False, "opengl window", "probe timed out"
    if rc == 0:
        return True, "opengl window", "ok"
    return (
        False,
        "opengl window",
        "can't open one (no desktop/display?) - use --headless or a desktop session",
    )


def cmd_tests(rest: list[str]) -> None:
    raise SystemExit(run([sys.executable, "-m", "pytest", SIM / "tests", "-q", *rest], check=False))


def cmd_stream(rest: list[str]) -> None:
    run([sys.executable, SIM / "stream_teleop.py", *rest])


# ------------------------------------------------------------------------------ entry
def usage() -> None:
    print((__doc__ or "").split("\n\n")[0])
    print("\ncommands:")
    for name, text in COMMANDS.items():
        print(f"  {name:<13} {text}")
    print("\n`twin demo --help` lists every game option (difficulty, shift, hall, ...).")


def menu() -> list[str] | None:
    print("Twin Depot - SO-101 digital-twin sorting\n")
    for i, name in enumerate(MENU, 1):
        print(f"  {i}. {name:<10} {COMMANDS[name]}")
    print("  q. quit")
    choice = input("\nchoose: ").strip().lower()
    if choice.isdigit() and 1 <= int(choice) <= len(MENU):
        return [MENU[int(choice) - 1]]
    return [choice] if choice in COMMANDS else None


def main(argv: list[str]) -> None:
    from_menu = not argv
    if from_menu:
        argv = menu() or []
        if not argv:
            return
    cmd, rest = argv[0], argv[1:]
    try:
        if cmd in ("-h", "--help", "help"):
            usage()
        elif cmd == "setup":
            cmd_setup(rest)
        elif cmd == "config":
            cmd_config(rest)
        elif cmd in COMMANDS:
            enter_env(argv)
            handler = {
                "ports": cmd_ports,
                "calibrate": cmd_calibrate,
                "stream": cmd_stream,
                "import-isaac": cmd_import_isaac,
                "tests": cmd_tests,
                "doctor": cmd_doctor,
            }.get(cmd)
            handler(rest) if handler else cmd_play(cmd, rest)
        else:
            usage()
            raise SystemExit(f"\n[twin] unknown command {cmd!r}")
    finally:
        if from_menu:  # double-clicked: keep the window open to show the result
            with contextlib.suppress(EOFError):
                input("\npress Enter to close ")


if __name__ == "__main__":
    main(sys.argv[1:])
