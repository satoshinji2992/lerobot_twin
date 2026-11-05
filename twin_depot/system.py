"""Operating-system glue shared by twin.py, play.py and stream_teleop.py (stdlib only).

Everything that differs between Windows, macOS and Linux lives here: serial-port naming, the
per-machine settings file, and macOS's rule that MuJoCo's viewer must run under ``mjpython``.
"""

from __future__ import annotations

import configparser
import glob
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import NamedTuple

IS_WINDOWS = os.name == "nt"
IS_MAC = sys.platform == "darwin"
IS_LINUX = sys.platform.startswith("linux")
IS_WSL = IS_LINUX and "microsoft" in platform.release().lower()
OS_NAME = "Windows" if IS_WINDOWS else "macOS" if IS_MAC else "WSL" if IS_WSL else "Linux"

SIM_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = SIM_DIR / "config.ini"  # per machine, git-ignored; written by `twin ports` / `twin config`
SECTION = "twin"
DEFAULTS = {
    "leader_port": "",  # empty = guess (see guess_ports); `twin ports` fills these in
    "follower_port": "",
    "leader_serial": "",  # USB serial number of each servo board: finds the arm again after its
    "follower_serial": "",  # port name changes (other USB socket, new COM number, reboot)
    "leader_id": "my_awesome_leader_arm",
    "follower_id": "my_awesome_follower_arm",
    "arm": "so101",
    "player": "",
    "conda_env": "twin-depot",  # used by `twin setup` when conda is available
}


def load_config() -> dict[str, str]:
    cfg = dict(DEFAULTS)
    if CONFIG_PATH.exists():
        parser = configparser.ConfigParser()
        parser.read(CONFIG_PATH, encoding="utf-8")
        if parser.has_section(SECTION):
            cfg.update({k: v.strip() for k, v in parser.items(SECTION)})
    return cfg


def save_config(updates: dict[str, str]) -> dict[str, str]:
    unknown = set(updates) - set(DEFAULTS)
    if unknown:
        raise KeyError(f"unknown setting(s): {', '.join(sorted(unknown))} (known: {', '.join(DEFAULTS)})")
    cfg = load_config()
    cfg.update(updates)
    parser = configparser.ConfigParser()
    parser[SECTION] = {k: cfg[k] for k in DEFAULTS}
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        f.write("# Twin Depot settings for this machine. `twin ports` finds the arms' serial ports.\n")
        parser.write(f)
    return cfg


def guess_ports() -> tuple[str, str]:
    """Best guess at (leader, follower) when config.ini has no ports. `twin ports` makes it certain."""
    if IS_WINDOWS:
        return "COM3", "COM4"
    if IS_MAC:  # the servo boards' CH343/CH340 show up as usbmodem / usbserial devices
        found = sorted(glob.glob("/dev/tty.usbmodem*") + glob.glob("/dev/tty.usbserial*"))
        fallback = ["/dev/tty.usbmodem1", "/dev/tty.usbmodem2"]
    else:
        found = sorted(glob.glob("/dev/ttyACM*")) + sorted(glob.glob("/dev/ttyUSB*"))
        fallback = ["/dev/ttyACM0", "/dev/ttyACM1"]
    found += [f for f in fallback if f not in found]
    return found[0], found[1]


class ArmPorts(NamedTuple):
    leader: str
    follower: str
    configured: bool  # False when either port is only a guess
    notes: list[str]  # e.g. "leader board moved COM5 -> COM7"


def list_usb_serial() -> list | None:
    """pyserial's USB serial ports, or None when pyserial isn't installed."""
    try:
        from serial.tools import list_ports
    except ImportError:
        return None
    return [p for p in list_ports.comports() if p.hwid.upper().startswith("USB")]


def resolve_ports(cfg: dict[str, str], present: list | None) -> ArmPorts:
    """Saved serial number -> wherever that board is plugged in now; else the saved name; else a guess."""
    guess = guess_ports()
    by_serial: dict[str, list[str]] = {}
    for p in present or []:
        if p.serial_number:
            by_serial.setdefault(p.serial_number, []).append(p.device)
    ports, notes = {}, []
    for role, fallback in (("leader", guess[0]), ("follower", guess[1])):
        saved, serial = cfg[f"{role}_port"], cfg[f"{role}_serial"]
        port = saved or fallback
        if serial and present is not None:
            found = by_serial.get(serial, [])
            if len(found) == 1:
                port = found[0]
                if saved and port != saved:
                    notes.append(f"{role} board (serial {serial}) moved {saved} -> {port}")
            else:
                notes.append(f"{role} board (serial {serial}) not plugged in - trying {port}")
        ports[role] = port
    configured = all(cfg[f"{r}_port"] or cfg[f"{r}_serial"] for r in ("leader", "follower"))
    return ArmPorts(ports["leader"], ports["follower"], configured, notes)


def arm_ports(cfg: dict[str, str] | None = None) -> ArmPorts:
    cfg = cfg or load_config()
    present = list_usb_serial() if (cfg["leader_serial"] or cfg["follower_serial"]) else None
    return resolve_ports(cfg, present)


def serial_permission_hint(port: str) -> str | None:
    """Linux: the servo boards need membership of the group that owns /dev/ttyACM* (dialout or uucp)."""
    if IS_WINDOWS or IS_MAC or not os.path.exists(port) or os.access(port, os.R_OK | os.W_OK):
        return None
    try:
        import grp

        group = grp.getgrgid(os.stat(port).st_gid).gr_name
    except (ImportError, KeyError, OSError):
        group = "dialout"
    return f"no permission for {port}: run `sudo usermod -aG {group} $USER`, then log out and back in"


def mjpython_path() -> str | None:
    exe = Path(sys.executable).with_name("mjpython")
    return str(exe) if exe.exists() else shutil.which("mjpython")


def ensure_viewer_process() -> None:
    """macOS: MuJoCo's interactive viewer only works under ``mjpython``; re-launch under it if needed."""
    if not IS_MAC:
        return
    import mujoco.viewer

    if getattr(mujoco.viewer, "_MJPYTHON", None) is not None:
        return
    if os.environ.get("TWIN_MJPYTHON"):
        raise SystemExit("[twin] started under mjpython but the MuJoCo viewer is still unavailable")
    exe = mjpython_path()
    if not exe:
        raise SystemExit(
            "[twin] macOS needs `mjpython` (installed by `pip install mujoco`) to open the viewer; "
            "or use --headless"
        )
    print("[twin] macOS: re-launching under mjpython for the MuJoCo viewer ...", flush=True)
    os.environ["TWIN_MJPYTHON"] = "1"
    sys.stdout.flush()
    os.execv(exe, [exe, *sys.orig_argv[1:]])
