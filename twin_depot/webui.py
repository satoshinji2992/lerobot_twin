"""Local web control panel: `twin ui` (or double-click twin.bat / twin.command, or ./twin.sh).

A stdlib HTTP server on 127.0.0.1 serves webui.html and runs the same twin.py commands as the
terminal, one job at a time: its output streams to the page, the page's input box feeds its stdin
(LeRobot calibration prompts, in-game commands), and a running shift reports its score through a
JSON status file. Every API call must carry the random per-session token baked into the page, so
other websites open in the same browser can't drive the arms.
"""

from __future__ import annotations

import codecs
import contextlib
import json
import os
import secrets
import signal
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import system

HERE = Path(__file__).resolve().parent
PAGE = HERE / "webui.html"
PLAY_SOURCES = ("demo", "real", "leader", "udp", "replay")
JOBS = (*PLAY_SOURCES, "setup", "calibrate", "doctor", "tests", "import-isaac", "stream")
LOG_LIMIT = 400_000  # characters kept for the page
PORTS_SNIPPET = (
    "import json\n"
    "from serial.tools import list_ports\n"
    "print(json.dumps([{'device': p.device, 'serial': p.serial_number or '', 'description': p.description,"
    " 'usb': p.hwid.upper().startswith('USB')} for p in list_ports.comports()]))\n"
)


class Job:
    """One child process with captured output and a writable stdin."""

    def __init__(self, name: str, argv: list[str], env: dict, status_file: Path | None):
        self.name, self.argv, self.status_file = name, argv, status_file
        self.started = time.time()
        self.text = ""
        self.offset = 0  # characters dropped from the front of self.text
        self.lock = threading.Lock()
        kw: dict = {}
        if system.IS_WINDOWS:
            kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kw["start_new_session"] = True  # own process group: stop() reaches grandchildren too
        self.proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(system.SIM_DIR),
            **kw,
        )
        self.append("$ " + " ".join(Path(a).name if i == 0 else a for i, a in enumerate(argv)) + "\n")
        threading.Thread(target=self._pump, daemon=True).start()

    def append(self, s: str) -> None:
        with self.lock:
            self.text += s
            if len(self.text) > LOG_LIMIT:
                cut = len(self.text) - LOG_LIMIT
                self.text, self.offset = self.text[cut:], self.offset + cut

    def _pump(self) -> None:
        dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
        out = self.proc.stdout
        while True:
            chunk = out.read1(4096) if hasattr(out, "read1") else out.read(1)
            if not chunk:
                break
            self.append(dec.decode(chunk).replace("\r\n", "\n"))
        self.proc.wait()
        self.append(
            dec.decode(b"", final=True) + f"\n[twin] {self.name} finished (exit {self.proc.returncode})\n"
        )

    def read(self, since: int) -> tuple[str, int]:
        with self.lock:
            start = max(0, since - self.offset)
            return self.text[start:], self.offset + len(self.text)

    @property
    def running(self) -> bool:
        return self.proc.poll() is None

    def send(self, line: str) -> None:
        if self.running and self.proc.stdin:
            try:
                self.proc.stdin.write((line + "\n").encode("utf-8"))
                self.proc.stdin.flush()
            except OSError:
                pass

    def stop(self) -> None:
        if not self.running:
            return
        if self.name in PLAY_SOURCES:  # let the game shut the source down cleanly first
            self.send("q")
            for _ in range(30):
                if not self.running:
                    return
                time.sleep(0.1)
        if system.IS_WINDOWS:
            subprocess.call(
                ["taskkill", "/T", "/F", "/PID", str(self.proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(timeout=3)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(self.proc.pid, signal.SIGKILL)


class Panel:
    def __init__(self, twin_script: Path, env_python: Callable[[], str | None]):
        self.twin_script = twin_script
        self.env_python = env_python
        self.token = secrets.token_urlsafe(18)
        self.job: Job | None = None
        self.status_file = Path(tempfile.gettempdir()) / f"twin-depot-status-{os.getpid()}.json"
        self.lock = threading.Lock()

    # ---------------------------------------------------------------- queries
    def state(self) -> dict:
        cfg = system.load_config()
        ports = system.arm_ports(cfg)
        py = self.env_python()
        try:
            from .isaac_scene import available

            isaac = available()
        except Exception:  # numpy missing before setup
            isaac = (system.SIM_DIR / "assets" / "isaac" / "full_warehouse" / "manifest.json").exists()
        job = None
        if self.job:
            job = {
                "name": self.job.name,
                "running": self.job.running,
                "rc": self.job.proc.returncode,
                "started": self.job.started,
            }
        status = None
        if self.job and self.job.name in PLAY_SOURCES and self.status_file.exists():
            try:
                status = json.loads(self.status_file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                status = None
        return {
            "os": system.OS_NAME,
            "env_python": py,
            "config": cfg,
            "ports": {"leader": ports.leader, "follower": ports.follower, "configured": ports.configured},
            "port_notes": ports.notes,
            "isaac": isaac,
            "job": job,
            "status": status,
            "config_path": str(system.CONFIG_PATH),
        }

    def serial_ports(self) -> dict:
        py = self.env_python()
        if py is None:
            return {"ok": False, "error": "先在“工具”里运行一次“安装环境”"}
        try:
            out = subprocess.run([py, "-c", PORTS_SNIPPET], capture_output=True, text=True, timeout=20)
            ports = json.loads(out.stdout)
        except (OSError, ValueError, subprocess.TimeoutExpired) as e:
            return {"ok": False, "error": f"无法列出串口：{e}"}
        for p in ports:
            p["hint"] = system.serial_permission_hint(p["device"])
        return {"ok": True, "ports": ports}

    # ---------------------------------------------------------------- actions
    def run(self, cmd: str, args: list[str]) -> dict:
        if cmd not in JOBS:
            return {"ok": False, "error": f"unknown command {cmd}"}
        if not all(isinstance(a, str) for a in args):
            return {"ok": False, "error": "bad arguments"}
        with self.lock:
            if self.job and self.job.running:
                return {"ok": False, "error": f"“{self.job.name}”还在运行，先停止它"}
            env = {**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"}
            if cmd == "setup":
                py = sys.executable  # any Python can bootstrap; twin.py picks or makes the env
                env.pop("TWIN_IN_ENV", None)
            else:
                py = self.env_python()
                if py is None:
                    return {"ok": False, "error": "还没有环境：先运行“安装环境”"}
                env["TWIN_IN_ENV"] = "1"  # run directly in the env: one process, clean stop
            status = None
            if cmd in PLAY_SOURCES:
                self.status_file.unlink(missing_ok=True)
                status = self.status_file
                args = [*args, "--stdin", "--status-file", str(status)]
            self.job = Job(cmd, [py, "-u", str(self.twin_script), cmd, *args], env, status)
        return {"ok": True}

    def send(self, line: str) -> dict:
        if not (self.job and self.job.running):
            return {"ok": False, "error": "没有正在运行的任务"}
        self.job.send(line)
        self.job.append(f"> {line}\n")
        return {"ok": True}

    def stop(self) -> dict:
        if self.job:
            threading.Thread(target=self.job.stop, daemon=True).start()
        return {"ok": True}

    def save_config(self, updates: dict) -> dict:
        try:
            cfg = system.save_config({str(k): str(v).strip() for k, v in updates.items()})
        except KeyError as e:
            return {"ok": False, "error": e.args[0]}
        return {"ok": True, "config": cfg}

    def shutdown_jobs(self) -> None:
        if self.job and self.job.running:
            self.job.stop()
        self.status_file.unlink(missing_ok=True)


def _handler(panel: Panel, stop_server: Callable[[], None]):
    class Handler(BaseHTTPRequestHandler):
        server_version = "TwinDepot"

        def log_message(self, format, *args):  # noqa: A002 - keep the console quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(
                code, json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8"), "application/json"
            )

        def _authorized(self) -> bool:
            if secrets.compare_digest(self.headers.get("X-Twin-Token", ""), panel.token):
                return True
            self._json({"ok": False, "error": "forbidden"}, 403)
            return False

        def do_GET(self):  # noqa: N802
            url = urlparse(self.path)
            if url.path in ("/", "/index.html"):
                html = PAGE.read_text(encoding="utf-8").replace("__TWIN_TOKEN__", panel.token)
                return self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
            if not url.path.startswith("/api/") or not self._authorized():
                return None if url.path.startswith("/api/") else self._send(404, b"not found", "text/plain")
            if url.path == "/api/state":
                return self._json(panel.state())
            if url.path == "/api/log":
                since = int(parse_qs(url.query).get("since", ["0"])[0] or 0)
                if not panel.job:
                    return self._json({"text": "", "next": 0, "reset": since > 0})
                text, nxt = panel.job.read(since)
                return self._json({"text": text, "next": nxt, "reset": since > nxt})
            if url.path == "/api/ports":
                return self._json(panel.serial_ports())
            return self._json({"ok": False, "error": "not found"}, 404)

        def do_POST(self):  # noqa: N802
            if not self._authorized():
                return
            try:
                n = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                return self._json({"ok": False, "error": "bad json"}, 400)
            path = urlparse(self.path).path
            if path == "/api/run":
                return self._json(panel.run(str(body.get("cmd", "")), list(body.get("args", []))))
            if path == "/api/input":
                return self._json(panel.send(str(body.get("line", ""))))
            if path == "/api/stop":
                return self._json(panel.stop())
            if path == "/api/config":
                return self._json(panel.save_config(dict(body.get("updates", {}))))
            if path == "/api/shutdown":
                self._json({"ok": True})
                threading.Thread(target=stop_server, daemon=True).start()
                return None
            return self._json({"ok": False, "error": "not found"}, 404)

    return Handler


def make_server(twin_script: Path, env_python: Callable[[], str | None], port: int = 8765):
    """Bind 127.0.0.1 (the first free port from `port`); returns (server, panel, url)."""
    panel = Panel(twin_script, env_python)
    server = None
    for p in [port, *range(port + 1, port + 20), 0] if port else [0]:
        try:
            server = ThreadingHTTPServer(("127.0.0.1", p), BaseHTTPRequestHandler)  # real handler below
            break
        except OSError:
            continue
    if server is None:
        raise SystemExit("[twin] no free local port for the web panel")
    server.RequestHandlerClass = _handler(panel, server.shutdown)
    server.daemon_threads = True
    return server, panel, f"http://127.0.0.1:{server.server_address[1]}/"


def serve(
    twin_script: Path, env_python: Callable[[], str | None], port: int = 8765, open_browser: bool = True
):
    server, panel, url = make_server(twin_script, env_python, port)
    print(f"[twin] control panel: {url}", flush=True)
    print(
        "[twin] keep this window open; close it (or press Ctrl+C, or 关闭控制台 on the page) to quit",
        flush=True,
    )
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.3)
    except KeyboardInterrupt:
        pass
    finally:
        panel.shutdown_jobs()
        server.server_close()
        print("[twin] control panel closed", flush=True)
