"""Local discovery and display coordinates shared by the console and MuJoCo viewers."""
import json
import os
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from urllib.request import Request, urlopen

ENDPOINT = Path(os.environ.get("TWIN_WORKCELL_ENDPOINT_FILE") or
                Path(__file__).resolve().parents[1] / "data/vision/active-workcell.json")


def publish(port, token):
    ENDPOINT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=ENDPOINT.parent, delete=False) as f:
        json.dump([port, token], f)
    os.replace(f.name, ENDPOINT)


def endpoint():
    port, token = json.loads(ENDPOINT.read_text())
    if type(port) is not int or not 0 < port < 65536 or not isinstance(token, str) or not token:
        raise ValueError("Invalid local workcell endpoint")
    return f"http://127.0.0.1:{port}", token


def read_state(origin, token):
    req = Request(origin + "/sim-state", headers={"X-Pair": token})
    with urlopen(req, timeout=0.5) as response:
        return json.load(response)


@contextmanager
def live_observations():
    """Keep network reads off the simulation clock, including when the camera is offline."""
    stop = threading.Event()
    latest = ([], "")

    def read():
        nonlocal latest
        while not stop.is_set():
            try:
                state = read_state(*endpoint())
                latest = observation_poses(state)
                delay = 0.05 if state.get("fresh") else 0.3
            except (OSError, ValueError):
                latest = ([], "")
                delay = 1
            stop.wait(delay)

    worker = threading.Thread(target=read, daemon=True)
    worker.start()
    try:
        yield lambda: latest
    finally:
        stop.set()
        worker.join(timeout=1)


COLORS = {
    "red": (0.88, 0.10, 0.08),
    "green": (0.12, 0.72, 0.27),
    "blue": (0.12, 0.39, 0.85),
    "yellow": (0.95, 0.78, 0.10),
}


def observation_poses(state):
    import numpy as np

    if not state.get("fresh"):
        return [], "Waiting for a fresh camera frame"
    if state.get("frame_mode") == "camera":
        # Illustrative mount beside the arm, units mm; optical +Z points along world +Y.
        camera = np.array([[1, 0, 0, 200], [0, 0, 1, -220], [0, -1, 0, 180], [0, 0, 0, 1]])
        world_from_rows = camera
        label = "Camera-relative preview (arm base unregistered)"
    elif state.get("registered") and state.get("camera_pose") is not None:
        camera = np.asarray(state["camera_pose"])
        world_from_rows = np.eye(4)
        label = "Registered arm base"
    else:
        return [], "Waiting for base registration and reference"
    poses = [(None, camera)]
    poses.extend((r["color"], world_from_rows @ r["pose"]) for r in state.get("rows", []))
    tracks = state.get("pose_tracks", state.get("completions", []))
    poses.extend((r["color"], camera @ r["camera_pose"]) for r in tracks)
    return poses[:13], label
