"""Interactive SO-101 viewer for registered observations and camera-relative previews."""

import json
import os
import sys
import threading
import time
import xml.etree.ElementTree as ET
from urllib.request import Request, urlopen

import mujoco
import mujoco.viewer
import numpy as np

from .workcell_link import COLORS, observation_poses
from .assets import ensure_so101_assets


def make_model():
    xml = ensure_so101_assets()
    root = ET.parse(xml).getroot()
    root.find("compiler").set("meshdir", str(xml.parent / "assets"))
    world = root.find("worldbody")
    for i in range(13):
        body = ET.SubElement(world, "body", name=f"observation{i}", mocap="true", pos="0 0 -3")
        ET.SubElement(
            body,
            "geom",
            name=f"marker{i}",
            type="box",
            size=".035 .025 .009" if i == 0 else ".0145 .0145 .0145",
            contype="0",
            conaffinity="0",
            rgba="0.3 0.6 0.8 0.8",
        )
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def place(model, data, state):
    data.mocap_pos[:] = [0, 0, -3]
    poses, label = observation_poses(state)
    if not poses:
        return label
    for i, (color, pose) in enumerate(poses[:13]):
        idx = model.body(f"observation{i}").mocapid[0]
        data.mocap_pos[idx] = pose[:3, 3] / 1000
        mujoco.mju_mat2Quat(data.mocap_quat[idx], np.ascontiguousarray(pose[:3, :3]).ravel())
        if color:
            model.geom(f"marker{i}").rgba[:] = [*COLORS[color], 0.8]
    return f"{label} | Cubes: {min(len(poses) - 1, 12)}"


def main(port):
    model = make_model()
    data = mujoco.MjData(model)
    request = Request(
        f"http://127.0.0.1:{port}/sim-state", headers={"X-Pair": os.environ.pop("TWIN_SIM_PAIR")}
    )
    stop = threading.Event()
    latest = {}

    def read():
        nonlocal latest
        while not stop.is_set():
            try:
                with urlopen(request, timeout=0.5) as reply:
                    latest = json.load(reply)
            except (OSError, ValueError):
                latest = {}
            stop.wait(0.05)

    threading.Thread(target=read, daemon=True).start()
    try:
        with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=True) as viewer:
            viewer.cam.lookat[:] = [0.09, 0, 0.15]
            viewer.cam.distance = 0.85
            viewer.cam.azimuth = 132
            viewer.cam.elevation = -22
            previous_label = None
            while viewer.is_running():
                start = time.monotonic()
                with viewer.lock():
                    label = place(model, data, latest)
                    mujoco.mj_step(model, data, nstep=max(1, round(1 / 60 / model.opt.timestep)))
                viewer.sync()
                if label != previous_label:
                    viewer.set_texts((None, None, label, None))
                    previous_label = label
                stop.wait(max(0, 1 / 60 - (time.monotonic() - start)))
    finally:
        stop.set()


if __name__ == "__main__":
    main(int(sys.argv[1]))
