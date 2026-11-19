import mujoco
import numpy as np

from twin_depot.arm_viewer import place


def test_track_visible_without_registration_and_transformed_once_with_registration():
    bodies = "".join(
        f'<body name="observation{i}" mocap="true"><geom name="marker{i}" size=".01"/></body>'
        for i in range(3)
    )
    model = mujoco.MjModel.from_xml_string(f"<mujoco><worldbody>{bodies}</worldbody></mujoco>")
    data = mujoco.MjData(model)
    pose = np.eye(4)
    pose[:3, 3] = [15, 25, 300]
    track = dict(color="green", camera_pose=pose.tolist())
    state = dict(fresh=True, frame_mode="camera", pose_tracks=[track], completions=[track])
    place(model, data, state)
    np.testing.assert_allclose(data.mocap_pos[1], [.215, .08, .155])
    assert data.mocap_pos[2, 2] == -3

    base_from_camera = np.eye(4)
    base_from_camera[0, 3] = 100
    state.update(frame_mode="workcell", registered=True, camera_pose=base_from_camera.tolist())
    place(model, data, state)
    np.testing.assert_allclose(data.mocap_pos[1], [.115, .025, .3])
    place(model, data, {**state, "fresh": False})
    assert data.mocap_pos[1, 2] == -3
