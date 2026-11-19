import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from twin_depot.cube_pose import FACE, face_pose
from twin_depot.robot_view import reconstruct


def test_lift_is_recovered_independently_of_phone_pose():
    k=np.array([[780.,0,360],[0,780,640],[0,0,1]])
    ref=np.eye(4);ref[2,3]=29
    for tilt in (.25,.48):
        camera=np.eye(4)
        camera[:3,:3]=cv2.Rodrigues(np.array([np.pi-tilt,0.,0.]))[0]
        camera[:3,3]=[20,30,450]
        for lift in (0.,38.):
            target=ref.copy();target[:3,3]+=[65,-10,lift]
            fits=[]
            for name,obj in [('red',ref),('green',target)]:
                t=camera@obj
                pix=cv2.projectPoints(FACE,cv2.Rodrigues(t[:3,:3])[0],t[:3,3],k,None)[0].reshape(4,2)
                fits.append((name,*face_pose(pix,k,None)))
            phone,cubes=reconstruct(fits,np.eye(4))
            assert cubes[1][1][:3,3]==pytest.approx([65,-10,14.5+lift],abs=1e-4)
            assert phone[:3,3]==pytest.approx(np.linalg.inv(camera)[:3,3],abs=1e-4)
