"""Calibrate the exact video stream used by robot-view from multi-view checkerboard stills."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("images",type=Path)
    p.add_argument("--inner",type=int,nargs=2,required=True,metavar=("COLS","ROWS"))
    p.add_argument("--square-mm",type=float,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    if min(a.inner)<3 or a.square_mm<=0:p.error("invalid checkerboard dimensions")
    grid=np.zeros((np.prod(a.inner),3),np.float32)
    grid[:,:2]=np.mgrid[:a.inner[0],:a.inner[1]].T.reshape(-1,2)*a.square_mm
    points=[];size=None
    for path in sorted(a.images.iterdir()):
        if path.suffix.lower() not in (".jpg",".jpeg",".png"):continue
        im=cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),cv2.IMREAD_GRAYSCALE)
        if im is None:continue
        if size and size!=im.shape[::-1]:p.error("all video frames must have the same pixel dimensions")
        found,corners=cv2.findChessboardCornersSB(im,tuple(a.inner))
        if found:
            size=im.shape[::-1];points.append(corners)
    if len(points)<10:p.error(f"only {len(points)} usable views; supply at least 10 distinct board poses")
    rms,k,dist,rotations,_=cv2.calibrateCamera([grid]*len(points),points,size,None,None)
    normals=np.array([cv2.Rodrigues(r)[0][:,2] for r in rotations])
    if np.ptp(normals,axis=0)[:2].min()<.2:p.error("take more tilted views around both camera axes")
    cfg=json.loads(a.out.read_text("utf8")) if a.out.exists() else {}
    cfg["intrinsics"]={"image_size":size,"K":k.tolist(),"dist":dist.ravel().tolist(),"rms_px":rms}
    a.out.write_text(json.dumps(cfg,indent=2),"utf8")
    print(f"{len(points)} views; reprojection RMS {rms:.3f} pixels -> {a.out}")


if __name__=="__main__":main()
