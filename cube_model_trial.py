"""Train three full-frame cube detectors on the same rendered scenes; keep live weights separate."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from twin_depot.cuboid_pnp import CENTRED, SYMMETRIES

ROOT = Path(__file__).resolve().parent
TRIAL = ROOT / "data/vision/model-trial"
NAMES = ["red", "green", "blue", "yellow"]
FACES = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
SIZE = 384
REFERENCE = cv2.Rodrigues(np.array([0.41, -0.33, 0.13]))[0]
MODELS = ("detect", "pose", "vit")


def backgrounds():
    desk = cv2.imread(str(ROOT / "data/vision/auto-check.jpg"))
    if desk is None:
        raise ValueError("Missing separate desk photo data/vision/auto-check.jpg")
    desk = cv2.resize(desk, (432, 768))
    mask = np.zeros(desk.shape[:2], np.uint8)
    mask[165:307, 174:249] = 255  # remove both real cubes from this background
    clean = cv2.inpaint(desk, mask, 5, cv2.INPAINT_TELEA)
    hand = desk[300:415, 243:370]
    hsv = cv2.cvtColor(hand, cv2.COLOR_BGR2HSV)
    skin = cv2.inRange(hsv, (0, 32, 43), (23, 210, 255))
    return clean, hand, skin


def render(rng, desk, hand, skin):
    h, w = desk.shape[:2]
    extent = int(rng.integers(145, w))
    x = int(rng.integers(0, w - extent + 1))
    y = int(rng.integers(0, h - extent + 1))
    im = cv2.resize(desk[y : y + extent, x : x + extent], (SIZE, SIZE))
    im = np.rot90(im, int(rng.integers(4))).copy()
    if rng.random() < 0.5:
        im = cv2.flip(im, 1)
    # Negative shapes around the scene include brightly printed labels and tape.
    for _ in range(rng.integers(4)):
        a = rng.integers(0, SIZE, 2)
        b = a + rng.integers([8, 3], [80, 22])
        color = rng.integers(20, 210, 3).tolist()
        cv2.rectangle(im, a, b, color, -1)
        cv2.line(im, a + 2, b - 2, (190, 190, 180), 1)
    f = float(rng.uniform(340, 510))
    k = np.array([[f, 0, SIZE / 2], [0, f, SIZE / 2], [0, 0, 1.0]])
    objects = []
    owners = np.zeros((SIZE, SIZE), np.uint8)
    areas = []
    for _ in range(rng.choice([0, 1, 1, 2, 2, 2, 3, 4])):
        cls = int(rng.integers(4))
        uv = rng.uniform(42, SIZE - 42, 2)
        z = rng.uniform(200, 650)
        t = np.r_[(uv - SIZE / 2) * z / f, z]
        axis = rng.normal(size=3)
        rotation = cv2.Rodrigues(axis / np.linalg.norm(axis) * rng.uniform(0, np.pi))[0]
        # All 24 physical cube symmetries produce the same image. Teach a
        # consistent corner order to the standard ordered YOLO keypoint head.
        rot = max((rotation @ s for s in SYMMETRIES), key=lambda r: np.trace(REFERENCE.T @ r))
        xyz = CENTRED @ rot.T + t
        pts = cv2.projectPoints(CENTRED, cv2.Rodrigues(rot)[0], t, k, None)[0].reshape(8, 2)
        lo, hi = pts.min(0), pts.max(0)
        if (
            np.any(lo < 2)
            or np.any(hi > SIZE - 2)
            or any(np.all(np.minimum(hi, o["hi"]) - np.maximum(lo, o["lo"]) > 0) for o in objects)
        ):
            continue
        hue = int(([0, 63, 114, 28][cls] + rng.integers(-6, 7)) % 180)
        hsv = np.uint8([[[hue, rng.integers(155, 256), rng.integers(100, 250)]]])
        base = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        for face in sorted(FACES, key=lambda ix: -xyz[ix, 2].mean()):
            shade = rng.uniform(0.38, 1.06)
            cv2.fillConvexPoly(
                im, np.rint(pts[face]).astype(np.int32), np.clip(base * shade, 0, 255).tolist(), cv2.LINE_AA
            )
        cube_mask = np.zeros((SIZE, SIZE), np.uint8)
        cv2.fillConvexPoly(cube_mask, cv2.convexHull(pts.astype(np.int32)), 1)
        areas.append(int(cube_mask.sum()))
        owners[cube_mask != 0] = len(areas)
        if rng.random() < 0.65:
            # Cover a portion of the object with the edge of a real skin patch.
            side = int(max(hi - lo) * rng.uniform(0.7, 1.8))
            patch = cv2.resize(hand, (side, side))
            alpha = cv2.resize(skin, (side, side))
            angle = rng.uniform(-np.pi, np.pi)
            center = uv + rng.uniform(0.44, 0.91) * side * np.array([np.cos(angle), np.sin(angle)])
            origin = np.int32(center - side / 2)
            x, y = np.maximum(0, origin)
            x1, y1 = np.minimum(SIZE, origin + side)
            if x1 > x and y1 > y:
                i, j = np.array([x, y]) - origin
                use = alpha[j : j + y1 - y, i : i + x1 - x] > 127
                # Preserve some observable pixels; annotations cover the full cube.
                covered = (cube_mask[y:y1, x:x1] * use).sum()
                if covered < 0.68 * areas[-1]:
                    im[y:y1, x:x1][use] = patch[j : j + y1 - y, i : i + x1 - x][use]
                    owners[y:y1, x:x1][use] = 0
        objects.append({"cls": cls, "lo": lo, "hi": hi, "pts": pts, "xyz": t, "rotation": rot})
    visible = np.bincount(owners.ravel(), minlength=len(objects) + 1)
    for i, obj in enumerate(objects):
        obj["occlusion"] = 1 - visible[i + 1] / areas[i]
    im = cv2.convertScaleAbs(im, alpha=rng.uniform(0.69, 1.17), beta=rng.uniform(-7, 8))
    if rng.random() < 0.45:
        im = cv2.GaussianBlur(im, (3, 3), rng.uniform(0.5, 1.25))
    return im, objects, k


def prepare():
    desk, hand, skin = backgrounds()
    for split, count, seed in [("train", 864, 901), ("val", 128, 902), ("test", 160, 903)]:
        rng = np.random.default_rng(seed)
        truth = []
        for task in ("pose", "detect"):
            for sub in ("images", "labels"):
                (TRIAL / task / sub / split).mkdir(parents=True, exist_ok=True)
        for index in range(count):
            im, objects, k = render(rng, desk, hand, skin)
            ok, jpg = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(65, 94))])
            if not ok:
                raise ValueError("JPEG encoding failed")
            label_rows = {"pose": [], "detect": []}
            for obj in objects:
                lo, hi = obj["lo"], obj["hi"]
                bbox = np.r_[(lo + hi) / 2, hi - lo] / SIZE
                prefix = str(obj["cls"]) + " " + " ".join(f"{v:.6f}" for v in bbox)
                label_rows["detect"].append(prefix)
                # kpt_shape [8,2] supervises every amodal projected corner.
                label_rows["pose"].append(
                    prefix + " " + " ".join(f"{v:.6f}" for v in obj["pts"].ravel() / SIZE)
                )
            for task, rows in label_rows.items():
                ident = f"{index:04}"
                (TRIAL / task / "images" / split / (ident + ".jpg")).write_bytes(jpg)
                (TRIAL / task / "labels" / split / (ident + ".txt")).write_text(
                    "\n".join(rows), encoding="ascii"
                )
            if split == "test":
                truth.append(
                    {
                        "k": k.tolist(),
                        "objects": [
                            {
                                "color": NAMES[o["cls"]],
                                "xyz": o["xyz"].tolist(),
                                "uv": ((o["lo"] + o["hi"]) / 2).tolist(),
                            }
                            for o in objects
                        ],
                    }
                )
        if truth:
            (TRIAL / "test-truth.json").write_text(json.dumps(truth))
    for task in ("detect", "pose"):
        config = {
            "path": str(TRIAL / task),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": NAMES,
        }
        if task == "pose":
            config.update(kpt_shape=[8, 2])
        (TRIAL / task / "data.yaml").write_text(yaml.safe_dump(config), encoding="utf8")


def train(kind, epochs):
    import torch
    import ultralytics
    from ultralytics import YOLO

    torch.set_num_threads(3)
    task = "detect" if kind == "detect" else "pose"
    weights = TRIAL / ("yolo11n.pt" if task == "detect" else "yolo11n-pose.pt")
    if not weights.exists():
        from ultralytics.utils.downloads import attempt_download_asset

        attempt_download_asset(str(weights))
    if kind == "vit":
        base = Path(ultralytics.__file__).parent / "cfg/models/11/yolo11-pose.yaml"
        arch = yaml.safe_load(base.read_text(encoding="utf8"))
        arch["scale"] = "n"
        # C3TR applies multi-head self-attention and an MLP to spatial tokens.
        # Replace the final backbone block in-place to preserve neck indices.
        arch["backbone"][-1] = [-1, 2, "C3TR", [1024]]
        archpath = TRIAL / "yolo11n-vit-pose.yaml"
        archpath.write_text(yaml.safe_dump(arch))
        model = YOLO(str(archpath)).load(str(weights))
    else:
        model = YOLO(str(weights))
    model.train(
        data=str(TRIAL / task / "data.yaml"),
        epochs=epochs,
        imgsz=SIZE,
        batch=32,
        workers=0,
        device=0,
        seed=819,
        amp=False,
        optimizer="AdamW",
        lr0=0.0015,
        warmup_epochs=1,
        cos_lr=True,
        fliplr=0,
        flipud=0,
        degrees=0,
        mosaic=0.25,
        close_mosaic=3,
        hsv_h=0.005,
        hsv_s=0.18,
        hsv_v=0.18,
        plots=False,
        save=True,
        cache="ram",
        project=str(TRIAL / "runs"),
        name=kind,
        exist_ok=True,
        verbose=False,
        patience=epochs,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", *MODELS))
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    else:
        train(args.action, args.epochs)
