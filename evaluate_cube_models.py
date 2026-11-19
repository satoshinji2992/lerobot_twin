"""Evaluate the experimental models on the same JPEGs, same labels and same GPU."""

import argparse
import json
import time

import cv2
import numpy as np
from cube_model_trial import MODELS, NAMES, ROOT, SIZE, TRIAL, backgrounds, render
from twin_depot.cuboid_pnp import SYMMETRIES
from twin_depot.cube_completion import Completion
from twin_depot.cube_observer import FastObserver
from twin_depot.cube_pose import camera_matrix
from twin_depot.cube_trial_observer import TrialObserver, load_trial

# Approximate image centres in the original 720 x 1280 iPhone recording. Only
# score all-object precision on frames where *both* physical cubes can be seen.
TWO_CUBES = {
    0: ((494, 751), (249, 833)),
    18: ((501, 758), (264, 840)),
    36: ((518, 464), (279, 857)),
    54: ((485, 788), (285, 854)),
    126: ((513, 754), (347, 801)),
    144: ((538, 637), (352, 678)),
    162: ((470, 694), (299, 738)),
    180: ((444, 745), (262, 800)),
    198: ((443, 787), (236, 780)),
    216: ((446, 809), (202, 676)),
    234: ((449, 814), (154, 656)),
    252: ((452, 816), (243, 655)),
    270: ((459, 816), (227, 679)),
    288: ((464, 816), (256, 705)),
    306: ((463, 806), (229, 826)),
    342: ((563, 655), (172, 839)),
    360: ((555, 679), (181, 849)),
    378: ((534, 651), (179, 847)),
    396: ((641, 649), (179, 829)),
    414: ((453, 720), (178, 812)),
    432: ((432, 671), (182, 786)),
}
# Same manual held-object labels used for the prior workcell tests.
HELD = {
    0: ("red", 492, 740),
    18: ("red", 492, 746),
    36: ("red", 503, 464),
    54: ("red", 474, 781),
    72: ("green", 287, 840),
    90: ("green", 322, 646),
    108: ("green", 351, 740),
    180: ("green", 281, 769),
    198: ("green", 246, 769),
    216: ("green", 205, 669),
    234: ("green", 181, 664),
    252: ("green", 240, 669),
    270: ("green", 234, 699),
    288: ("green", 246, 693),
    306: ("green", 222, 816),
    342: ("red", 574, 646),
    360: ("red", 550, 669),
    378: ("red", 544, 646),
    396: ("red", 580, 640),
}


def load_model(kind):
    if kind == "current":
        return FastObserver(Completion("cuda"))
    weights = TRIAL / "runs" / ("pose" if kind in ("rescue", "pose-raw") else kind) / "weights"
    onnx = weights / "best.onnx"
    if not onnx.exists() or onnx.stat().st_mtime < (weights / "best.pt").stat().st_mtime:
        import torch
        from ultralytics import YOLO

        torch.set_num_threads(2)
        YOLO(str(weights / "best.pt")).export(
            format="onnx", imgsz=SIZE, opset=17, simplify=False, device="cpu"
        )
    if kind in ("pose", "pose-raw"):
        return TrialObserver(onnx, "cuda", verify_pose=kind == "pose")
    return load_trial(kind, "cuda")


def targets(observer, image, k, dist, at):
    _, partial, _ = observer.observe(image, None, k, dist, at)
    pose = observer.estimates
    boxes = observer.boxes if isinstance(observer, TrialObserver) else [*pose, *partial]

    def rectangle_center(item):
        x, y, w, h = item["box"]
        return [x + w / 2, y + h / 2]

    return (
        [{"color": o["color"], "uv": rectangle_center(o), "box": o["box"]} for o in boxes],
        [
            {
                "color": o["color"],
                "uv": np.asarray(o["pixels"]).mean(0).tolist(),
                "xyz": o["camera_xyz"],
                "pixels": o["pixels"],
            }
            for o in pose
        ],
    )


def match(predictions, truth, tolerance):
    pairs = sorted(
        (np.linalg.norm(np.subtract(p["uv"], t["uv"])), i, j)
        for i, p in enumerate(predictions)
        for j, t in enumerate(truth)
        if p["color"] == t["color"]
    )
    hits = {}
    used = set()
    for err, i, j in pairs:
        if err <= tolerance and i not in used and j not in hits:
            hits[j] = i
            used.add(i)
    return hits


def evaluate(kind):
    observer = load_model(kind)
    cap = cv2.VideoCapture(str(ROOT / "test_video.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS)
    config = json.loads((ROOT / "data/vision/workcell.json").read_text(encoding="utf-8-sig"))["intrinsics"]
    records = []
    scores = {
        "hits_2d": 0,
        "hits_3d": 0,
        "false_2d": 0,
        "false_3d": 0,
        "held_2d": 0,
        "held_3d": 0,
        "blue_yellow_2d_frames": 0,
        "blue_yellow_3d_frames": 0,
        "duplicate_red_green_3d_frames": 0,
    }
    times = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        i = len(records)
        jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 83])[1]
        start = time.perf_counter()
        frame = cv2.imdecode(jpeg, cv2.IMREAD_REDUCED_COLOR_2 | cv2.IMREAD_IGNORE_ORIENTATION)
        k, dist = camera_matrix(img.shape, config)
        k[:2] *= 0.5
        k[:2, 2] -= 0.25
        boxes, poses = targets(observer, frame, k, dist, i / fps)
        times.append((time.perf_counter() - start) * 1000)
        scores["duplicate_red_green_3d_frames"] += any(
            sum(p["color"] == color for p in poses) > 1 for color in ("red", "green")
        )
        for source, stage in ((boxes, "2d"), (poses, "3d")):
            for item in source:
                item["uv"] = (np.asarray(item["uv"]) * 2 + 0.5).tolist()
            scores["blue_yellow_" + stage + "_frames"] += any(
                p["color"] in ("blue", "yellow") for p in source
            )
            if i in TWO_CUBES:
                truth = [{"color": n, "uv": p} for n, p in zip(("red", "green"), TWO_CUBES[i], strict=True)]
                count = len(match(source, truth, 65))
                scores["hits_" + stage] += count
                scores["false_" + stage] += len(source) - count
            if i in HELD:
                n, x, y = HELD[i]
                scores["held_" + stage] += bool(match(source, [{"color": n, "uv": [x, y]}], 65))
        records.append({"boxes": boxes, "poses": poses})
    cap.release()
    if len(records) != 435:
        raise ValueError("These manual real-image annotations refer to the 435-frame test_video.mp4")
    times = np.array(times[10:])
    scores.update(median_ms=float(np.median(times)), p95_ms=float(np.percentile(times, 95)))
    truth = json.loads((TRIAL / "test-truth.json").read_text())
    errors = []
    total = 0
    synthetic_false = 0
    by_color = {color: {"targets": 0, "hits": 0} for color in ("red", "green", "blue", "yellow")}
    for i, data in enumerate(truth):
        img = cv2.imread(str(TRIAL / "pose/images/test" / f"{i:04}.jpg"))
        # Each synthetic view is a separate scene; reset temporal associations.
        _, poses = targets(observer, img, np.array(data["k"]), None, i + 1000)
        hits = match(poses, data["objects"], 20)
        synthetic_false += len(poses) - len(hits)
        for j, target in enumerate(data["objects"]):
            by_color[target["color"]]["targets"] += 1
            by_color[target["color"]]["hits"] += int(j in hits)
        total += len(data["objects"])
        errors.extend(
            np.linalg.norm(np.subtract(poses[index]["xyz"], data["objects"][j]["xyz"]))
            for j, index in hits.items()
        )
    scores.update(
        synthetic_targets=total,
        synthetic_false_poses=synthetic_false,
        synthetic_by_color=by_color,
        synthetic_poses=len(errors),
        synthetic_median_mm=float(np.median(errors)) if errors else None,
        synthetic_p95_mm=float(np.percentile(errors, 95)) if errors else None,
    )
    (TRIAL / (kind + "-replay.json")).write_text(json.dumps(records))
    (TRIAL / (kind + "-scores.json")).write_text(json.dumps(scores, indent=2))
    print(kind, json.dumps(scores), flush=True)


def evaluate_angles():
    """Paired orientation errors on new scenes independent of the training seeds."""
    rng = np.random.default_rng(817421)
    backdrop = backgrounds()
    model = load_trial("pose", "cuda")
    records = []
    total = 0
    matched = [0, 0]
    for _ in range(400):
        image, objects, k = render(rng, *backdrop)
        total += len(objects)
        quality = int(rng.integers(65, 94))
        jpg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])[1]
        frame = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
        truth = [dict(uv=(o["lo"] + o["hi"]) / 2, color=NAMES[o["cls"]]) for o in objects]
        predictions = []
        for mode in (False, True):
            model.reset()
            model.observe(frame, None, k, None, 1., refine_edges=mode)
            rows = [dict(row, uv=np.mean(row["pixels"], axis=0)) for row in model.estimates]
            pairs = match(rows, truth, 20)
            matched[int(mode)] += len(pairs)
            predictions.append({j: rows[i] for j, i in pairs.items()})
        old, new = predictions
        for j in old.keys() & new.keys():
            reference = model.fit(objects[j]["pts"], k, None, SIZE)[0][:3, :3]
            angles = []
            for row in (old[j], new[j]):
                delta = reference.T @ np.array(row["camera_pose"])[:3, :3]
                cosine = max((np.trace(delta @ symmetry) - 1) / 2 for symmetry in SYMMETRIES)
                angles.append(np.rad2deg(np.arccos(np.clip(cosine, -1, 1))))
            records.append([*angles, bool(new[j].get("outline_refined"))])
    errors = np.array(records).reshape(-1, 3)
    print(f"Targets: {total}; matches before/after: {matched}")
    for name, sample in (("all paired", errors), ("refined", errors[errors[:, 2] > 0])):
        if len(sample):
            print(name, len(sample), "P50/P95 degrees (before, after):",
                  np.percentile(sample[:, :2], [50, 95], axis=0).round(2).tolist(),
                  "under 10 degrees:", (sample[:, :2] < 10).sum(axis=0).tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", help="current, detect, pose-raw, pose, vit or rescue")
    parser.add_argument("--angle-check", action="store_true", help="paired symmetry-aware pose benchmark")
    args = parser.parse_args()
    cv2.setNumThreads(2)
    if args.angle_check:
        evaluate_angles()
        raise SystemExit(0)
    for kind in args.models or ("current", "pose-raw", *MODELS, "rescue"):
        if kind not in ("current", "pose-raw", *MODELS, "rescue"):
            parser.error(f"Unknown model: {kind}")
        evaluate(kind)
