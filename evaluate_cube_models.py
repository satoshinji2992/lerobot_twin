"""Evaluate the experimental models on the same JPEGs, same labels and same GPU."""

import argparse
import json
import time

import cv2
import numpy as np
from cube_model_trial import MODELS, NAMES, REFERENCE, ROOT, SIZE, TRIAL, backgrounds, render
from twin_depot.cuboid_pnp import CENTRED, SYMMETRIES
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


def evaluate_angles(seed=817421, frames=400):
    """Paired orientation errors on new scenes independent of the training seeds."""
    rng = np.random.default_rng(seed)
    backdrop = backgrounds()
    model = load_trial("pose", "cuda")
    records = []
    total = 0
    matched = [0, 0]
    def rotation_error(a, b):
        cosine = (np.trace((a.T @ b) @ symmetries, axis1=1, axis2=2).max() - 1) / 2
        return float(np.rad2deg(np.arccos(np.clip(cosine, -1, 1))))

    symmetries = np.stack(SYMMETRIES)
    # A symmetry permutes the eight physical corners together.
    permutation = np.array([np.linalg.norm((CENTRED @ s.T)[:, None] - CENTRED, axis=2).argmin(1)
                            for s in symmetries])
    corner_inputs = {}
    def record_fit(corners, k, dist, side):
        fit = TrialObserver.fit(corners, k, dist, side)
        if fit is not None:
            corner_inputs[fit[0].tobytes()] = corners
        return fit

    model.fit = record_fit
    for _ in range(frames):
        corner_inputs.clear()
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
            obj = objects[j]
            truth_r = obj["rotation"]
            raw = np.array(old[j]["camera_pose"])
            final = np.array(new[j]["camera_pose"])
            corners = corner_inputs[raw.tobytes()]
            oracle, _ = TrialObserver.fit(obj["pts"], k, None, SIZE)
            aligned = np.linalg.norm(corners - obj["pts"][permutation], axis=2).mean(1).min()
            records.append(dict(
                oracle=rotation_error(truth_r, oracle[:3, :3]),
                raw=rotation_error(truth_r, raw[:3, :3]),
                final=rotation_error(truth_r, final[:3, :3]),
                raw_center_mm=float(np.linalg.norm(raw[:3, 3] - obj["xyz"])),
                final_center_mm=float(np.linalg.norm(final[:3, 3] - obj["xyz"])),
                pred_to_reference=rotation_error(REFERENCE, raw[:3, :3]),
                truth_to_reference=rotation_error(REFERENCE, truth_r),
                corner_error_px=float(aligned),
                reprojection_px=float(np.linalg.norm(corners - old[j]["pixels"], axis=1).mean()),
                box_px=float(max(obj["hi"] - obj["lo"])),
                occlusion=float(obj["occlusion"]),
                refined=bool(new[j].get("outline_refined")),
                edge_method=new[j].get("edge_method"),
            ))
    report = dict(seed=seed, frames=frames, targets=total, matched=matched, paired=len(records))
    for name, sample in (("all", records), ("refined", [r for r in records if r["refined"]]),
                         ("unrefined", [r for r in records if not r["refined"]]),
                         ("uncovered", [r for r in records if r["occlusion"] == 0]),
                         ("covered_20pct_or_more", [r for r in records if r["occlusion"] >= 0.20]),
                         ("three_edges", [r for r in records if r["edge_method"] == "three_edges"]),
                         ("face_intersections", [r for r in records if r["edge_method"] == "face_intersections"])):
        if sample:
            report[name] = dict(n=len(sample), below10=sum(r["final"] < 10 for r in sample))
            for key in sorted(sample[0].keys() - {"refined", "edge_method"}):
                report[name][key] = np.percentile([r[key] for r in sample], [50, 95]).round(4).tolist()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", help="current, detect, pose-raw, pose, vit or rescue")
    parser.add_argument("--angle-check", action="store_true", help="paired symmetry-aware pose benchmark")
    parser.add_argument("--seed", type=int, default=817421)
    parser.add_argument("--frames", type=int, default=400)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    if args.angle_check:
        print(json.dumps(evaluate_angles(args.seed, args.frames), indent=2))
        raise SystemExit(0)
    for kind in args.models or ("current", "pose-raw", *MODELS, "rescue"):
        if kind not in ("current", "pose-raw", *MODELS, "rescue"):
            parser.error(f"Unknown model: {kind}")
        evaluate(kind)
