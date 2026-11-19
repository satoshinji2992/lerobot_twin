"""Run paired, batch-one CUDA evaluations for RACE-6D RGB and the deployed YOLO cube pipeline."""

import hashlib
import json
import time

import cv2
import numpy as np
import torch
from evaluate_cube_models import HELD, TWO_CUBES, match
from race6d_cube_trial import NAMES, OUT, ROOT, SIDE, angles, batch_data, network
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from twin_depot.cube_pose import camera_matrix
from twin_depot.cube_trial_observer import load_trial
from twin_depot.cuboid_pnp import CENTRED


class ExportRGB(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        out = self.model(image)
        return tuple(out[k] for k in ("pred_logits", "pred_boxes", "pred_rotations", "pred_translations"))


def race_runtime():
    import onnxruntime as ort

    weights = OUT / "cube_rgb_best.pth"
    artifact = OUT / "race_cube_rgb.onnx"
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    net = ExportRGB(network(weights).deploy()).cuda().eval()
    probe = torch.rand((1, 3, SIDE, SIDE), device="cuda")
    with torch.no_grad():
        torch.onnx.export(
            net,
            probe,
            artifact.as_posix(),
            input_names=["image"],
            output_names=["logits", "boxes", "rotations", "translations"],
            opset_version=17,
            dynamo=False,
        )
    ort.preload_dlls()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(
        str(artifact),
        sess_options=options,
        providers=[("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC", "use_tf32": "0"})],
    )
    assert session.get_providers()[0] == "CUDAExecutionProvider"
    # DETR queries are an unordered set; near-tied background queries can change order.
    checked = 0
    worst = np.zeros(4)
    for path in sorted((ROOT / "data/vision/model-trial/pose/images/val").glob("*.jpg"))[:32]:
        blob = cv2.dnn.blobFromImage(cv2.imread(str(path)), 1 / 255, swapRB=True)
        with torch.no_grad():
            reference = [x[0].cpu().numpy() for x in net(torch.from_numpy(blob).cuda())]
        actual = [x[0] for x in session.run(None, {"image": blob})]
        ref = reference[0].max(-1) >= np.log(0.25 / 0.75)
        got = actual[0].max(-1) >= np.log(0.25 / 0.75)
        assert ref.sum() == got.sum(), path
        if not ref.any():
            continue
        a, b = linear_sum_assignment(cdist(reference[1][ref], actual[1][got]))
        for n, (r, v) in enumerate(zip(reference, actual)):
            diff = abs(r[ref][a] - v[got][b])
            worst[n] = max(worst[n], diff.max())
            np.testing.assert_allclose(r[ref][a], v[got][b], atol=0.01, rtol=0.006)
        checked += int(ref.sum())
    assert checked > 0
    print("ORT parity: retained detections, max errors (logits, box, R, t)", checked, worst, flush=True)
    # Same exported network, production CUDA math mode (Tensor Cores).
    fast = ort.InferenceSession(
        str(artifact),
        sess_options=options,
        providers=[("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC", "use_tf32": "1"})],
    )
    assert fast.get_providers()[0] == "CUDAExecutionProvider"
    return session, fast


class Experiment:
    def __init__(self, name, model):
        self.name = name
        self.model = model

    def reset(self):
        if self.name == "yolo":
            self.model.reset()

    def __call__(self, image, k, dist, at):
        if self.name == "yolo":
            self.model.observe(image, None, k, dist, at)
            return [
                dict(
                    color=r["color"], rot=np.array(r["camera_pose"])[:3, :3], uv=np.mean(r["pixels"], axis=0)
                )
                for r in self.model.estimates
            ]
        h, w = image.shape[:2]
        scale = SIDE / max(w, h)
        nw, nh = round(w * scale), round(h * scale)
        left, top = (SIDE - nw) // 2, (SIDE - nh) // 2
        canvas = np.full((SIDE, SIDE, 3), 114, np.uint8)
        canvas[top : top + nh, left : left + nw] = cv2.resize(image, (nw, nh))
        blob = cv2.dnn.blobFromImage(canvas, 1 / 255, swapRB=True)
        logits, boxes, rotations, translations = self.model.run(None, {"image": blob})
        ids = logits[0].argmax(-1)
        scores = logits[0].max(-1)
        rows = []
        for i in np.flatnonzero(scores >= np.log(0.25 / 0.75)):
            box = boxes[0, i] * SIDE
            t = translations[0, i]
            uv = (box[:2] + t[:2] * box[2:] - [left, top]) / [nw / w, nh / h]
            depth = float(np.exp(t[2]) * 1000)
            # The translation head encodes the projected object centre and metric log depth.
            xy = cv2.undistortPoints(uv.reshape(1, 1, 2), k, dist).ravel()
            xyz = np.r_[xy, 1.0] * depth
            rotation = rotations[0, i].astype(float).reshape(3, 3)
            pixels = cv2.projectPoints(CENTRED, cv2.Rodrigues(rotation)[0], xyz, k, dist)[0].reshape(-1, 2)
            rows.append(dict(color=NAMES[ids[i]], rot=rotation, uv=pixels.mean(0)))
        return rows


def summary(values):
    v = np.asarray(values)
    return (
        dict(n=len(v), median=float(np.median(v)), p95=float(np.percentile(v, 95)), below10=int(sum(v < 10)))
        if len(v)
        else dict(n=0)
    )


def compare_angles(models):
    imgs, target, ks = batch_data("heldout", 817421, 400)
    outcomes = {m.name: {} for m in models}
    predictions = {m.name: 0 for m in models}
    for i, (im, gt, k) in enumerate(zip(imgs, target, ks, strict=True)):
        for m in models:
            m.reset()
            rows = m(im, k, None, i + 1.0)
            predictions[m.name] += len(rows)
            outcomes[m.name].update({(i, j): deg for j, deg in angles(rows, gt).items()})
    common = set.intersection(*(set(vals) for vals in outcomes.values()))
    return dict(
        frames=len(imgs),
        targets=sum(map(len, target)),
        models={
            k: dict(
                own_matches=summary(list(v.values())),
                paired=summary([v[j] for j in common]),
                predictions=predictions[k],
            )
            for k, v in outcomes.items()
        },
    )


def video(model):
    path = ROOT / "test_video.mp4"
    config = json.loads((ROOT / "data/vision/workcell.json").read_text(encoding="utf-8-sig"))["intrinsics"]
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    ok, frame = cap.read()
    assert ok
    k, dist = camera_matrix(frame.shape, config)
    k[:2] *= 0.5
    k[:2, 2] -= 0.25

    def preprocess(frame):
        return cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2), interpolation=cv2.INTER_AREA)

    for i in range(30):
        model(preprocess(frame), k, dist, i / fps)
    model.reset()
    cap.release()
    cap = cv2.VideoCapture(str(path))
    times = []
    pipeline = []
    hits = false = held = observations = 0
    start = time.perf_counter()
    while True:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break
        im = preprocess(frame)
        i = len(times)
        t1 = time.perf_counter()
        rows = model(im, k, dist, i / fps)
        t2 = time.perf_counter()
        times.append((t2 - t1) * 1000)
        pipeline.append((t2 - t0) * 1000)
        observations += len(rows)
        for row in rows:
            row["uv"] = row["uv"] * 2 + 0.5
        if i in TWO_CUBES:
            truth = [dict(color=c, uv=uv) for c, uv in zip(("red", "green"), TWO_CUBES[i])]
            n = len(match(rows, truth, 65))
            hits += n
            false += len(rows) - n
        if i in HELD:
            c, x, y = HELD[i]
            held += bool(match(rows, [dict(color=c, uv=[x, y])], 65))
    elapsed = time.perf_counter() - start
    cap.release()
    assert len(times) == 435
    return dict(
        frames=len(times),
        source_fps=fps,
        fps=len(times) / elapsed,
        algorithm_ms=summary(times),
        pipeline_ms=summary(pipeline),
        within_60fps_budget=sum(t <= 1000 / 60 for t in pipeline),
        within_20fps_budget=sum(t <= 50 for t in pipeline),
        observations=observations,
        labeled_hits=hits,
        labeled_false=false,
        labeled_targets=2 * len(TWO_CUBES),
        held=held,
        held_labels=len(HELD),
    )


def main():
    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    full, fast = race_runtime()
    models = [
        Experiment("yolo", load_trial("pose", "cuda")),
        Experiment("race_fp32", full),
        Experiment("race_tf32", fast),
    ]
    results = dict(
        gpu=torch.cuda.get_device_name(),
        best_epoch=int((OUT / "best_epoch.txt").read_text()),
        checkpoint_sha256=hashlib.sha256((OUT / "cube_rgb_best.pth").read_bytes()).hexdigest(),
        angles=compare_angles(models),
    )
    print("angles", results["angles"], flush=True)
    results["video"] = {}
    # Rotate order across three full passes. Report every run, not the fastest one.
    for run in range(3):
        for model in models[run:] + models[:run]:
            stats = video(model)
            results["video"].setdefault(model.name, []).append(stats)
            print("video", model.name, run, stats, flush=True)
    (OUT / "comparison.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
