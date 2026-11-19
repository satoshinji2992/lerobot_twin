"""RGB-only RACE-6D transfer trial on the same synthetic cube scenes as YOLO11n-Pose."""

import argparse
import hashlib
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from cube_model_trial import NAMES, ROOT, backgrounds, render
from evaluate_cube_models import match
from scipy.optimize import linear_sum_assignment
from torchvision.ops import sigmoid_focal_loss
from twin_depot.cuboid_pnp import CENTRED, SYMMETRIES

OUT = ROOT / "data/vision/race6d"
SIDE = 384
HASH = "03430ec1625f5eac3b37bd45c7dea56b901076bce4027d568fc9af0fed341cc7"
EDGES = [(a, b) for i, a in enumerate(CENTRED) for b in CENTRED[i + 1 :] if np.count_nonzero(a != b) == 1]
POINTS = np.vstack([CENTRED, [(2 * a + b) / 3 for a, b in EDGES], [(a + 2 * b) / 3 for a, b in EDGES]])


def import_network():
    # Isolate the upstream network from optional BOP rendering/training imports.
    # No forward operators are replaced. PyTorch3D's conversion functions are pure torch.
    module = OUT / "rgb_network"
    module.mkdir(exist_ok=True)
    root = OUT / "upstream/src"
    for name in (
        "common",
        "presnet",
        "utils",
        "hybrid_encoder",
        "box_ops",
        "denoising",
        "race6d_decoder_dqe",
        "race6d",
    ):
        src = root / ("nn/backbone" if name in ("common", "presnet") else "zoo/race6d") / (name + ".py")
        code = (
            src.read_text(encoding="utf8")
            .replace("from ...core import register", "")
            .replace("@register()", "")
        )
        code = code.replace("from pytorch3d.transforms import", "from .rotations import")
        code = code.replace("from pytorch3d.ops import sample_farthest_points", "")
        (module / (name + ".py")).write_text(code, encoding="utf8")
    code = (OUT / "rotation_conversions.py").read_text(encoding="utf8")
    (module / "rotations.py").write_text(
        code.replace("from ..common.datatypes import Device", "Device = torch.device"), encoding="utf8"
    )
    sys.path.insert(0, str(OUT))
    from rgb_network.hybrid_encoder import HybridEncoder
    from rgb_network.presnet import PResNet
    from rgb_network.race6d import RACE6D
    from rgb_network.race6d_decoder_dqe import RACE6DTransformer_DQE

    backbone = PResNet(
        depth=50,
        variant="d",
        freeze_at=-1,
        return_idx=[1, 2, 3],
        num_stages=4,
        freeze_norm=False,
        pretrained=False,
    )
    encoder = HybridEncoder(
        in_channels=[512, 1024, 2048], depth_mult=0.67, enc_act="quick_gelu", eval_spatial_size=[SIDE, SIDE]
    )
    decoder = RACE6DTransformer_DQE(
        num_classes=4,
        feat_channels=[256] * 3,
        num_queries=50,
        num_layers=3,
        num_points=[3, 6, 3],
        activation="quick_gelu",
        mlp_act="quick_gelu",
        eval_spatial_size=[SIDE, SIDE],
        vis_enc=True,
        num_denoising=0,
    )
    return RACE6D(backbone, encoder, decoder)


def network(weights=None):
    net = import_network()
    if weights:
        net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    else:
        file = OUT / "tless.pth"
        assert hashlib.sha256(file.read_bytes()).hexdigest() == HASH
        state = torch.load(file, map_location="cpu", weights_only=True)["ema"]["module"]
        own = net.state_dict()
        compatible = {
            k: v
            for k, v in state.items()
            if k in own and v.shape == own[k].shape and not k.endswith(("valid_mask", "anchors_bbox"))
        }
        missing = net.load_state_dict(compatible, strict=False).missing_keys
        print("Loaded", len(compatible), "tensors; regenerated or cube-specific:", missing, flush=True)
        assert all("score_head" in k or k.endswith(("valid_mask", "anchors_bbox")) for k in missing)
    return net.cuda()


def batch_data(split, seed, count):
    rng = np.random.default_rng(seed)
    bg = backgrounds()
    images, targets, cameras = [], [], []
    for i in range(count):
        raw, objects, k = render(rng, *bg)
        quality = int(rng.integers(65, 94))
        if split in ("train", "val"):
            data = ROOT / "data/vision/model-trial/pose"
            frame = cv2.imread(str(data / f"images/{split}/{i:04}.jpg"))
            labels = (data / f"labels/{split}/{i:04}.txt").read_text().splitlines()
            assert len(labels) == len(objects)
            for row, obj in zip(labels, objects):
                np.testing.assert_allclose(
                    np.array(row.split()[5:], float).reshape(8, 2) * SIDE, obj["pts"], atol=0.0003
                )
        else:
            frame = cv2.imdecode(
                cv2.imencode(".jpg", raw, [cv2.IMWRITE_JPEG_QUALITY, quality])[1], cv2.IMREAD_COLOR
            )
        images.append(frame)
        cameras.append(k)
        rows = []
        for o in objects:
            rotation = o["rotation"]
            rot = cv2.Rodrigues(rotation)[0]
            lo, hi = o["lo"], o["hi"]
            size = hi - lo
            center = (hi + lo) / 2
            pts = cv2.projectPoints(POINTS, rot, o["xyz"], k, None)[0].reshape(-1, 2)
            uv = cv2.projectPoints(np.zeros((1, 3)), rot, o["xyz"], k, None)[0].reshape(2)
            rows.append(
                [
                    o["cls"],
                    *center / SIDE,
                    *size / SIDE,
                    *rotation.ravel(),
                    *((pts - center) / size).ravel(),
                    *(uv - center) / size,
                    np.log(o["xyz"][2] / 1000),
                ]
            )
        targets.append(np.array(rows, dtype=np.float32).reshape(-1, 81))
    return images, targets, cameras


def to_gpu(images):
    return (
        torch.as_tensor(np.stack(images)[..., ::-1].copy(), device="cuda")
        .permute(0, 3, 1, 2)
        .float()
        .div_(255)
    )


def assign(out, target, counts):
    probs = out["pred_logits"].detach().float().sigmoid()
    boxes = out["pred_boxes"].detach().float()
    cost = (
        (
            6 * torch.cdist(boxes, target[:, :, 1:5], p=1)
            - probs.gather(2, target[:, None, :, 0].clamp_min(0).long().expand(-1, boxes.shape[1], -1))
        )
        .cpu()
        .numpy()
    )
    ids = []
    for batch, mat in enumerate(cost):
        n = counts[batch]
        if n:
            q, t = linear_sum_assignment(mat[:, :n])
            ids.extend(zip([batch] * n, q, t))
    if not ids:
        return tuple(torch.empty(0, device="cuda", dtype=torch.long) for _ in range(3))
    return torch.tensor(ids, device="cuda").T.unbind()


def loss(out, target, indices, sym):
    b, q, t = indices
    label = torch.zeros_like(out["pred_logits"])
    gt = target[b, t]
    label[b, q, gt[:, 0].long()] = 1
    total = sigmoid_focal_loss(out["pred_logits"].float(), label.float(), reduction="sum") / max(len(b), 1)
    if len(b):
        r = out["pred_rotations"][b, q].reshape(-1, 3, 3).float()
        equiv = gt[:, None, 5:14].reshape(-1, 1, 3, 3) @ sym
        rerr = ((r[:, None] - equiv) ** 2).mean((2, 3)).min(1).values.mean()
        total = total + 6 * F.l1_loss(out["pred_boxes"][b, q].float(), gt[:, 1:5]) + 2 * rerr
        total = total + 2 * F.l1_loss(out["pred_keypoints"][b, q].float(), gt[:, 14:78])
        total = total + F.l1_loss(out["pred_translations"][b, q].float(), gt[:, 78:81])
    return total


def decode(out, w=SIDE, h=SIDE):
    score, cls = out["pred_logits"][0].float().sigmoid().max(-1)
    selected = score >= 0.25
    vals = (
        torch.cat((cls[:, None], score[:, None], out["pred_boxes"][0], out["pred_rotations"][0]), 1)[selected]
        .detach()
        .float()
        .cpu()
        .numpy()
    )
    return [
        dict(color=NAMES[int(v[0])], score=float(v[1]), uv=v[2:4] * [w, h], rot=v[6:].reshape(3, 3))
        for v in vals
    ]


def angles(rows, target):
    truth = [dict(color=NAMES[int(r[0])], uv=r[1:3] * SIDE) for r in target]
    hits = match(rows, truth, 20)
    errors = {}
    for j, i in hits.items():
        delta = target[j, 5:14].reshape(3, 3).T @ rows[i]["rot"]
        c = (np.trace(delta @ SYM_ARRAY, axis1=1, axis2=2).max() - 1) / 2
        errors[j] = float(np.degrees(np.arccos(np.clip(c, -1, 1))))
    return errors


@torch.no_grad()
def validate(net, imgs, targets):
    net.eval()
    errors = []
    ntruth = sum(map(len, targets))
    n = 0
    for i in range(0, len(imgs), 8):
        out = net(to_gpu(imgs[i : i + 8]))
        for j, gt in enumerate(targets[i : i + 8]):
            rows = decode({k: out[k][j : j + 1] for k in ("pred_logits", "pred_boxes", "pred_rotations")})
            errors.extend(angles(rows, gt).values())
            n += len(rows)
    return (
        len(errors),
        n,
        round(float(np.median(errors)), 2) if errors else None,
        round(sum(e < 15 for e in errors) / max(1, ntruth + n - len(errors)), 4),
    )


def train(epochs):
    torch.manual_seed(83)
    rng = np.random.default_rng(83)
    imgs, tg, _ = batch_data("train", 901, 864)
    val, vt, _ = batch_data("val", 902, 128)
    net = network()
    sym = torch.tensor(SYM_ARRAY, device="cuda", dtype=torch.float32)
    # Cache fixed pre-trained backbone feature maps for the small transfer set.
    net.backbone.eval().requires_grad_(False)
    feats = [[] for _ in range(3)]
    with torch.no_grad():
        for i in range(0, len(imgs), 12):
            xs = net.backbone(to_gpu(imgs[i : i + 12]))
            for dest, x in zip(feats, xs):
                dest.append(x.half())
    feats = [torch.cat(t) for t in feats]
    optimizer = torch.optim.AdamW(
        [p for p in net.parameters() if p.requires_grad], lr=0.00016, weight_decay=0.0001
    )
    best = -1
    for epoch in range(epochs):
        net.train()
        net.backbone.eval()
        start = time.perf_counter()
        losses = []
        for group in np.array_split(rng.permutation(len(imgs)), 72):
            counts = [len(tg[i]) for i in group]
            truth = np.zeros((len(group), max(max(counts), 1), 81), np.float32)
            for j, ix in enumerate(group):
                truth[j, : counts[j]] = tg[ix]
            target = torch.as_tensor(truth, device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = net.decoder(net.encoder([t[group].float() for t in feats]))
            idx = assign(out, target, counts)
            total = loss(out, target, idx, sym)
            for aux in out["aux_outputs"]:
                total = total + 0.4 * loss(aux, target, idx, sym)
            enc = out["enc_aux_outputs"][0]
            total = total + 0.75 * loss(enc, target, assign(enc, target, counts), sym)
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            losses.append(float(total.detach()))
        if epoch + 1 == max(2, int(epochs * 0.7)):
            optimizer.param_groups[0]["lr"] *= 0.35
        metrics = validate(net, val, vt)
        print(
            "epoch",
            epoch + 1,
            "seconds",
            round(time.perf_counter() - start, 1),
            "loss",
            round(np.mean(losses), 3),
            "val matches/predictions/median/selection",
            metrics,
            flush=True,
        )
        if metrics[-1] > best:
            best = metrics[-1]
            torch.save(net.state_dict(), OUT / "cube_rgb_best.pth")
            (OUT / "best_epoch.txt").write_text(str(epoch + 1))


SYM_ARRAY = np.stack(SYMMETRIES)
if __name__ == "__main__":
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    torch.set_float32_matmul_precision("high")
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=18, type=int)
    args = parser.parse_args()
    train(args.epochs)
