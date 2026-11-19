"""Train the small occluded-cube corner estimator from procedurally rendered labelled views."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional
from twin_depot.cube_completion import CENTRED, WEIGHTS, patch_input
from twin_depot.cube_pose import visible_regions
from twin_depot.cuboid_pnp import SYMMETRIES

FACES = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]])


class CornerNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        for cin, cout in ((2, 16), (16, 24), (24, 40)):
            layers.extend([nn.Conv2d(cin, cout, 3, stride=2, padding=1), nn.BatchNorm2d(cout), nn.ReLU()])
        self.net = nn.Sequential(
            *layers, nn.Flatten(), nn.Linear(40 * 5 * 5, 96), nn.ReLU(), nn.Linear(96, 17)
        )

    def forward(self, x):
        return self.net(x)


def views(count, seed):
    rng = np.random.default_rng(seed)
    images = []
    labels = []
    objects = []
    while len(images) < count:
        positive = rng.random() > 0.23
        value = np.zeros((96, 96), np.uint8)
        mask = np.zeros_like(value)
        quat = rng.normal(size=4)
        quat /= np.linalg.norm(quat)
        rv = 2 * np.arccos(quat[0]) * quat[1:] / max(0.00001, np.linalg.norm(quat[1:]))
        rot = cv2.Rodrigues(rv)[0]
        vertices = CENTRED @ rot.T
        z = rng.uniform(150, 850)
        points = 48 + vertices[:, :2] / (vertices[:, 2:3] + z) * (z * rng.uniform(1.25, 1.7))
        if positive:
            for face in sorted(FACES, key=lambda f: -vertices[f, 2].mean()):
                polygon = np.int32(np.rint(points[face]))
                cv2.fillConvexPoly(mask, polygon, 255)
                cv2.fillConvexPoly(value, polygon, int(rng.uniform(26, 218)))
        else:
            if rng.random() < 0.4:
                cv2.ellipse(
                    mask, (48, 48), tuple(rng.integers(9, 34, 2)), rng.uniform(0, 180), 0, 360, 255, -1
                )
            else:
                p = rng.integers(16, 80, (rng.integers(4, 12), 2)).astype(np.int32)
                cv2.fillPoly(mask, [p], 255)
            value[mask > 0] = rng.integers(28, 220)
            for _ in range(rng.integers(0, 8)):
                cv2.line(
                    value,
                    tuple(rng.integers(8, 90, 2)),
                    tuple(rng.integers(8, 90, 2)),
                    int(rng.integers(10, 230)),
                    2,
                )
        full = cv2.countNonZero(mask)
        for _ in range(rng.choice([0, 1, 1, 1, 2, 2, 3])):
            theta = rng.uniform(-np.pi, np.pi)
            center = np.int32(48 + rng.uniform(17, 48) * np.array([np.cos(theta), np.sin(theta)]))
            cv2.ellipse(
                mask,
                tuple(center),
                tuple(rng.integers([8, 13], [22, 40])),
                rng.uniform(0, 180),
                0,
                360,
                0,
                -1,
            )
        if cv2.countNonZero(mask) < max(90, 0.27 * full):
            continue
        if rng.random() < 0.4:
            value = cv2.GaussianBlur(value, (5, 5), rng.uniform(0.5, 1.9))
        value = np.uint8(
            np.clip(value.astype(float) + rng.normal(0, rng.uniform(0.1, 8), value.shape), 0, 255)
        )
        # Same proposal crop as inference, including fingers splitting one silhouette.
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        c = max(contours, key=cv2.contourArea)
        mask[:] = 0
        cv2.fillPoly(mask, [c], 255)
        if cv2.contourArea(c) < max(80, 0.24 * full):
            continue
        out = patch_input(value, mask, cv2.boundingRect(c))
        if out is None:
            continue
        tensor, origin, side = out
        images.append(tensor)
        labels.append((points - origin) / side * 2 - 1)
        objects.append(positive)
    return np.stack(images), np.float32(labels), np.float32(objects)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=WEIGHTS)
    parser.add_argument("--samples", type=int, default=24000)
    parser.add_argument("--steps", type=int, default=2200)
    parser.add_argument(
        "--negative", type=Path, nargs="*", default=[], help="optional photos containing no target cubes"
    )
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(981)
    cv2.setNumThreads(1)
    torch.set_num_threads(3)
    print("Generating labelled cube views", flush=True)

    def tensors(n, seed):
        return [torch.as_tensor(a, device=device) for a in views(n, seed)]

    train = tensors(args.samples, 981)
    validation = tensors(800, 1935)
    background = []
    for file in args.negative:
        photo = cv2.imread(str(file))
        if photo is None:
            raise ValueError(f"Cannot open negative image {file}")
        for scale in (1, 0.6):
            img = cv2.resize(photo, None, fx=scale, fy=scale)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            for _, (x, y, w, h), contour in visible_regions(hsv):
                mask = np.zeros((h, w), np.uint8)
                cv2.fillPoly(mask, [np.int32(contour - [x, y])], 255)
                example = patch_input(hsv[y : y + h, x : x + w, 2], mask, (0, 0, w, h))
                if example is not None:
                    for k in range(4):
                        patch = np.rot90(example[0], k, axes=(1, 2))
                        background.extend([patch.copy(), patch[:, :, ::-1].copy()])
    if background:
        # Real labels, tape and skin are hard negatives for hue-based proposals.
        rng = np.random.default_rng(649)
        ids = rng.choice(args.samples, args.samples // 9, replace=False)
        train[0][ids] = torch.as_tensor(
            np.stack([background[n] for n in rng.integers(len(background), size=len(ids))]), device=device
        )
        train[2][ids] = 0
    permutations = torch.as_tensor(
        np.asarray(
            [
                np.linalg.norm((CENTRED @ r)[:, None, :] - CENTRED[None, :, :], axis=2).argmin(1)
                for r in SYMMETRIES
            ]
        ),
        device=device,
    )
    net = CornerNet().to(device)
    # Seed distinct corner identities so the symmetry-invariant loss can select a rotation.
    template = CENTRED @ cv2.Rodrigues(np.array([0.34, 0.27, 0.11]))[0]
    with torch.no_grad():
        net.net[-1].bias[:16].copy_(torch.tensor(template[:, :2].ravel() / 38))
    opt = torch.optim.AdamW(net.parameters(), lr=0.0016)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps, eta_min=0.00012)

    def losses(data, idx):
        images, labels, objects = data
        targets = labels[idx]
        positive = objects[idx]
        pred = net(images[idx].float() / 255)
        distance = (
            (pred[:, :16].reshape(-1, 1, 8, 2) - targets[:, permutations])
            .square()
            .sum(-1)
            .mean(-1)
            .min(-1)
            .values
        )
        corner = (distance * positive).sum() / positive.sum().clamp_min(1)
        return (
            7 * corner + functional.binary_cross_entropy_with_logits(pred[:, -1], positive),
            distance,
            positive,
            pred,
        )

    for _ in range(args.steps):
        idx = torch.randint(0, args.samples, (128,), device=device)
        loss, *_ = losses(train, idx)
        opt.zero_grad()
        loss.backward()
        opt.step()
        schedule.step()
    net.eval()
    with torch.no_grad():
        _, d, p, pred = losses(validation, torch.arange(len(validation[0]), device=device))
        print(
            "Held-out median / P95 corner RMS as fraction of crop width:",
            torch.quantile(d[p.bool()].sqrt() / 2, torch.tensor([0.5, 0.95], device=device)).cpu().tolist(),
            "classification accuracy",
            ((pred[:, -1] > 0) == p.bool()).float().mean().item(),
            flush=True,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        net.cpu(),
        torch.zeros(1, 2, 40, 40),
        str(args.out),
        input_names=["crop"],
        output_names=["corners_and_logit"],
        dynamic_axes={"crop": {0: "batch"}, "corners_and_logit": {0: "batch"}},
        opset_version=13,
        dynamo=False,
    )


if __name__ == "__main__":
    main()
