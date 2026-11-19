"""Quality-controlled calibration of the actual phone stream from a flat screen."""

from pathlib import Path

import cv2
import numpy as np

PATTERN = (9, 6)
MIN_VIEWS = 15
MAX_VIEWS = 45


def _coverage(corners, size):
    """Locations are coverage of observed inner corners, in camera sensor coordinates."""
    cells = np.zeros((3, 4), dtype=int)
    if not corners:
        return cells, np.zeros(4)
    xy = np.concatenate(corners).reshape(-1, 2) / size
    pos = np.clip((xy * (4, 3)).astype(int), (0, 0), (3, 2))
    np.add.at(cells, (pos[:, 1], pos[:, 0]), 1)
    return cells, np.r_[xy.min(axis=0), xy.max(axis=0)]


def _edges_covered(bounds):
    return bool(np.all(bounds[:2] <= .25) and np.all(bounds[2:] >= .75))


def _grid():
    grid = np.zeros((54, 3), np.float32)
    grid[:, :2] = np.mgrid[:9, :6].T.reshape(-1, 2)
    return grid


def _fit(train, size, grid):
    return cv2.calibrateCameraExtended([grid] * len(train), train, size, None, None,
                                       flags=cv2.CALIB_FIX_K3)


def _radial_ok(k, dist, size):
    # A fold inside the image yields ambiguous undistortion even when central RMS is low.
    w, h = size
    rim = np.array([[0, 0], [0, h], [w, h], [w, 0]])
    radius = np.linalg.norm((rim - [k[0, 2], k[1, 2]]) / [k[0, 0], k[1, 1]], axis=1).max()
    r2 = np.linspace(0, radius**2, 50)
    k1, k2, _, _, k3 = dist.ravel()
    return np.all(1 + 3*k1*r2 + 5*k2*r2**2 + 7*k3*r2**3 > .10)


def _check_views(corners, hold_ids, k, dist):
    grid = _grid()
    # Estimate the board pose on 15 corners from each held-out image. Score the other 39.
    subset = np.all(grid[:, :2].astype(int) % 2 == 0, axis=1)
    residuals = []
    for i in hold_ids:
        pts = corners[i].reshape(-1, 2)
        ok, rv, tv = cv2.solvePnP(grid[subset], pts[subset], k, dist)
        if not ok or tv[2].item() <= 0:
            raise ValueError(f"验证图 {i+1} 的棋盘姿态无法确定")
        expected = cv2.projectPoints(grid[~subset], rv, tv, k, dist)[0].reshape(-1, 2)
        residuals.append(np.linalg.norm(expected - pts[~subset], axis=1))
    err = np.concatenate(residuals)
    vrms = float(np.sqrt(np.mean(err**2)))
    p95 = float(np.percentile(err, 95))
    worst = float(max(np.sqrt(np.mean(r**2)) for r in residuals))
    return vrms, p95, worst, bool(np.isfinite(err).all())


class ScreenCalibration:
    def __init__(self, folder: Path):
        self.folder = folder
        self.corners = []
        self.size = None
        self.session = None
        self.last_quality = {}
        self.last_outline = []
        self.validation = None

    def report(self):
        cells, bounds = _coverage(self.corners, self.size)
        missing = [name for name, outside in zip(("左", "上", "右", "下"),
                   (bounds[0] > .25, bounds[1] > .25, bounds[2] < .75, bounds[3] < .75),
                   strict=True) if outside] if self.corners else []
        return dict(quality=self.last_quality, coverage=cells.tolist(), outline=self.last_outline,
                    edges_covered=_edges_covered(bounds), missing_edges=missing,
                    ready=len(self.corners) >= MIN_VIEWS,
                    image_size=self.size, validation=self.validation, minimum=MIN_VIEWS)

    def offer(self, jpeg, session):
        gray = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError("相机帧解码失败")
        self.last_quality = {}
        self.last_outline = []
        size = gray.shape[::-1]
        if self.corners and (size != self.size or session != self.session):
            raise ValueError("相机已重连或画面尺寸已变，请新开一组")
        if len(self.corners) >= MAX_VIEWS:
            raise ValueError(f"已收集 {MAX_VIEWS} 张，请暂停后计算内参")
        lo, hi = np.percentile(gray[::4, ::4], [1, 99])
        if hi - lo < 32:
            return "画面对比太低。请检查镜头遮挡、屏幕亮度与曝光。"
        found, pts = cv2.findChessboardCornersSB(gray, PATTERN)
        if not found or not np.isfinite(pts).all():
            return "未找到完整棋盘。让四周白边入镜，并避开反光。"
        xy = pts.reshape(-1, 2)
        scaled = xy / size
        self.last_outline = scaled[[0, 8, 53, 45]].round(3).tolist()
        area = cv2.contourArea(cv2.convexHull(xy)) / gray.size
        self.last_quality["area_pct"] = round(100 * area, 1)
        if area < .035:
            return "棋盘在画面中太小。请靠近屏幕。"

        profiles = [cv2.estimateChessboardSharpness(gray, PATTERN, pts, .8, vertical)[0]
                    for vertical in (False, True)]
        blur = max(v[0] for v in profiles)
        contrast = min(v[2] - v[1] for v in profiles)
        if not np.isfinite([blur, contrast]).all():
            return "未能测得有效黑白边缘。请调整对焦。"
        self.last_quality.update(blur_px=round(blur, 2), contrast=round(contrast))
        if contrast < 40:
            return "黑白格对比太低。请避开反光，调整屏幕亮度。"
        if blur > 3.5:
            return "格边模糊。请对焦屏幕，保持手机静止后再采。"

        for previous in self.corners:
            old = previous.reshape(-1, 2) / size
            distance = min(np.linalg.norm(scaled - order, axis=1).mean() for order in (old, old[::-1]))
            if distance < .035:
                return "这个视角已采过。请改变倾角，或让棋盘移向尚未覆盖的位置。"
        self.folder.mkdir(parents=True, exist_ok=True)
        (self.folder / f"view-{len(self.corners)+1:02}.jpg").write_bytes(jpeg)
        self.size, self.session = size, session
        self.corners.append(pts)
        self.validation = None
        return "已记录。请变换视角，再稍停片刻。"

    def migrate(self, original, profile, clockwise):
        old = (original or {}).get("capture_profile") or {}
        if not profile or not old or old.get("source") != "phone" or not old.get("camera_key"):
            raise ValueError("需要同一手机镜头的竖屏标定和横屏验证画面")
        for key in ("source", "camera_key", "zoom", "facing", "resize_mode"):
            if old.get(key) != profile.get(key):
                raise ValueError("镜头、倍率或裁切设置与竖屏标定不一致")
        w, h = original["image_size"]
        if (w >= h or self.size != (h, w) or old["video_size"][::-1] != profile["video_size"]
                or sorted(old["native_size"]) != sorted(profile["native_size"])):
            raise ValueError("横屏验证要求同一采集尺寸，宽高与竖屏标定对调")
        if len(self.corners) < 4 or not _edges_covered(_coverage(self.corners, self.size)[1]):
            raise ValueError("至少采集四个不同的横屏视角，让棋盘覆盖画面四边")
        k = np.array(original["K"], dtype=float)
        fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
        k[0, 0], k[1, 1] = fy, fx
        k[0, 2], k[1, 2] = (h-1-cy, cx) if clockwise else (cy, w-1-cx)
        dist = np.array(original["dist"], dtype=float).ravel()
        if dist.size != 5:
            raise ValueError("目前仅支持项目使用的五系数畸变模型")
        p1, p2 = dist[2:4]
        dist[2:4] = (p2, -p1) if clockwise else (-p2, p1)
        ids = range(len(self.corners))
        rms, p95, worst, finite = _check_views(self.corners, ids, k, dist)
        if not finite or rms > 1 or p95 > 1.8 or worst > 1.5:
            raise ValueError(f"横屏验证未通过，RMS {rms:.3f} px，P95 {p95:.3f} px；请核对方向与镜头")
        return dict(image_size=list(self.size), K=k.tolist(), dist=dist.tolist(),
                    capture_profile=profile, views=len(ids), rotation="cw" if clockwise else "ccw",
                    validation=dict(heldout_rms_px=round(rms, 3), heldout_p95_px=round(p95, 3),
                                    worst_view_rms_px=round(worst, 3), heldout_views=len(ids), passed=True))

    def solve(self):
        if len(self.corners) < MIN_VIEWS:
            raise ValueError(f"至少需要 {MIN_VIEWS} 张不同视角")
        self.validation = None
        # Reserve every fifth view across the session. They never enter the fitted camera model.
        train_ids = [i for i in range(len(self.corners)) if i % 5 != 4]
        hold_ids = [i for i in range(4, len(self.corners), 5)]
        grid = _grid()
        def fit():
            return _fit([self.corners[i] for i in train_ids], self.size, grid)
        rms, k, dist, rs, _, _, _, per_view = fit()
        errors = per_view.ravel()
        med = np.median(errors)
        mad = np.median(np.abs(errors - med))
        bad = errors > max(1., med + 4 * 1.4826 * mad)
        excluded = [i for i, drop in zip(train_ids, bad, strict=True) if drop]
        if excluded:
            if len(excluded) > max(1, len(train_ids) // 8) or len(train_ids)-len(excluded) < 12:
                raise ValueError("训练样本中有异常视角，请补拍清晰照片。检查编号 " +
                                 "、".join(str(i+1) for i in excluded))
            train_ids = [i for i in train_ids if i not in excluded]
            rms, k, dist, rs, _, _, _, per_view = fit()

        if not np.isfinite(rms) or not np.isfinite(k).all() or not np.isfinite(dist).all() or rms > 1.:
            raise ValueError(f"拟合 RMS {rms:.2f} px，请检查条纹、反光及镜头切换")
        normals = np.array([cv2.Rodrigues(r)[0][:, 2] for r in rs])
        normals = np.where(normals[:, 2:3] < 0, -normals, normals)
        if np.ptp(normals, axis=0)[:2].min() < .25:
            raise ValueError("棋盘倾角变化不足。请补拍左右斜视和上下斜视。")
        cells, bounds = _coverage([self.corners[i] for i in train_ids], self.size)
        if not _edges_covered(bounds) or np.count_nonzero(cells) < 9:
            raise ValueError("画面边缘覆盖不足。请将棋盘分别移至左、右、上、下及四角。")
        w, h = self.size
        fx, fy = k[0, 0], k[1, 1]
        if (not (.20*max(w,h) < min(fx, fy) <= max(fx, fy) < 6*max(w,h))
                or not .75 < fx/fy < 1.33 or not .15*w < k[0, 2] < .85*w
                or not .15*h < k[1, 2] < .85*h or not _radial_ok(k, dist, self.size)):
            raise ValueError("内参或边缘畸变解不稳定。请增加边缘视角与倾斜幅度。")

        vrms, p95, worst, finite = _check_views(self.corners, hold_ids, k, dist)
        self.validation = dict(train_views=len(train_ids), heldout_views=len(hold_ids),
                               heldout_images=[i+1 for i in hold_ids], excluded_images=[i+1 for i in excluded],
                               fit_rms_px=round(float(rms),3), heldout_rms_px=round(vrms,3),
                               heldout_p95_px=round(p95,3), worst_view_rms_px=round(worst,3),
                               passed=bool(finite and vrms <= 1 and worst <= 1.5 and p95 <= 1.8))
        if not self.validation["passed"]:
            raise ValueError(f"留出图像验证未通过，RMS {vrms:.2f} px，P95 {p95:.2f} px。请检查运动模糊及屏幕条纹。")
        return dict(image_size=list(self.size), K=k.tolist(), dist=dist.ravel().tolist(),
                    rms_px=float(rms), views=len(self.corners), validation=self.validation.copy())
