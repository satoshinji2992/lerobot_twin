"""Measure FastSAM instance masks in a surveyed plane at cube-top height (millimetres)."""

import cv2
import numpy as np


def project(points, matrix):
    return cv2.perspectiveTransform(np.asarray(points, np.float32).reshape(1, -1, 2), matrix)[0]


class Plane:
    def __init__(self, image, points, width, height, cube_reference=False):
        self.shape = image.shape[:2]
        self.points = np.asarray(points, np.float32)
        self.cube_reference = cube_reference
        self.width, self.height = (29., 29.) if cube_reference else (float(width), float(height))
        if not cube_reference and not all(35 <= n <= 3000 for n in (self.width, self.height)):
            raise ValueError("参考矩形的两条边必须在 35–3000 mm 范围内")
        p = self.points
        if p.shape != (4, 2) or not np.isfinite(p).all() or not cv2.isContourConvex(p):
            raise ValueError("请沿矩形依次点选原点、+X 角、对角和 +Y 角")
        if cv2.contourArea(p) < (1000 if cube_reference else 2500) or np.min(np.linalg.norm(p - np.roll(p, 1, axis=0), axis=1)) < 30:
            raise ValueError("参考区域太小，请让手机靠近一些后重新选四角")
        dst = np.float32([[0, 0], [self.width, 0], [self.width, self.height], [0, self.height]])
        self.to_mm = cv2.getPerspectiveTransform(p, dst)
        self.to_px = cv2.getPerspectiveTransform(dst, p)
        self.patches = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = self.shape
        x, y, rw, rh = cv2.boundingRect(p)
        margin = 3 * max(rw, rh) if cube_reference else 0
        self.crop = (max(0, x-margin), max(0, y-margin), min(w, x+rw+margin), min(h, y+rh+margin))
        self.margin_mm = 87 if cube_reference else 0
        for x, y in np.rint(p).astype(int):
            if not (22 <= x < w-22 and 22 <= y < h-22):
                raise ValueError("四个参考点要与画面边缘留出至少 22 像素")
            patch = gray[y-10:y+11, x-10:x+11].copy()
            if patch.std() < (6 if cube_reference else 12):
                raise ValueError("参考点缺少纹理，请使用清楚的深浅交界角点")
            self.patches.append((x, y, patch))

    def matches(self, image):
        if image.shape[:2] != self.shape:
            return False
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for x, y, patch in self.patches:
            search = gray[y-20:y+21, x-20:x+21]
            scores = cv2.matchTemplate(search, patch, cv2.TM_CCOEFF_NORMED)
            _, score, _, (u, v) = cv2.minMaxLoc(scores)
            if score < 0.76 or np.hypot(u-10, v-10) > 1.5:
                return False
        return True

    def in_workspace(self, xy):
        points = np.asarray(xy)
        return np.all(points >= -self.margin_mm) and np.all(points <= np.array([self.width, self.height]) + self.margin_mm)

    def cubes(self, polygons):
        candidates = []
        for polygon in polygons:
            if len(polygon) < 4:
                continue
            contour = project(polygon, self.to_mm)
            if not np.isfinite(contour).all():
                continue
            # Only use complete top faces inside the measured rectangle.
            if not self.in_workspace(contour):
                continue
            area = cv2.contourArea(contour)
            if not (0.65 * 29**2 < area < 1.35 * 29**2):
                continue
            hull = cv2.convexHull(contour)
            quad = cv2.approxPolyDP(hull, 0.035 * cv2.arcLength(hull, True), True).reshape(-1, 2)
            if len(quad) != 4 or abs(cv2.contourArea(quad) - area) > 0.13 * area:
                continue
            edges = np.roll(quad, -1, axis=0) - quad
            sides = np.linalg.norm(edges, axis=1)
            if np.max(abs(sides - 29)) > 4.5:
                continue
            unit = edges / sides[:, None]
            if np.max(abs(np.sum(unit * np.roll(unit, 1, axis=0), axis=1))) > 0.22:
                continue
            center = quad.mean(axis=0)
            if self.cube_reference and np.all((center >= 0) & (center <= 29)):
                continue
            # Nearest square (a cube has no unique heading beyond a quarter turn).
            angle = float(np.degrees(np.arctan2(edges[0, 1], edges[0, 0])) % 90)
            candidates.append((np.mean(abs(sides-29)), center, {
                "x_mm": round(float(center[0]), 2), "y_mm": round(float(center[1]), 2),
                "yaw_mod90_deg": round(angle, 1),
                "outline_px": project(quad, self.to_px).round(1).tolist(),
                "center_px": project([center], self.to_px)[0].round(1).tolist(),
            }))
        cubes = []
        for _, center, item in sorted(candidates, key=lambda c: c[0]):
            if all(np.linalg.norm(center - [c["x_mm"], c["y_mm"]]) > 14.5 for c in cubes):
                cubes.append(item)
        return cubes
