"""Filter calibrated bearings and metric depth independently for a rigid 29 mm cube."""

import cv2
import numpy as np

from .cuboid_pnp import CENTRED, HALF
from .workcell_tracking import FORBIDDEN, assign_centres

EDGE = 2 * HALF
PREDICT_S = 0.12
HOLD_S = 0.35
DEPTH_HOLD_S = 0.60


class Motion:
    """Constant-velocity Kalman filter for independent axes sharing the same noise."""

    def __init__(self, point, at, sigma, acceleration):
        self.point = np.asarray(point, dtype=float)
        self.velocity = np.zeros_like(self.point)
        self.at = at
        self.cov = np.diag([sigma**2, (12 * sigma) ** 2])
        self.acceleration = acceleration

    def forecast(self, at):
        return self.point + min(max(0, at - self.at), PREDICT_S) * self.velocity

    def update(self, point, at, sigma):
        dt = at - self.at
        transition = np.array([[1, dt], [0, 1]])
        cov = transition @ self.cov @ transition.T
        cov += self.acceleration**2 * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
        gain = cov[:, 0] / (cov[0, 0] + sigma**2)
        prediction = self.forecast(at)
        residual = point - prediction
        self.point = prediction + gain[0] * residual
        self.velocity += gain[1] * residual
        self.cov = cov - np.outer(gain, cov[0])
        self.at = at


class CubeTrack:
    def __init__(self, ident, observation, at):
        self.ident = ident
        self.color = observation["color"]
        x, y, z = observation["camera_xyz"]
        # Bearing times 1000, independent of the unreliable PnP range.
        self.bearing = Motion(1000 * np.array([x, y]) / z, at, 4.0, 390)
        self.depth = Motion(z, at, 5.0, 170)
        self.rotation = np.asarray(observation["camera_pose"])[:3, :3]
        self.good_depths = 1
        self.reacquire = None

    def hand_in_missing_outline(self, observation, image, k, dist):
        """Look for fingers in the part of a full-size silhouette the new proposal lost."""
        xyz = np.asarray(observation["camera_xyz"])
        expected = xyz * (self.depth.point / xyz[2])
        outline = cv2.projectPoints(CENTRED, cv2.Rodrigues(self.rotation)[0], expected, k, dist)[0]
        hull = cv2.convexHull(np.float32(outline))
        x, y, w, h = cv2.boundingRect(hull)
        left, top = max(0, x), max(0, y)
        crop = image[top : max(top, y + h), left : max(left, x + w)]
        if not crop.size:
            return False
        margin = np.zeros(crop.shape[:2], np.uint8)
        origin = np.array([left, top])
        cv2.fillConvexPoly(margin, np.int32(np.rint(hull - origin)), 1)
        full_area = cv2.countNonZero(margin)
        small = cv2.convexHull(np.float32(observation["pixels"]))
        cv2.fillConvexPoly(margin, np.int32(np.rint(small - origin)), 0)
        missing = cv2.countNonZero(margin)
        if missing < max(16, 0.12 * full_area):
            return False
        ycc = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
        skin = cv2.inRange(ycc, (28, 133, 76), (255, 182, 134))
        # A saturated red cube face can share a finger's red chrominance.
        sat = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[:, :, 1]
        return np.count_nonzero((margin != 0) & (skin != 0) & (sat < 190)) > 0.45 * missing

    def correct(self, observation, at, k, dist, image):
        x, y, z = observation["camera_xyz"]
        depth = float(self.depth.point)
        ray = 1000 * np.array([x, y]) / z
        clear = observation["confidence"] >= 0.60 and observation["color_support"] >= 0.55
        if clear and image is not None and z > depth * 1.07:
            clear = not self.hand_in_missing_outline(observation, image, k, dist)
        # A shrunken set of corners can have a low reprojection error.
        jump = abs(z - depth) > max(15.0, 0.12 * depth)
        reliable = clear and not jump
        if clear and jump:
            if (
                self.reacquire is None
                or at - self.reacquire[2] > 0.09
                or abs(z - self.reacquire[0]) > 0.10 * z
            ):
                self.reacquire = (z, at, at)
            else:
                base, start, _ = self.reacquire
                self.reacquire = ((base + z) / 2, start, at)
                if at - start >= 0.12:
                    # Several clear views can establish a genuine change in range.
                    self.depth = Motion(z, at, 5.0, 170)
                    reliable = True
        else:
            self.reacquire = None
        if reliable:
            if self.depth.at != at:
                self.depth.update(z, at, max(3.0, 0.018 * z))
            self.rotation = np.asarray(observation["camera_pose"])[:3, :3]
            self.good_depths += 1
        else:
            # A visible fragment also biases the image centre towards one side.
            self.bearing.velocity *= np.exp(-(at - self.bearing.at) / 0.15)
            prior = self.bearing.forecast(at)
            residual = ray - prior
            ray = prior + residual * min(
                1.0, (0.38 * EDGE * 1000 / depth) / max(np.linalg.norm(residual), 1e-6)
            )
        fraction = 0.045 if reliable else 0.10 if observation["color_support"] >= 0.40 else 0.18
        self.bearing.update(ray, at, fraction * EDGE * 1000 / depth)
        return "observed" if reliable else "depth_hold"

    def xyz(self, at):
        z = float(self.depth.point)
        last_xy = self.bearing.point * z / 1000
        move = (self.bearing.forecast(at) - self.bearing.point) * z / 1000
        return np.r_[last_xy + move * min(1.0, EDGE / max(np.linalg.norm(move), 1e-6)), z]

    def render(self, at, k, dist, source, index):
        xyz = self.xyz(at)
        pose = np.eye(4)
        pose[:3, 3] = xyz
        pose[:3, :3] = self.rotation
        pixels = cv2.projectPoints(CENTRED, cv2.Rodrigues(self.rotation)[0], xyz, k, dist)[0].reshape(8, 2)
        return {
            "track_id": self.ident,
            "color": self.color,
            "source": source,
            "measurement_index": index,
            "camera_xyz": xyz.tolist(),
            "camera_pose": pose.tolist(),
            "pixels": pixels.tolist(),
            "missing_ms": round(1000 * (at - self.bearing.at)),
            "depth_age_ms": round(1000 * (at - self.depth.at)),
        }


class RigidCubeTracker:
    def __init__(self):
        self.tracks = []
        self.next_id = 1
        self.last_at = -float("inf")

    def update(self, observations, at, k, dist, image=None):
        if at <= self.last_at or at - self.last_at > HOLD_S:
            self.tracks.clear()
        self.last_at = at
        self.tracks = [t for t in self.tracks if at - t.bearing.at <= HOLD_S]
        cost = np.full((len(self.tracks), len(observations)), FORBIDDEN)
        for i, t in enumerate(self.tracks):
            predicted = t.xyz(at)
            for j, obs in enumerate(observations):
                point = np.asarray(obs["camera_xyz"])
                if obs["color"] != t.color or not 0.50 < point[2] / predicted[2] < 2.0:
                    continue
                lateral = np.linalg.norm(point[:2] * predicted[2] / point[2] - predicted[:2])
                if lateral < EDGE * 1.65:
                    cost[i, j] = lateral + 0.08 * abs(point[2] - predicted[2])
        matches, ambiguous = assign_centres(cost)
        assigned = set(matches.values())
        result = []
        for i, t in enumerate(self.tracks):
            if i in assigned or at - t.depth.at > DEPTH_HOLD_S or t.good_depths < 3:
                continue
            if any(cost[i, j] < FORBIDDEN for j in ambiguous):
                continue
            source = "predicted" if at - t.bearing.at <= PREDICT_S else "held"
            result.append(t.render(at, k, dist, source, None))
        for j, obs in enumerate(observations):
            if j in ambiguous:
                continue
            if j in matches:
                t = self.tracks[matches[j]]
                source = t.correct(obs, at, k, dist, image)
            else:
                if obs["color_support"] < 0.55 or obs["confidence"] < 0.50:
                    continue
                t = CubeTrack(self.next_id, obs, at)
                self.next_id += 1
                self.tracks.append(t)
                source = "observed"
            if at - t.depth.at <= DEPTH_HOLD_S:
                result.append(t.render(at, k, dist, source, j))
        return result

    def compensate(self, poses, result_at):
        by_id = {t.ident: t for t in self.tracks}
        for p in poses:
            t = by_id[p["track_id"]]
            age = result_at - t.bearing.at
            if p["source"] == "observed" and t.good_depths >= 3 and 0 <= age <= PREDICT_S:
                z = float(t.depth.forecast(result_at))
                future = np.r_[t.bearing.forecast(result_at) * z / 1000, z]
                if np.linalg.norm(future - p["camera_xyz"]) <= 35:
                    p.update(lead_camera_xyz=future.tolist(), lead_ms=round(1000 * age))
