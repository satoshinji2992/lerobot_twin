"""Constant-velocity Kalman tracks in the surveyed XY plane (mm, seconds)."""

from collections import deque

import numpy as np
from scipy.optimize import linear_sum_assignment


OBS_VAR = 1.2**2  # assumed per-axis detection noise, mm²; not a calibration certificate
ACCEL_NOISE = 35.0  # continuous white acceleration, mm/s^(3/2)
PREDICT_S = 0.35
MAX_SIGMA_MM = 5.0
ASSOC_S = 1.5  # low-rate cameras can still re-observe a stationary cube


class Track:
    def __init__(self, ident, xy, stamp):
        self.ident = ident
        self.xy = xy.copy()
        self.velocity = np.zeros(2)
        # Identical position/velocity covariance in X and Y.
        self.cov = np.diag([OBS_VAR, 75.0**2])
        self.seen = stamp
        self.count = 1
        self.recent = deque(maxlen=64)

    def predict(self, dt):
        self.xy += self.velocity * dt
        f = np.array([[1, dt], [0, 1]])
        q = ACCEL_NOISE**2 * np.array([[dt**3/3, dt**2/2], [dt**2/2, dt]])
        self.cov = f @ self.cov @ f.T + q

    def correct(self, xy, stamp):
        residual = xy - self.xy
        gain = self.cov[:, 0] / (self.cov[0, 0] + OBS_VAR)
        self.xy += gain[0] * residual
        self.velocity += gain[1] * residual
        self.cov -= np.outer(gain, self.cov[0].copy())
        self.seen = stamp
        self.count += 1

    def observation(self, xy, stamp):
        # Lock eligibility uses consecutive real observations, independent of smoothing.
        while self.recent and stamp-self.recent[0][0] > 2 * ASSOC_S:
            self.recent.popleft()
        # Keep the contiguous suffix of samples near this observation.
        for i in range(len(self.recent)-1, -1, -1):
            if np.linalg.norm(xy-self.recent[i][1]) > 1.5:
                for _ in range(i+1):
                    self.recent.popleft()
                break
        self.recent.append((stamp, xy))
        return dict(self.estimate(), stable=len(self.recent) >= 3 and stamp-self.recent[0][0] >= 0.35)

    def estimate(self):
        return dict(track_id=self.ident, fused_x_mm=round(float(self.xy[0]), 2),
                    fused_y_mm=round(float(self.xy[1]), 2), vx_mm_s=round(float(self.velocity[0]), 2),
                    vy_mm_s=round(float(self.velocity[1]), 2), sigma_mm=round(float(np.sqrt(self.cov[0, 0])), 2))


class CubeTracker:
    def __init__(self):
        self.tracks = []
        self.next_id = 1
        self.stamp = None

    def reset(self):
        self.tracks.clear()
        self.stamp = None

    def update(self, cubes, stamp):
        """Annotate visible detections and return separate hypotheses for missing cubes."""
        if self.stamp is not None and (stamp < self.stamp or stamp-self.stamp > ASSOC_S):
            self.reset()
        dt = 0 if self.stamp is None else stamp-self.stamp
        self.stamp = stamp
        self.tracks = [t for t in self.tracks if stamp-t.seen <= ASSOC_S]
        points = np.array([[c["x_mm"], c["y_mm"]] for c in cubes], dtype=float).reshape(-1, 2)
        costs = np.full((len(self.tracks), len(points)), 1e6)
        for i, track in enumerate(self.tracks):
            track.predict(dt)
            distances = np.linalg.norm(points - track.xy, axis=1)
            variance = track.cov[0, 0] + OBS_VAR
            allowed = distances <= min(25, 3.5 * np.sqrt(variance))
            costs[i, allowed] = distances[allowed]**2 / variance

        used, seen = set(), set()
        for i, j in zip(*linear_sum_assignment(costs), strict=True):
            if costs[i, j] >= 1e6:
                continue
            track = self.tracks[i]
            track.correct(points[j], stamp)
            cubes[j].update(track.observation(points[j], stamp))
            seen.add(track.ident)
            used.add(j)
        predictions = []
        for track in self.tracks:
            if track.ident not in seen:
                track.recent.clear()
                age = stamp-track.seen
                if track.count >= 3 and age <= PREDICT_S and track.cov[0, 0] <= MAX_SIGMA_MM**2:
                    predictions.append(dict(track.estimate(), missing_s=age))
        for j, xy in enumerate(points):
            if j not in used:
                track = Track(self.next_id, xy, stamp)
                self.next_id += 1
                self.tracks.append(track)
                cubes[j].update(track.observation(xy, stamp))
        return predictions
