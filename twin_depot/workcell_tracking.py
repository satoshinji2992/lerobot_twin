"""Associate coloured cube centres in one workcell coordinate system (mm, seconds)."""

from dataclasses import dataclass, field
from math import isfinite, sqrt

import numpy as np
from scipy.optimize import linear_sum_assignment

HISTORY_S = 1.2
OBS_VARIANCE = 2.5**2  # Filter tuning, not a statement of measurement accuracy.
ACCEL_NOISE = 65.0
MAX_RESIDUAL_MM = 48.0
DUPLICATE_MM = 6.0  # Overlapping mask fits for a 29 mm cube.
AMBIGUITY_MM = 3.0
FORBIDDEN = 1e7


@dataclass
class _Track:
    ident: int
    color: str
    xyz: np.ndarray
    seen: float
    confirmed: bool
    samples: int = 1
    motion_span: float = 0.0
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    covariance: np.ndarray = field(default_factory=lambda: np.diag([OBS_VARIANCE, 140.0**2]))

    def forecast(self, at):
        # All three axes share the same position/velocity process and noise assumptions.
        dt = at - self.seen
        motion = np.array([[1.0, dt], [0.0, 1.0]])
        noise = ACCEL_NOISE**2 * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
        return self.xyz + dt * self.velocity, motion @ self.covariance @ motion.T + noise

    def correct(self, xyz, forecast, at):
        predicted, cov = forecast
        residual = xyz - predicted
        dt = at - self.seen
        if dt > 0.18 or np.linalg.norm(residual) > 12:
            self.samples, self.motion_span = 1, 0.0
        else:
            self.samples += 1
            self.motion_span += dt
        gain = cov[:, 0] / (cov[0, 0] + OBS_VARIANCE)
        self.xyz = predicted + gain[0] * residual
        self.velocity += gain[1] * residual
        self.covariance = cov - np.outer(gain, cov[0])
        self.seen = at
        self.confirmed = True


def _observations(rows):
    """Keep the best-fitting instance when overlapping masks yield the same cube."""
    candidates = []
    for index, row in enumerate(rows):
        if not isinstance(row.get("color"), str):
            continue
        try:
            xyz = np.asarray(row["xyz"], dtype=float)
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            continue
        error = row.get("error_px")
        quality = float(error) if isinstance(error, (int, float)) and isfinite(error) else float("inf")
        candidates.append((quality, index, row, xyz))
    kept = []
    for _, index, row, xyz in sorted(candidates, key=lambda candidate: candidate[:2]):
        if any(
            row["color"] == other["color"] and np.linalg.norm(xyz - point) < DUPLICATE_MM
            for _, other, point in kept
        ):
            continue
        kept.append((index, row, xyz))
    return sorted(kept, key=lambda item: item[0])


def assign_centres(cost):
    """Return unique matches and withhold observations with near-equivalent assignments."""
    indices, detections = linear_sum_assignment(cost)
    best = cost[indices, detections].sum()
    matched, ambiguous = {}, set()
    for i, j in zip(indices, detections, strict=True):
        distance = cost[i, j]
        if distance == FORBIDDEN:
            continue
        # Test the best alternative assignment before crossing two identities. A
        # missing association costs FORBIDDEN, so cardinality takes priority.
        if np.count_nonzero(cost[:, j] < FORBIDDEN) > 1 or np.count_nonzero(cost[i] < FORBIDDEN) > 1:
            cost[i, j] = FORBIDDEN
            alt_i, alt_j = linear_sum_assignment(cost)
            alternative = cost[alt_i, alt_j].sum()
            cost[i, j] = distance
            if alternative <= best + AMBIGUITY_MM:
                ambiguous.add(j)
                # An unmatched detection that could have taken this same slot is
                # ambiguous too. Do not give it a new identity in this frame.
                ambiguous.update(
                    aj for ai, aj in zip(alt_i, alt_j, strict=True) if ai == i and cost[ai, aj] < FORBIDDEN
                )
                continue
        matched[j] = i
    return matched, ambiguous


class Tracker:
    """Tracks per-camera-session, per-coordinate-frame. Pass monotonic capture times in seconds."""

    def __init__(self):
        self._tracks = []
        self._next_id = 1
        self._last_at = None

    def update(self, rows, at):
        """Return current measurements in input order, plus last sightings of missing confirmed tracks."""
        if not isfinite(at):
            raise ValueError("capture time must be finite")
        if self._last_at is not None and at <= self._last_at:
            return [], []  # A duplicate or delayed frame cannot supersede newer evidence.
        self._last_at = at
        self._tracks = [track for track in self._tracks if at - track.seen <= HISTORY_S]
        observations = _observations(rows)
        forecasts = [track.forecast(at) for track in self._tracks]
        cost = np.full((len(self._tracks), len(observations)), FORBIDDEN)
        for i, (track, (predicted, cov)) in enumerate(zip(self._tracks, forecasts, strict=True)):
            gate = min(MAX_RESIDUAL_MM, 3.5 * sqrt(cov[0, 0] + OBS_VARIANCE))
            for j, (_, row, point) in enumerate(observations):
                if track.color == row["color"]:
                    distance = np.linalg.norm(point - predicted)
                    if distance <= gate:
                        cost[i, j] = distance
        matches, ambiguous = assign_centres(cost)
        known_colors = {track.color for track in self._tracks}
        live = []
        for j, (_, row, point) in enumerate(observations):
            if j in ambiguous:
                continue
            if j in matches:
                track = self._tracks[matches[j]]
                track.correct(point, forecasts[matches[j]], at)
            else:
                # A jump far from an existing track can be a misfit or a second
                # same-colour cube. Require another observation before publishing it.
                track = _Track(
                    self._next_id, row["color"], point.copy(), at, row["color"] not in known_colors
                )
                self._next_id += 1
                self._tracks.append(track)
            if track.confirmed:
                live.append(
                    dict(row, raw_xyz=point.tolist(), xyz=track.xyz.round(3).tolist(), track_id=track.ident)
                )
        self._tracks = [track for track in self._tracks if track.confirmed or track.seen == at]
        history = [
            {
                "track_id": track.ident,
                "color": track.color,
                "xyz": track.xyz.round(3).tolist(),
                "missing_ms": round((at - track.seen) * 1000),
            }
            for track in self._tracks
            if track.confirmed and track.seen < at
        ]
        return live, history

    def compensate(self, rows, history, result_at):
        """Short constant-velocity extrapolation to server result time; keep observations intact."""
        tracks = {track.ident: track for track in self._tracks}
        for row in [*rows, *history]:
            track = tracks[row["track_id"]]
            dt = result_at - track.seen
            if track.samples < 3 or track.motion_span < 0.08 or not 0 <= dt <= 0.12:
                continue
            point, cov = track.forecast(result_at)
            if cov[0, 0] > 8**2 or np.linalg.norm(point - track.xyz) > 35:
                continue
            row.update(lead_xyz=point.round(2).tolist(), lead_ms=round(1000 * dt))
