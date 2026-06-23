"""Convert live video detections into the entity feature tensor the model expects.

Each timestep is represented as a [N, F] tensor where:
    row 0     = ball
    rows 1-22 = outfield players (team0 slots 1-11, team1 slots 12-22)

Per-entity feature layout (F=10):
    [0] x          pitch x, normalized to [-1, 1] on a 120m pitch
    [1] y          pitch y, normalized to [-1, 1] on an 80m pitch
    [2] vx         estimated velocity in normalized units per second
    [3] vy         estimated velocity in normalized units per second
    [4] team0      one-hot team flag
    [5] team1      one-hot team flag
    [6] position_id learned role id (0 because video has no roles)
    [7] possession 1 if this player is the current ball carrier
    [8] ball       1 for the ball entity
    [9] on_pitch   1 if the entity has valid pitch coordinates

Team orientation: team0 is dynamically defined as the team with the lower
mean pitch x (attacking left-to-right / +x). This mirrors the pretraining
orientation without needing the actual kickoff team.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

PITCH_LENGTH_M = 120.0
PITCH_WIDTH_M = 80.0
N_SLOTS = 22
N_ENTITIES = 23
RAW_FEATURE_DIM = 10

# Feature indices, aligned with spatial_encoder.py.
IDX_X, IDX_Y, IDX_VX, IDX_VY = 0, 1, 2, 3
IDX_TEAM0, IDX_TEAM1 = 4, 5
IDX_POSITION = 6
IDX_POSS = 7
IDX_BALL = 8
IDX_ON_PITCH = 9


def _cm_to_norm(x_cm: float | None, y_cm: float | None) -> tuple[float, float]:
    """Convert centimetre pitch coordinates to [-1, 1] normalized coordinates."""
    if x_cm is None or y_cm is None or not (np.isfinite(x_cm) and np.isfinite(y_cm)):
        return 0.0, 0.0
    x_m = x_cm / 100.0
    y_m = y_cm / 100.0
    x_norm = 2.0 * x_m / PITCH_LENGTH_M - 1.0
    y_norm = 2.0 * y_m / PITCH_WIDTH_M - 1.0
    return float(x_norm), float(y_norm)


def _has_pitch(record: dict[str, Any]) -> bool:
    px = record.get("pitch_x_cm")
    py = record.get("pitch_y_cm")
    return px is not None and py is not None and np.isfinite(px) and np.isfinite(py)


class EntityStateBuilder:
    """Stateful builder that tracks velocities and emits one [N, F] tensor per frame."""

    def __init__(
        self,
        max_speed_mps: float = 12.0,
        fps: float = 25.0,
        velocity_history: int = 3,
    ):
        self.max_speed_mps = max_speed_mps
        self.fps = max(fps, 1.0)
        self.velocity_history = velocity_history

        # Track recent normalised positions per track_id for velocity smoothing.
        self._history: dict[int, deque[tuple[float, float]]] = {}
        self._last_ball: tuple[float, float] | None = None

    def reset(self) -> None:
        self._history.clear()
        self._last_ball = None

    def _smoothed_velocity(
        self,
        track_id: int,
        current: tuple[float, float],
        dt: float,
    ) -> tuple[float, float]:
        """Return velocity in normalised units per second, smoothed over history."""
        hist = self._history.setdefault(track_id, deque(maxlen=self.velocity_history))
        hist.append(current)

        if len(hist) < 2 or dt <= 0:
            return 0.0, 0.0

        # Use the oldest point in the buffer for a stable velocity estimate.
        oldest = hist[0]
        dx = current[0] - oldest[0]
        dy = current[1] - oldest[1]
        # Effective dt over the buffer in seconds.
        effective_dt = dt * (len(hist) - 1)
        vx = dx / effective_dt
        vy = dy / effective_dt

        # Clamp to plausible sprint speed in normalised units/s.
        max_speed_norm = 2.0 * self.max_speed_mps / PITCH_LENGTH_M
        speed = np.hypot(vx, vy)
        if speed > max_speed_norm and speed > 0:
            scale = max_speed_norm / speed
            vx *= scale
            vy *= scale
        return float(vx), float(vy)

    def build(
        self,
        records: list[dict[str, Any]],
        ball_holder_track_id: int | None = None,
        dt: float | None = None,
    ) -> np.ndarray:
        """Build a [23, 10] entity feature tensor from one frame's detection records.

        Args:
            records: detection rows from _detections_to_records.
            ball_holder_track_id: track_id of the player currently controlling the
                ball, if known.
            dt: seconds since the previous processed frame. If None, 1/fps is used.

        Returns:
            ndarray of shape [23, 10] (float32).
        """
        dt = dt if dt is not None and dt > 0 else 1.0 / self.fps

        # Separate players/goalkeepers with valid pitch coords from the rest.
        player_records = [
            r for r in records
            if r["class_name"] in {"player", "goalkeeper"} and _has_pitch(r)
        ]

        # Ball record(s): prefer the one with highest confidence.
        ball_records = [r for r in records if r["class_name"] == "ball" and _has_pitch(r)]
        ball_record = max(ball_records, key=lambda r: r.get("confidence", 0.0)) if ball_records else None

        # Dynamic team orientation: the team with lower mean x is team0.
        team_means: dict[int, list[float]] = {0: [], 1: []}
        for r in player_records:
            tid = int(r.get("team_id", -1))
            if tid in team_means:
                team_means[tid].append(float(r["pitch_x_cm"]))

        mean_x_0 = np.mean(team_means[0]) if team_means[0] else float("inf")
        mean_x_1 = np.mean(team_means[1]) if team_means[1] else float("inf")

        # If one team is missing, keep existing mapping (0 stays 0, 1 stays 1).
        if np.isfinite(mean_x_0) and np.isfinite(mean_x_1):
            # team0 is the left-side (lower x) team.
            left_team = 0 if mean_x_0 < mean_x_1 else 1
        else:
            left_team = 0

        # Map original team ids to oriented team ids.
        # If left_team == 0, original 0 stays 0, original 1 stays 1.
        # If left_team == 1, we flip: original 1 becomes team0, original 0 becomes team1.
        def oriented_team(original_team: int) -> int:
            if left_team == 0:
                return int(original_team)
            return 1 - int(original_team)

        # Allocate slots per team. To keep velocities stable, sort by track_id.
        team_players: dict[int, list[dict]] = {0: [], 1: []}
        for r in player_records:
            tid = int(r.get("team_id", -1))
            if tid not in {0, 1}:
                continue
            oteam = oriented_team(tid)
            team_players[oteam].append(r)

        for team in (0, 1):
            team_players[team].sort(key=lambda r: int(r.get("track_id", 0)))
            team_players[team] = team_players[team][:11]

        # Build player rows.
        player_tensor = np.zeros((N_SLOTS, RAW_FEATURE_DIM), dtype=np.float32)
        player_mask = np.zeros(N_SLOTS, dtype=np.float32)

        slot = 0
        for team in (0, 1):
            for r in team_players[team]:
                x_norm, y_norm = _cm_to_norm(r.get("pitch_x_cm"), r.get("pitch_y_cm"))
                tid = int(r.get("track_id", -1))
                vx, vy = self._smoothed_velocity(tid, (x_norm, y_norm), dt)
                is_poss = 1 if ball_holder_track_id is not None and int(tid) == int(ball_holder_track_id) else 0

                player_tensor[slot] = [
                    x_norm,
                    y_norm,
                    vx,
                    vy,
                    1.0 if team == 0 else 0.0,
                    1.0 if team == 1 else 0.0,
                    0.0,  # position_id unknown from video
                    float(is_poss),
                    0.0,  # not ball
                    1.0,  # on_pitch
                ]
                player_mask[slot] = 1.0
                slot += 1

        # Build ball row.
        ball_row = np.zeros(RAW_FEATURE_DIM, dtype=np.float32)
        if ball_record is not None:
            x_norm, y_norm = _cm_to_norm(
                ball_record.get("pitch_x_cm"), ball_record.get("pitch_y_cm")
            )
            vx, vy = 0.0, 0.0
            if self._last_ball is not None:
                vx = (x_norm - self._last_ball[0]) / dt
                vy = (y_norm - self._last_ball[1]) / dt
                max_speed_norm = 2.0 * self.max_speed_mps / PITCH_LENGTH_M
                speed = np.hypot(vx, vy)
                if speed > max_speed_norm and speed > 0:
                    scale = max_speed_norm / speed
                    vx *= scale
                    vy *= scale
            self._last_ball = (x_norm, y_norm)
            ball_row[IDX_X] = x_norm
            ball_row[IDX_Y] = y_norm
            ball_row[IDX_VX] = float(vx)
            ball_row[IDX_VY] = float(vy)
            ball_row[IDX_BALL] = 1.0
            ball_row[IDX_ON_PITCH] = 1.0
        else:
            self._last_ball = None

        state = np.concatenate([ball_row[None, :], player_tensor], axis=0)
        return state


FootballStateBuilder = EntityStateBuilder  # public alias
