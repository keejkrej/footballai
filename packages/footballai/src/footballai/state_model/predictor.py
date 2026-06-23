"""Sliding-window predictor that emits real-time probability readouts.

Maintains a fixed-size buffer of recent entity tensors, runs the pretrained
FootballStateModel over the window, and returns the last timestep's predictions.
Because the pretraining used a bidirectional GRU, running it over a sliding
window lets us reuse the learned weights unchanged.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

from footballai.state_model.config import ModelConfig
from footballai.state_model.model import FootballStateModel
from footballai.state_model.checkpoint import load_state_from_checkpoint
from footballai.state_model.state_builder import EntityStateBuilder


class FootballStatePredictor:
    """Real-time wrapper around FootballStateModel.

    Args:
        checkpoint_path: Path to a pretraining Lightning checkpoint.
        config: ModelConfig. If None, the default config is used.
        device: Torch device to run inference on.
        fps: Expected frame rate for velocity estimation.
        strict: Whether to strictly enforce checkpoint key matching.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        config: ModelConfig | None = None,
        device: str = "cuda",
        fps: float = 25.0,
        strict: bool = True,
    ):
        self.config = config or ModelConfig()
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.model = load_state_from_checkpoint(
            checkpoint_path,
            config=self.config,
            device=str(self.device),
            strict=strict,
        )
        self.builder = EntityStateBuilder(fps=fps)
        self._window: deque[np.ndarray] = deque(maxlen=self.config.seq_len)
        self._empty_state = np.zeros((self.config.max_entities, self.config.raw_feature_dim), dtype=np.float32)

    def reset(self) -> None:
        """Clear the sliding window and velocity history."""
        self._window.clear()
        self.builder.reset()

    def _records_to_tensor(
        self,
        records: list[dict[str, Any]],
        ball_holder_track_id: int | None = None,
    ) -> np.ndarray:
        """Build a [23, 10] tensor for this frame."""
        return self.builder.build(records, ball_holder_track_id=ball_holder_track_id)

    @torch.no_grad()
    def predict(
        self,
        records: list[dict[str, Any]],
        ball_holder_track_id: int | None = None,
    ) -> dict[str, Any]:
        """Add one frame and return the latest-frame probability readouts.

        Args:
            records: detection rows for the current frame.
            ball_holder_track_id: track_id of the player in possession.

        Returns:
            Dict with float probability values and a small state vector:
                {
                    "shot_prob": float,
                    "xg": float,
                    "turnover_prob": float,
                    "top_receiver_slot": int,
                    "top_receiver_prob": float,
                    "pass_end_x": float,
                    "pass_end_y": float,
                    "state_vector": list[float],
                }
        """
        entity_state = self._records_to_tensor(records, ball_holder_track_id)
        self._window.append(entity_state)

        # Stack window; if shorter than seq_len, pad at the beginning with zeros.
        window_len = len(self._window)
        if window_len == 0:
            return self._empty_prediction()

        states = list(self._window)
        if window_len < self.config.seq_len:
            pad_count = self.config.seq_len - window_len
            states = [self._empty_state.copy() for _ in range(pad_count)] + states

        tensor = torch.from_numpy(np.stack(states, axis=0)).unsqueeze(0).to(self.device)  # [1, T, N, F]
        outputs = self.model(tensor)  # no packing needed for full window

        # Read last timestep. Padding is always at the beginning, so the latest
        # real frame is always at index seq_len - 1.
        last_idx = self.config.seq_len - 1
        pass_logits = outputs["pass_receiver"]["receiver_logits"][0, last_idx]  # [slots]
        end_xy = outputs["pass_receiver"]["end_xy"][0, last_idx]  # [2]
        shot_prob = float(torch.sigmoid(outputs["shot_xg"]["shot_logits"][0, last_idx]).item())
        xg = float(outputs["shot_xg"]["xg"][0, last_idx].item())
        turnover_prob = float(outputs["turnover"]["turnover_prob"][0, last_idx].item())
        state_vector = outputs["state"][0, last_idx].cpu().numpy().tolist()

        # Receiver probabilities among the 22 outfield player slots.
        receiver_probs = torch.softmax(pass_logits, dim=-1).cpu().numpy()
        top_slot = int(receiver_probs.argmax())
        top_prob = float(receiver_probs[top_slot])

        return {
            "shot_prob": round(shot_prob, 4),
            "xg": round(xg, 4),
            "turnover_prob": round(turnover_prob, 4),
            "top_receiver_slot": top_slot,
            "top_receiver_prob": round(top_prob, 4),
            "pass_end_x": round(float(end_xy[0].item()), 4),
            "pass_end_y": round(float(end_xy[1].item()), 4),
            "state_vector": state_vector,
        }

    def _empty_prediction(self) -> dict[str, Any]:
        return {
            "shot_prob": 0.0,
            "xg": 0.0,
            "turnover_prob": 0.0,
            "top_receiver_slot": -1,
            "top_receiver_prob": 0.0,
            "pass_end_x": 0.0,
            "pass_end_y": 0.0,
            "state_vector": [],
        }
