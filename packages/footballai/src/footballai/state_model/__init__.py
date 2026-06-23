"""Football state representation model for real-time video inference.

Public API:
    - ModelConfig            : Hyperparameter dataclass.
    - FootballStateModel     : End-to-end spatial + temporal + heads model.
    - load_state_from_checkpoint : Load plain PyTorch weights from a Lightning ckpt.
    - FootballStatePredictor : Sliding-window predictor that turns per-frame
                               detection records into probability readouts.
"""

from footballai.state_model.config import ModelConfig
from footballai.state_model.model import FootballStateModel
from footballai.state_model.checkpoint import load_state_from_checkpoint
from footballai.state_model.predictor import FootballStatePredictor

__all__ = [
    "ModelConfig",
    "FootballStateModel",
    "load_state_from_checkpoint",
    "FootballStatePredictor",
]
