"""Checkpoint loading helpers.

The pretraining checkpoints are PyTorch Lightning checkpoints. This module strips
out the plain model weights and loads them into a FootballStateModel.
"""

from pathlib import Path
from typing import Union

import torch

from footballai.state_model.config import ModelConfig
from footballai.state_model.model import FootballStateModel


_CKPT_PREFIX = "model."


def load_state_from_checkpoint(
    checkpoint_path: Union[str, Path],
    config: ModelConfig | None = None,
    device: str = "cpu",
    strict: bool = True,
) -> FootballStateModel:
    """Load a FootballStateModel from a PyTorch Lightning checkpoint.

    Args:
        checkpoint_path: Path to a `.ckpt` file produced by the pretraining
            LightningModule. The checkpoint is expected to contain a `state_dict`
            whose keys are prefixed with ``model.``.
        config: ModelConfig to use. If None, a default config is created.
        device: Device to move the loaded model to.
        strict: Whether to strictly enforce that keys match.

    Returns:
        A FootballStateModel with loaded weights, in eval mode.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    config = config or ModelConfig()
    model = FootballStateModel(config)

    ckpt = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state_dict = ckpt.get("state_dict", ckpt)

    # Strip the LightningModule's "model." prefix.
    stripped: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith(_CKPT_PREFIX):
            stripped[key[len(_CKPT_PREFIX) :]] = value
        else:
            stripped[key] = value

    missing, unexpected = model.load_state_dict(stripped, strict=strict)
    if missing:
        raise RuntimeError(f"Missing keys after loading checkpoint: {missing}")
    if unexpected:
        # Keep going for unexpected keys but warn so the caller knows.
        print(f"Warning: unexpected checkpoint keys ignored: {unexpected[:5]}")

    model.to(device)
    model.eval()
    return model
