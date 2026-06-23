"""End-to-end football state model wrapper."""

import torch
import torch.nn as nn

from footballai.state_model.config import ModelConfig
from footballai.state_model.spatial_encoder import FootballStateEncoder
from footballai.state_model.temporal_model import FootballStateTemporalModel
from footballai.state_model.pretrain_heads import FootballPretrainHeads


class FootballStateModel(nn.Module):
    """End-to-end model: spatial encoder -> temporal model -> multi-task heads.

    Input:
        entity_features: [B, T, N, F] where N is variable number of entities
                         (players + ball) and F = config.raw_feature_dim.
        seq_len: optional [B] actual sequence lengths for packing.

    Output:
        Nested dict of task predictions plus intermediate "state" and "temporal_hidden".
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.spatial_encoder = FootballStateEncoder(config)
        self.temporal_model = FootballStateTemporalModel(config)
        self.pretrain_heads = FootballPretrainHeads(config)

    def forward(
        self,
        entity_features: torch.Tensor,
        seq_len: torch.Tensor = None,
    ) -> dict:
        """
        Args:
            entity_features: [B, T, N, F]
            seq_len:         [B]
        Returns:
            {
                "state":           [B, T, D]  spatial state vectors,
                "temporal_hidden": [B, T, H]  temporal hidden states,
                "pass_receiver":   {...},
                "shot_xg":         {...},
                "turnover":        {...},
            }
        """
        B, T, N, F = entity_features.shape
        flat = entity_features.view(B * T, N, F)
        state, entity_features = self.spatial_encoder(flat)  # [B*T, D], [B*T, N, D]
        state = state.view(B, T, -1)                          # [B, T, D]
        entity_features = entity_features.view(B, T, N, -1)   # [B, T, N, D]

        temporal_hidden = self.temporal_model(state, seq_len=seq_len)  # [B, T, H]
        predictions = self.pretrain_heads(temporal_hidden, entity_features=entity_features)

        return {
            "state": state,
            "temporal_hidden": temporal_hidden,
            **predictions,
        }
