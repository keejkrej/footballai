# FootballAI State Representation — Model Architecture

This document describes the neural network architecture used to pretrain a football game state representation from StatsBomb event data. It is intended as a reference you can share with collaborators or publish alongside the code.

## High-level overview

The model processes a **sequence of game states** and learns a compact, fixed-size representation of the match situation at each timestep. It is trained on three self-supervised / multi-task objectives:

1. **Pass receiver prediction** — where a pass will end and which teammate receives it within the next 5 seconds.
2. **Shot expected goals (xG)** — whether a shot will occur and its estimated xG value within the next 5 seconds.
3. **Turnover prediction** — whether possession will change via interception, ball recovery, or dispossession within the next 5 seconds.

The shared backbone (spatial encoder + temporal model) is the pretrained state representation that can be reused for real-time inference from video feeds.

```text
                    ┌──────────────────────────────────────────────┐
                    │  Input: sequence of football game states    │
                    │  shape: [B, T, N, F]                         │
                    │  B = batch, T = timesteps,                   │
                    │  N = entities (ball + up to 22 players),     │
                    │  F = 10 features per entity                  │
                    └────────────────────┬─────────────────────────┘
                                         │
                    ┌────────────────────▼─────────────────────────┐
                    │  EntityEmbedding + Distance-aware Transformer │
                    │  → permutation-invariant spatial encoder       │
                    │  output: [B, T, D]  D = 128                  │
                    └────────────────────┬─────────────────────────┘
                                         │
                    ┌────────────────────▼─────────────────────────┐
                    │  Bidirectional GRU (temporal backbone)         │
                    │  output: [B, T, H]  H = 256                  │
                    └────────────────────┬─────────────────────────┘
                                         │
           ┌─────────────────────────────┼─────────────────────────────┐
           │                             │                             │
           ▼                             ▼                             ▼
   ┌───────────────┐          ┌─────────────────┐          ┌─────────────────┐
   │  Pass head    │          │  Shot / xG head │          │  Turnover head  │
   │  - end_xy     │          │  - shot_prob    │          │  - binary prob  │
   │  - receiver   │          │  - xg_value     │          │                 │
   │    slot       │          │                 │          │                 │
   └───────────────┘          └─────────────────┘          └─────────────────┘
```

## Input representation

Each timestep is a **set of entities** (not a fixed grid or image). The ball is entity 0, followed by up to 22 players.

Per-entity feature vector `F = 10`:

| Index | Feature | Description |
|---|---|---|
| 0 | `x` | Pitch length coordinate normalized to `[-1, 1]` |
| 1 | `y` | Pitch width coordinate normalized to `[-1, 1]` |
| 2 | `vx` | Inferred velocity along x, in normalized units/s |
| 3 | `vy` | Inferred velocity along y, in normalized units/s |
| 4 | `team0` | One-hot: 1 if player belongs to reference (kickoff) team |
| 5 | `team1` | One-hot: 1 if player belongs to the opponent team |
| 6 | `position_id` | Learned role embedding index (goalkeeper, back, mid, forward, etc.) |
| 7 | `possession` | 1 if this player is the current ball controller |
| 8 | `ball` | 1 for the ball entity, 0 for players |
| 9 | `on_pitch` | Padding / validity flag; 0 for missing players |

Coordinate orientation: both halves are rotated so the team that kicked off the first half always attacks the `+x` direction. This removes the half-time direction flip from the model's job.

## Spatial encoder

File: `models/spatial_encoder.py`

### EntityEmbedding

Each entity is projected into a `D = 128` dimensional vector by summing four components:

1. **Continuous projection** of `[x, y, vx, vy, possession, ball, on_pitch]`.
2. **Position embedding** lookup on `position_id`.
3. **Entity-type embedding** lookup on the `ball` flag (player vs ball).
4. **Team projection** of the 2-dim team one-hot.

Output is layer-normalized and dropped out (`p = 0.1`).

Shape: `[B, N, F] → [B, N, D]`

### Distance-aware Transformer

A stack of `L = 4` Transformer encoder blocks with `H = 4` attention heads and `D = 128`.

Each block adds a **learned pairwise distance bias** to the attention scores:

```text
scores_ij = (q_i · k_j) / sqrt(D/H) + mlp(||coord_i - coord_j||)
```

This lets attention explicitly attend to nearby players / the ball, which is a strong football prior. Padding is masked with `-inf` so missing players do not participate.

Shape: `[B, N, D] → [B, N, D]`

### Pooling

After the Transformer stack, the variable-length entity set is reduced to a fixed-size state vector by **mean pooling over valid entities** (`on_pitch == 1`). The ball is always valid. No CLS token is used.

Shape: `[B, N, D] → [B, D]`

**Total spatial-encoder parameters:** ~400 K

| Component | Parameters |
|---|---|
| EntityEmbedding | ~12 K |
| 4 Transformer blocks | ~370 K |
| Output norm | ~256 |

## Temporal model

File: `models/temporal_model.py`

A 2-layer bidirectional GRU processes the sequence of state vectors.

- Input: `[B, T, D]` where `D = 128`
- Hidden dim per direction: `128`
- Output: `[B, T, H]` where `H = 256` (concatenated forward + backward)
- Dropout between layers: `p = 0.1`

Padded sequences are packed/unpacked for correct bidirectional handling.

**Temporal model parameters:** ~300 K

## Pretraining heads

File: `models/pretrain_heads.py`

Three lightweight MLP heads sit on top of the temporal hidden state `h_t`.

### Pass receiver head

```text
h_t  [B, T, H=256]
  ↓  MLP(H → 128 → 128)
  ├─→ Linear(128, 2)    → end_xy     [B, T, 2]   (team-relative pitch coords)
  └─→ Linear(128, 22)   → receiver_logits [B, T, 22]
```

### Shot / xG head

```text
h_t  [B, T, H=256]
  ↓  MLP(H → 128)
  ├─→ Linear(128, 1) + sigmoid  → shot_prob  [B, T, 1]
  └─→ Linear(128, 16) + ReLU + Linear(16, 1) + sigmoid → xg [B, T, 1]
```

### Turnover head

```text
h_t  [B, T, H=256]
  ↓  MLP(H → 128 → 1) + sigmoid
  → turnover_prob  [B, T, 1]
```

**Head parameters:** ~80 K total

## Loss function

File: `models/lightning_module.py`

The model is trained end-to-end on a weighted sum of masked losses:

| Task | Loss | Masking |
|---|---|---|
| Pass end location | MSE | only timesteps with a pass in the 5 s horizon |
| Pass receiver slot | cross-entropy | only timesteps with a pass in the 5 s horizon, slot != -1 |
| Shot probability | BCE | all valid timesteps |
| xG value | MSE | only timesteps where a shot occurs |
| Turnover | BCE | all valid timesteps |

Total loss:

```text
L = w_pass * (L_pass_xy + L_pass_slot) + w_shot * (L_shot_prob + L_shot_xg) + w_turnover * L_turnover
```

Default weights: `w_pass = w_shot = w_turnover = 1.0`.

## Output interface for downstream use

After pretraining, the backbone can be extracted and used for real-time inference:

```python
model = FootballStateModel(config)
outputs = model(entity_features, seq_len=lengths)

state_vectors      = outputs["state"]            # [B, T, 128]
temporal_hidden    = outputs["temporal_hidden"]  # [B, T, 256]
pass_predictions   = outputs["pass_receiver"]    # dict with end_xy and receiver_logits
shot_predictions   = outputs["shot_xg"]        # dict with shot_prob and xg
turnover_prediction = outputs["turnover"]        # dict with turnover_prob
```

For streaming video inference, replace the event-based input builder with one that produces the same `[N, F]` entity feature tensor from player/ball detections, and run the temporal model in forward-only mode.

## Model size summary

| Component | Parameters | Forward pass (batch 64, T=50, N=23) |
|---|---|---|
| Spatial encoder | ~400 K | `[64, 50, 23, 10] → [64, 50, 128]` |
| Temporal model | ~300 K | `[64, 50, 128] → [64, 50, 256]` |
| Pretraining heads | ~80 K | per-task predictions |
| **Total** | **~780 K** | small, fits easily on a single GPU |

## Key design decisions

1. **Set-based input.** Football is naturally a set of players + ball. Using a Transformer instead of a flattened vector makes the model permutation-invariant and robust to missing detections.
2. **Distance bias in attention.** Explicitly conditioning attention on player-to-player distance gives the model a strong geometric prior without hand-coded rules.
3. **Bidirectional GRU for pretraining.** The full sequence is known during pretraining, so both directions improve the representation. For real-time deployment the same weights can be used in a causal forward-only GRU.
4. **Multi-task pretraining.** Pass, shot, and turnover objectives force the state vector to encode ball movement, scoring opportunity, and defensive pressure simultaneously.
5. **Small model.** Less than 1 M parameters allows fast inference on CPU/GPU and easy deployment from video feeds.

## Files map

| File | Responsibility |
|---|---|
| `config.py` | Hyperparameters and derived dimensions |
| `models/spatial_encoder.py` | EntityEmbedding + Distance-aware Transformer |
| `models/temporal_model.py` | Bidirectional GRU backbone |
| `models/pretrain_heads.py` | Pass, shot/xG, turnover heads |
| `models/__init__.py` | `FootballStateModel` end-to-end wrapper |
| `models/lightning_module.py` | PyTorch Lightning module with multi-task loss |
| `data/state_builder.py` | StatsBomb event → `[N, F]` tensor parser |
| `data/preprocessed_dataset.py` | Fast `.pt` tensor dataset |
| `train.py` | PyTorch Lightning training entrypoint |
