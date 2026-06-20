#!/usr/bin/env bash
# Full training run using ALL preprocessed matches.
# Run `python preprocess.py --out_dir ./data/processed` first.

set -e
source .venv/bin/activate

PROCESSED_DIR="./data/processed"
CHECKPOINT_DIR="./checkpoints"
LOG_DIR="./runs"

mkdir -p "$CHECKPOINT_DIR" "$LOG_DIR"

if [ ! -d "$PROCESSED_DIR" ] || [ -z "$(ls -A "$PROCESSED_DIR" 2>/dev/null)" ]; then
    echo "ERROR: preprocessed data not found at $PROCESSED_DIR"
    echo "Run first: python preprocess.py --out_dir $PROCESSED_DIR --num_workers 8"
    exit 1
fi

echo "=== Full training on $PROCESSED_DIR ==="
python train.py \
  --processed_dir "$PROCESSED_DIR" \
  --epochs 50 \
  --batch_size 128 \
  --seq_len 50 \
  --seq_stride 25 \
  --horizon 5.0 \
  --val_ratio 0.15 \
  --num_workers 8 \
  --lr 1e-3 \
  --patience 5 \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --log_dir "$LOG_DIR"
