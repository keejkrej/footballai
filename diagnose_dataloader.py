"""Diagnose StatsBomb DataLoader bottleneck."""

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import ModelConfig
from data import SequenceDataset, padded_collate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_matches", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--batches", type=int, default=20)
    args = parser.parse_args()

    cfg = ModelConfig(
        seq_len=args.seq_len,
        seq_stride=args.seq_len // 2,
        max_matches=args.max_matches,
    )

    print("Building dataset...")
    t0 = time.time()
    ds = SequenceDataset(
        data_root=cfg.data_root,
        seq_len=cfg.seq_len,
        stride=cfg.seq_stride,
        horizon_seconds=cfg.label_horizon_seconds,
        max_matches=args.max_matches,
    )
    print(f"Dataset built in {time.time() - t0:.2f}s, length={len(ds)}")

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=padded_collate,
        drop_last=True,
    )

    print(f"\nLoading {args.batches} batches (num_workers={args.num_workers}, batch_size={args.batch_size})...")
    times = []
    batch_sizes = []
    for i, batch in enumerate(loader):
        if i == 0:
            start = time.time()
        else:
            elapsed = time.time() - start
            times.append(elapsed)
            batch_sizes.append(batch["frames"].shape[0])
            if i >= args.batches:
                break
            start = time.time()
        # Simulate forward-only tensor transfer to GPU to mimic training prelude.
        _ = batch["frames"].to("cuda", non_blocking=True)

    times = times[1:] if len(times) > 1 else times
    if times:
        mean = sum(times) / len(times)
        print(f"\nMean batch load time: {mean:.3f}s")
        print(f"Batches/sec: {1.0 / mean:.2f}")
        print(f"Samples/sec: {sum(batch_sizes) / sum(times):.2f}")
        print(f"Min: {min(times):.3f}s  Max: {max(times):.3f}s")
    else:
        print("\nNot enough batches to measure timing.")


if __name__ == "__main__":
    main()
