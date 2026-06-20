"""Preprocess StatsBomb raw events into per-match PyTorch tensors.

This avoids rebuilding MatchState from JSON on every training epoch. Run once:

    python preprocess.py --data_dir /home/jack/workspace/open-data/data --out_dir ./data/processed

Output per match `{match_id}.pt` contains:
    states:  [N_events, 23, 10]
    labels:  dict of [N_events, ...] tensors
    masks:   [N_events, 22]
"""

import argparse
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from data import load_events, load_lineups
from data.state_builder import MatchState


def _preprocess_worker(path: Path, data_root: str, out_dir: str) -> int:
    match_id = int(path.stem)
    preprocess_match(match_id, Path(data_root), Path(out_dir))
    return match_id


def preprocess_match(match_id: int, data_root: Path, out_dir: Path) -> None:
    """Convert one match's events to tensors and save."""
    events = load_events(match_id, str(data_root))
    lineups = load_lineups(match_id, str(data_root))
    state = MatchState(events, lineups)

    n = len(state)
    states = torch.stack([state[i][0] for i in range(n)], dim=0)
    masks = torch.stack([state[i][2] for i in range(n)], dim=0)

    # Convert labels to tensors.
    pass_xy = torch.zeros(n, 2, dtype=torch.float32)
    pass_slot = torch.full((n,), -1, dtype=torch.long)
    shot_flag = torch.zeros(n, dtype=torch.float32)
    shot_xg = torch.zeros(n, dtype=torch.float32)
    turnover = torch.zeros(n, dtype=torch.float32)

    for i in range(n):
        lbl = state[i][1]
        pr = lbl["pass_receiver"]
        pass_xy[i] = pr[:2]
        slot = int(pr[2].item()) if pr[2].item() >= 0 else -1
        if slot >= 22:
            slot = -1
        pass_slot[i] = slot
        xg = lbl["shot_score"].item()
        shot_flag[i] = 1.0 if xg > 0 else 0.0
        shot_xg[i] = xg
        turnover[i] = lbl["turnover"].item()

    torch.save(
        {
            "states": states,
            "masks": masks,
            "pass_xy": pass_xy,
            "pass_slot": pass_slot,
            "shot_flag": shot_flag,
            "shot_xg": shot_xg,
            "turnover": turnover,
            "match_id": match_id,
        },
        out_dir / f"{match_id}.pt",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/home/jack/workspace/open-data/data")
    parser.add_argument("--out_dir", type=str, default="./data/processed")
    parser.add_argument("--max_matches", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=1)
    args = parser.parse_args()

    data_root = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    event_files = sorted((data_root / "events").glob("*.json"))
    if args.max_matches:
        event_files = event_files[:args.max_matches]

    print(f"Preprocessing {len(event_files)} matches into {out_dir} ...")
    t0 = time.time()

    if args.num_workers <= 1:
        for path in tqdm(event_files):
            match_id = int(path.stem)
            preprocess_match(match_id, data_root, out_dir)
    else:
        import multiprocessing as mp
        from functools import partial

        worker = partial(_preprocess_worker, data_root=str(data_root), out_dir=str(out_dir))
        with mp.Pool(processes=args.num_workers) as pool:
            list(tqdm(pool.imap(worker, event_files), total=len(event_files)))

    elapsed = time.time() - t0
    print(f"Done. Processed {len(event_files)} matches in {elapsed:.1f}s ({elapsed / len(event_files):.2f}s/match)")


if __name__ == "__main__":
    main()
