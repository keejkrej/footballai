"""Fast dataset that reads preprocessed .pt tensors instead of raw JSON.

Use after running `python preprocess.py`.
"""

from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from config import ModelConfig


class PreprocessedMatchDataset(Dataset):
    """One sample per event, loaded from a preprocessed .pt file.

    This is a thin wrapper used by PreprocessedSequenceDataset.
    """

    def __init__(self, processed_dir: str, max_matches: Optional[int] = None):
        super().__init__()
        self.processed_dir = Path(processed_dir)
        self.files = sorted(self.processed_dir.glob("*.pt"))
        if max_matches is not None:
            self.files = self.files[:max_matches]

        # Build cumulative lengths for global indexing.
        self._match_ids: List[int] = []
        self._counts: List[Tuple[int, int]] = []
        self._lengths: List[int] = []
        for path in self.files:
            match_id = int(path.stem)
            # Load only the length metadata quickly.
            data = torch.load(path, weights_only=False)
            n = data["states"].shape[0]
            self._match_ids.append(match_id)
            self._counts.append((match_id, n))
            self._lengths.append(n)
        self._cum_lengths = np.cumsum(self._lengths).tolist()

    def __len__(self) -> int:
        return int(self._cum_lengths[-1]) if self._cum_lengths else 0

    def _find_match(self, idx: int) -> Tuple[int, int]:
        import bisect

        pos = bisect.bisect_left(self._cum_lengths, idx + 1)
        prev = self._cum_lengths[pos - 1] if pos > 0 else 0
        local_idx = idx - prev
        return pos, local_idx

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"index {idx} out of range")
        pos, local_idx = self._find_match(idx)
        path = self.files[pos]
        data = torch.load(path, weights_only=False)

        state = data["states"][local_idx]
        mask = data["masks"][local_idx]
        pass_xy = data["pass_xy"][local_idx]
        pass_slot = data["pass_slot"][local_idx]
        labels = {
            "pass_receiver": torch.cat([pass_xy, pass_slot.unsqueeze(0)]),
            "shot_score": data["shot_xg"][local_idx],
            "turnover": data["turnover"][local_idx],
        }

        return state, labels, mask

    @property
    def match_counts(self) -> List[Tuple[int, int]]:
        return list(self._counts)

    @property
    def match_ids(self) -> List[int]:
        return list(self._match_ids)


class PreprocessedSequenceDataset(Dataset):
    """Fixed-length sequences sliced from preprocessed match tensors.

    Much faster than building state from JSON on the fly. Keeps an LRU cache of
    loaded match tensors so consecutive sequences from the same match avoid
    repeated `torch.load` calls.
    """

    def __init__(
        self,
        processed_dir: str,
        seq_len: int = 50,
        stride: int = 25,
        max_matches: Optional[int] = None,
        cache_matches: int = 16,
    ):
        super().__init__()
        self.processed_dir = Path(processed_dir)
        self.files = sorted(self.processed_dir.glob("*.pt"))
        if max_matches is not None:
            self.files = self.files[:max_matches]

        self.seq_len = seq_len
        self.stride = stride
        self.cache_matches = cache_matches
        self._cache: OrderedDict[int, dict] = OrderedDict()

        # Build index of sequences.
        self._index: List[Tuple[int, int, int, int]] = []
        self._match_ids: List[int] = []
        self._match_lengths: List[int] = []
        for path in self.files:
            match_id = int(path.stem)
            data = torch.load(path, weights_only=False)
            n = data["states"].shape[0]
            self._match_ids.append(match_id)
            self._match_lengths.append(n)
            for start in range(0, n, stride):
                actual_len = min(seq_len, n - start)
                if actual_len < 2:
                    continue
                self._index.append((match_id, start, actual_len, n))

    def __len__(self) -> int:
        return len(self._index)

    def _load_match(self, match_id: int) -> dict:
        """Load a match tensor, with LRU caching."""
        if match_id in self._cache:
            self._cache.move_to_end(match_id)
            return self._cache[match_id]

        path = self.processed_dir / f"{match_id}.pt"
        data = torch.load(path, weights_only=False)
        self._cache[match_id] = data
        self._cache.move_to_end(match_id)
        while len(self._cache) > self.cache_matches:
            self._cache.popitem(last=False)
        return data

    def __getitem__(self, idx: int) -> dict:
        match_id, start, actual_len, _ = self._index[idx]
        data = self._load_match(match_id)
        end = start + actual_len

        states = data["states"][start:end]
        masks = data["masks"][start:end]
        pass_xy_all = data["pass_xy"][start:end]
        pass_slot_all = data["pass_slot"][start:end]
        shot_xg_all = data["shot_xg"][start:end]
        turnover_all = data["turnover"][start:end]

        pad_len = self.seq_len - actual_len
        if pad_len > 0:
            states = torch.cat([states, torch.zeros(pad_len, *states.shape[1:], dtype=states.dtype)])
            masks = torch.cat([masks, torch.zeros(pad_len, *masks.shape[1:], dtype=masks.dtype)])
            pass_xy_all = torch.cat([pass_xy_all, torch.zeros(pad_len, 2, dtype=pass_xy_all.dtype)])
            pass_slot_all = torch.cat([pass_slot_all, torch.full((pad_len,), -1, dtype=pass_slot_all.dtype)])
            shot_xg_all = torch.cat([shot_xg_all, torch.zeros(pad_len, dtype=shot_xg_all.dtype)])
            turnover_all = torch.cat([turnover_all, torch.zeros(pad_len, dtype=turnover_all.dtype)])

        pass_xy = pass_xy_all
        pass_slot = pass_slot_all
        shot_flag = (shot_xg_all > 0).float()
        shot_xg = torch.stack([shot_flag, shot_xg_all], dim=-1)
        turnover = turnover_all

        return {
            "frames": states,
            "lengths": torch.tensor(actual_len, dtype=torch.long),
            "mask": masks,
            "pass_receiver_xy": pass_xy,
            "pass_receiver_slot": pass_slot,
            "shot_xg": shot_xg,
            "turnover": turnover,
        }
