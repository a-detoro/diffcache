"""Common shape every paired-sample dataset in this framework exposes:
a (source image, target image, text prompt) triple, PIL-level (no tensor
conversion here -- that's the adapter's `encode_sample()` job, see
`pipelines/base.py`).

`shard_start`/`shard_count` generalize the convention already used by
`precompute_bench.py --shard_start/--shard_count` and
`LiveMultiViewSampleDataset` (dataset_live.py): two processes with the same
deterministic sample ordering slice disjoint, non-overlapping ranges of it
without needing to coordinate -- used both for precompute's own simple data
parallelism (Experiment 3 in project_qwen_bull_benchmarks) and, with
`shard_count=None`, for the normal single-process case.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

from torch.utils.data import Dataset


class PairedSampleDataset(Dataset, ABC):
    def __init__(self, max_samples: int, shard_start: int = 0, shard_count: int | None = None):
        full = self._build_sample_pairs()[:max_samples]
        end = shard_start + shard_count if shard_count is not None else len(full)
        self.samples: List[Dict] = full[shard_start:end]
        self.shard_start = shard_start

    @abstractmethod
    def _build_sample_pairs(self) -> List[Dict]:
        """Returns the full, deterministically-ordered list of sample dicts
        (at least `src_path`/`tgt_path`/`prompt` keys) before any
        max_samples/shard slicing is applied."""

    def __len__(self) -> int:
        return len(self.samples)

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict:
        """Returns PIL-level data: `source_pil`, `target_pil`, `prompt`."""


def collate_pils(examples: List[Dict]) -> Dict:
    return {
        "source_pils": [e["source_pil"] for e in examples],
        "target_pils": [e["target_pil"] for e in examples],
        "prompts": [e["prompt"] for e in examples],
    }
