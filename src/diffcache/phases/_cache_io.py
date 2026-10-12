"""Shared disk-cache helpers for `finetune_cached.py` and `data_parallel.py`
-- both read the same `.pt` pool `precompute.py` wrote, through the
pipeline adapter's `cache_max_seq_len`/`collate_cached`.
"""
from __future__ import annotations

import glob
import os
from typing import Dict, List, Tuple

import torch


def load_precomputed_index(adapter, precomputed_dir: str) -> Tuple[List[str], int]:
    paths = sorted(glob.glob(os.path.join(precomputed_dir, "sample_*.pt")))
    if not paths:
        raise RuntimeError(f"No .pt files in {precomputed_dir}. Run phase=precompute first.")
    items = [torch.load(p, map_location="cpu", weights_only=True) for p in paths]
    max_seq_len = adapter.cache_max_seq_len(items)
    return paths, max_seq_len


def load_and_collate_batch(adapter, paths: List[str], max_seq_len: int, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    items = [torch.load(p, map_location="cpu", weights_only=True) for p in paths]
    return adapter.collate_cached(items, max_seq_len, dtype)


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: str, compute_dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    """Moves every tensor to `device`, casting floating-point tensors to
    `compute_dtype` and leaving integer tensors (masks, indices) as-is."""
    moved = {}
    for key, tensor in batch.items():
        if tensor.is_floating_point():
            moved[key] = tensor.to(device, dtype=compute_dtype)
        else:
            moved[key] = tensor.to(device)
    return moved


def resolve_gradient_checkpointing(run_config) -> bool:
    if run_config.gradient_checkpointing == "auto":
        return run_config.precision.quant_type == "int8"
    return run_config.gradient_checkpointing == "on"
