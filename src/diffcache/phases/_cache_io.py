"""Shared disk-cache helpers for `finetune_cached.py` and `data_parallel.py`
-- both read the same `.pt` pool `precompute.py` wrote, through the
pipeline adapter's `cache_index`/`collate_cached`.
"""
from __future__ import annotations

import glob
import os
from typing import Any, Dict, List, Tuple

import torch

from ..prefetch import map_tensors


def load_precomputed_index(adapter, precomputed_dir: str) -> Tuple[List[str], Any]:
    paths = sorted(glob.glob(os.path.join(precomputed_dir, "sample_*.pt")))
    if not paths:
        raise RuntimeError(f"No .pt files in {precomputed_dir}. Run phase=precompute first.")
    items = [torch.load(p, map_location="cpu", weights_only=True) for p in paths]
    cache_index = adapter.cache_index(items)
    return paths, cache_index


def load_and_collate_batch(adapter, paths: List[str], cache_index: Any, dtype: torch.dtype) -> Dict[str, Any]:
    items = [torch.load(p, map_location="cpu", weights_only=True) for p in paths]
    return adapter.collate_cached(items, cache_index, dtype)


def move_batch_to_device(batch: Dict[str, Any], device: str, compute_dtype: torch.dtype) -> Dict[str, Any]:
    """Moves every tensor found anywhere in `batch` (including nested inside
    lists/tuples -- some pipelines, e.g. OmniGen2/DeepGen, collate reference
    latents as lists of tensors instead of a single stacked tensor, see
    their `collate_cached`) to `device`, casting floating-point tensors to
    `compute_dtype` and leaving integer tensors (masks, indices) and any
    non-tensor metadata (pair ids, resolutions) as-is."""

    def _move(t: torch.Tensor) -> torch.Tensor:
        if t.is_floating_point():
            return t.to(device, dtype=compute_dtype)
        return t.to(device)

    return map_tensors(batch, _move)


def resolve_gradient_checkpointing(run_config) -> bool:
    if run_config.gradient_checkpointing == "auto":
        return run_config.precision.quant_type == "int8"
    return run_config.gradient_checkpointing == "on"
