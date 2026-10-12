"""STUB -- FLUX.1-Kontext-dev is not yet ported to this framework.

Reference implementation to port from:
`/home/antonio/Proyectos/Flux.1/precompute_live_zero2_bench/flux_bench_utils.py`
(+ `dataset_llvip.py`, `precompute_bench.py`, `finetune_precomputed_bench.py`,
`finetune_live_multigpu.py`, `finetune_zero2_bench.py` in that same folder).

Key facts for whoever ports this (see project_flux_kontext_thermal_bench /
project_jsc_paper memories): CLIP-L/14 (123M) + T5-v1.1-XXL (4.76B) + FLUX
VAE (84M) frozen, Rectified-Flow DiT 11.901B real size (QLoRA-trainable),
guidance-distilled (no CFG double-forward). NF4-only on a 16GB consumer GPU
(bf16 doesn't fit) -- `supported_precisions()` below must stay `("nf4",)`
once implemented unless tested on a >=24GB card. Source image is VAE-encoded
and concatenated to the target latent along the token sequence, same
conditioning mechanism as Qwen's -- `encode_sample`/`forward_loss` should
follow the same shape as `qwen_image_edit.py`'s.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

import torch

from ..config import PrecisionConfig
from ..datasets.base import PairedSampleDataset
from .base import FrozenComponents, PipelineAdapter

_NOT_PORTED = (
    "FluxKontextAdapter is not implemented yet. Port it from "
    "Flux.1/precompute_live_zero2_bench/flux_bench_utils.py -- see this "
    "module's docstring for the facts needed to do it."
)


class FluxKontextAdapter(PipelineAdapter):
    name = "flux_kontext"

    def supported_precisions(self) -> Sequence[str]:
        return ("nf4",)

    def needs_two_gpus_for_live(self) -> bool:
        return False  # fits one GPU in NF4 (~7GB DiT + ~10GB frozen encoders), see memory

    def build_dataset(self, dataset_root: str, max_samples: int, split: str = "train_val",
                       shard_start: int = 0, shard_count: int | None = None) -> PairedSampleDataset:
        raise NotImplementedError(_NOT_PORTED)

    def load_frozen(self, device: str, dtype: torch.dtype) -> FrozenComponents:
        raise NotImplementedError(_NOT_PORTED)

    def load_trainable(self, device: str, precision: PrecisionConfig, lora_rank: int, lora_alpha: int,
                        lora_dropout: float, gradient_checkpointing: bool) -> torch.nn.Module:
        raise NotImplementedError(_NOT_PORTED)

    def load_scheduler(self) -> Any:
        raise NotImplementedError(_NOT_PORTED)

    def encode_sample(self, frozen: FrozenComponents, prompts: List[str], source_pils: List[Any],
                       target_pils: List[Any], device: str, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
        raise NotImplementedError(_NOT_PORTED)

    def cache_max_seq_len(self, cached_items: List[Dict[str, torch.Tensor]]) -> int:
        raise NotImplementedError(_NOT_PORTED)

    def collate_cached(self, cached_items: List[Dict[str, torch.Tensor]], max_seq_len: int,
                        dtype: torch.dtype) -> Dict[str, torch.Tensor]:
        raise NotImplementedError(_NOT_PORTED)

    def forward_loss(self, model: torch.nn.Module, batch: Dict[str, torch.Tensor], device: str,
                      compute_dtype: torch.dtype, seed: int) -> torch.Tensor:
        raise NotImplementedError(_NOT_PORTED)
