"""STUB -- DeepGen 1.0 is not yet ported to this framework.

Reference implementation to port from:
`/home/antonio/Proyectos/DeepGen/precompute_live_zero2_bench/deepgen_bench_utils.py`
(+ `dataset_llvip.py`, `precompute_bench.py`, `finetune_precomputed_bench.py`,
`finetune_live_singlegpu.py`, `finetune_live_multigpu.py`,
`finetune_zero2_bench.py` in that same folder -- DeepGen is the one pipeline
with a working single-GPU live baseline, since its ~5B total is small
enough; see `phases/live.py`'s `num_gpus` option once this is ported).

Key facts for whoever ports this (see project_deepgen_popos_bench memory):
3B VLM (Qwen2.5-VL-3B, frozen) + SCB connector (0.4324B, frozen, bridges
multi-layer VLM hidden states to DiT conditioning) + SD3 DiT ("UniPic2-
SD3.5M-Kontext-2B", 2.2432B, QLoRA-trainable) + small frozen VAE. Loss is
the same manual logit-normal rectified-flow scheme as FLUX's (uses
`FlowMatchEulerDiscreteScheduler`), NOT OmniGen2's `transport` module.
Resolution is fixed 512x512 (DeepGen's own native default), unlike the other
3 pipelines which VAE-encode LLVIP at ~1280x1024. Known API-compatibility
footguns already found once against a newer `transformers` than DeepGen's
own pinned version (see memory) -- re-verify against whatever version this
package's own `transformers` dependency resolves to before trusting the
port blindly.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

import torch

from ..config import PrecisionConfig
from ..datasets.base import PairedSampleDataset
from .base import FrozenComponents, PipelineAdapter

_NOT_PORTED = (
    "DeepGenAdapter is not implemented yet. Port it from "
    "DeepGen/precompute_live_zero2_bench/deepgen_bench_utils.py -- see this "
    "module's docstring for the facts needed to do it."
)


class DeepGenAdapter(PipelineAdapter):
    name = "deepgen"

    def supported_precisions(self) -> Sequence[str]:
        return ("none", "nf4")

    def needs_two_gpus_for_live(self) -> bool:
        return False  # ~5B total fits one 16GB card even unquantized, see memory

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
