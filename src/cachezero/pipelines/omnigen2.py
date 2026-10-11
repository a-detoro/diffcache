"""STUB -- OmniGen2 is not yet ported to this framework.

Reference implementation to port from:
`/home/antonio/Proyectos/OmniGen2/precompute_live_zero2_bench/omnigen2_bench_utils.py`
(+ `dataset_llvip.py`, `precompute_bench.py`, `finetune_precomputed_bench.py`,
`finetune_live_multigpu.py`, `finetune_zero2_bench.py` in that same folder).

Key facts for whoever ports this (see project_omnigen2_jupyter /
project_omnigen2_popos_bench memories): text encoder is Qwen2.5-VL-**3B**
(text-only mode -- never sees the reference image during training, unlike
Qwen-Image-Edit's own VLM), VAE is FLUX's own (bundled in the
`OmniGen2/OmniGen2` HF repo, do NOT pull `black-forest-labs/FLUX.1-dev`
directly, it's gated), DiT is `OmniGen2Transformer2DModel` (~3.97B,
Lumina2-style joint attention, 3-axis RoPE). Both bf16 and NF4 fit a 16GB
card, so `supported_precisions()` below should become `("none", "nf4")`
once implemented. Loss uses the repo's own `omnigen2.transport` module, not
a hand-rolled flow-matching formula -- a real difference from the other 3
pipelines' `forward_loss`, worth keeping explicit when porting.
Known footgun already found once (see memory): `OmniGen2RotaryPosEmbed.
get_freqs_cis()` returns a `List[torch.Tensor]`, not a single tensor --
don't call `.to(device)` on it directly, the model's own `_get_freqs_cis`
already moves each axis internally.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

import torch

from ..config import PrecisionConfig
from ..datasets.base import PairedSampleDataset
from .base import FrozenComponents, PipelineAdapter

_NOT_PORTED = (
    "OmniGen2Adapter is not implemented yet. Port it from "
    "OmniGen2/precompute_live_zero2_bench/omnigen2_bench_utils.py -- see "
    "this module's docstring for the facts needed to do it."
)


class OmniGen2Adapter(PipelineAdapter):
    name = "omnigen2"

    def supported_precisions(self) -> Sequence[str]:
        return ("none", "nf4")

    def needs_two_gpus_for_live(self) -> bool:
        return False  # ~3.97B DiT + 3B VLM both fit one 16GB/24GB card, see memory

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
