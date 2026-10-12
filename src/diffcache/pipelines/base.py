"""The one abstraction every model-specific pipeline must implement.

This is deliberately small and does NOT try to be a universal diffusion-model
wrapper -- each architecture encodes its prompt/images and computes its
flow-matching loss differently (Qwen2.5-VL vision+text vs. OmniGen2's
text-only VLM vs. DeepGen's VLM+connector vs. FLUX's dual CLIP+T5, each with
its own tensor shapes and masking convention). What IS common across all 4
pipelines validated for the JSC paper -- caching to disk, the live/precompute/
data-parallel control flow, timing, VRAM accounting, hardware quirks -- lives
in `phases/*.py` and is written once, against this interface, instead of
once per pipeline.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import torch

from ..config import PrecisionConfig
from ..datasets.base import PairedSampleDataset


@dataclass
class FrozenComponents:
    """Opaque container for a pipeline's frozen (never-trained) modules --
    whatever a concrete adapter needs (text encoder, VLM, VAE, connector...).
    Only consumed by that same adapter's own `encode_sample`, never
    inspected generically by `phases/*.py`."""
    modules: Dict[str, Any]

    def __getattr__(self, item: str) -> Any:
        try:
            return self.modules[item]
        except KeyError as exc:
            raise AttributeError(item) from exc


class PipelineAdapter(ABC):
    """One concrete subclass per validated pipeline (Qwen-Image-Edit, FLUX.1
    Kontext, OmniGen2, DeepGen). See `qwen_image_edit.py` for a complete
    implementation and `flux_kontext.py`/`omnigen2.py`/`deepgen.py` for the
    not-yet-ported stubs (their reference implementations live in the
    sibling `Flux.1/`, `OmniGen2/`, `DeepGen/` projects' own
    `precompute_live_zero2_bench/*_bench_utils.py`)."""

    name: str

    @abstractmethod
    def supported_precisions(self) -> Sequence[str]:
        """Quant types this pipeline's trainable component can actually use,
        e.g. `("none", "nf4")` -- a pipeline whose DiT doesn't fit a given
        GPU's VRAM in bf16 (FLUX on a 16GB consumer card) should exclude
        `"none"` here rather than let it silently OOM."""

    @abstractmethod
    def needs_two_gpus_for_live(self) -> bool:
        """True when the frozen encoders + trainable DiT together don't fit
        one GPU (Qwen's 20.4B DiT + 7B VLM on a 32GB V100) -- determines
        whether `phases/live.py` must run pipeline-parallel across 2 devices
        or can run the single-GPU variant."""

    @abstractmethod
    def build_dataset(self, dataset_root: str, max_samples: int, split: str = "train_val",
                       shard_start: int = 0, shard_count: int | None = None) -> PairedSampleDataset:
        ...

    @abstractmethod
    def load_frozen(self, device: str, dtype: torch.dtype) -> FrozenComponents:
        """Loads and freezes every component that is never trained (VLM/text
        encoder, VAE, connector...), moved to `device` in `dtype`."""

    @abstractmethod
    def load_trainable(self, device: str, precision: PrecisionConfig, lora_rank: int, lora_alpha: int,
                        lora_dropout: float, gradient_checkpointing: bool) -> torch.nn.Module:
        """Loads the trainable component (DiT/MMDiT), quantized per
        `precision.quant_type`, with a fresh LoRA adapter attached."""

    @abstractmethod
    def load_scheduler(self) -> Any:
        """The flow-matching noise scheduler (e.g. diffusers'
        `FlowMatchEulerDiscreteScheduler`), used by `forward_loss` for its
        `num_train_timesteps`."""

    @abstractmethod
    def encode_sample(self, frozen: FrozenComponents, prompts: List[str], source_pils: List[Any],
                       target_pils: List[Any], device: str, dtype: torch.dtype) -> Dict[str, Any]:
        """Runs the frozen components once over a batch of raw samples,
        returning a dict (tensors left on `device`, plus any small
        per-sample metadata a pipeline needs -- e.g. FLUX/OmniGen2/DeepGen's
        latent height/width) -- this exact method is used both by the live
        baseline (which transfers the result straight GPU-to-GPU to the
        training device every step) and by `phases/precompute.py` (which
        `.cpu()`s the tensors itself right before `torch.save`). Keeping
        this single method used by both call sites is what guarantees the
        live-vs-cached comparison measures ONLY "when" the encode happens,
        not "how" -- and not moving to CPU here avoids adding a hop the live
        baseline never actually pays.

        Qwen's own underlying VLM/VAE calls batch all samples in one forward
        pass, so its implementation takes the full `prompts`/`source_pils`/
        `target_pils` lists directly. FLUX/OmniGen2/DeepGen's own encode
        functions (`encode_pair` in their respective `*_bench_utils.py`)
        only take ONE sample at a time -- their adapters loop internally and
        reuse their own ported `collate_batch` logic (exposed as
        `collate_cached` below, since it's the same "combine N per-sample
        dicts into one batch" operation either way) before returning."""

    @abstractmethod
    def cache_index(self, cached_items: List[Dict[str, Any]]) -> Any:
        """Any metadata `collate_cached` needs from the FULL pool of cached
        items before it can combine an arbitrary subset of them into one
        batch -- e.g. Qwen's longest prompt length, used to pad a batch
        collated from independently-saved `.pt` files. Must be computed once
        from the full pool (not just the current batch) so every rank in a
        data-parallel run agrees on the same shapes. Pipelines whose samples
        are already uniform (FLUX/OmniGen2/DeepGen: fixed prompt + fixed
        target resolution) don't need this -- return `None`."""

    @abstractmethod
    def collate_cached(self, cached_items: List[Dict[str, Any]], cache_index: Any,
                        dtype: torch.dtype) -> Dict[str, Any]:
        """Batches (and pads where needed, using `cache_index`) a list of
        per-sample cached dicts -- the disk-read side of what `encode_sample`
        produces. For pipelines that don't pad, this is the same combine
        logic `encode_sample` reuses internally for its own per-sample
        loop."""

    @abstractmethod
    def forward_loss(self, model: torch.nn.Module, batch: Dict[str, torch.Tensor], device: str,
                      compute_dtype: torch.dtype, seed: int) -> torch.Tensor:
        """Samples noise/timesteps, runs the trainable model's forward pass,
        and returns the scalar flow-matching loss. `batch` is whatever
        `collate_cached`/`encode_sample` produced, already moved to
        `device`."""
