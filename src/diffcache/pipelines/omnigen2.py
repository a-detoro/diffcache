"""OmniGen2 adapter -- ported from
`OmniGen2/precompute_live_zero2_bench/omnigen2_bench_utils.py` on PopOS-IA.

Architecture (RGB->thermal task on LLVIP, same prompt/dataset as the other
2 new pipelines): text encoder is Qwen2.5-VL-3B-Instruct in TEXT-ONLY mode
(never sees the reference image, unlike Qwen-Image-Edit's own VLM); VAE is
FLUX.1-dev's, loaded from OmniGen2's own HF repo (NOT
`black-forest-labs/FLUX.1-dev` directly, that one is gated); DiT is
`OmniGen2Transformer2DModel` (~3.97B, Lumina2-style joint attention,
3-axis RoPE) -- small enough that NF4 isn't required (`supported_precisions`
includes `"none"`), trained in plain bf16+LoRA like the repo's own
`options/ft_lora.yml`. Loss uses the repo's own `omnigen2.transport` module
(flow-matching), not a hand-rolled formula like FLUX's/DeepGen's.

Requires the `OMNIGEN2_REPO_PATH` env var (a checkout of the `OmniGen2`
repo, e.g. `/home/antonio/Proyectos/OmniGen2`) -- the `omnigen2` Python
package isn't on PyPI, same convention as Qwen's `QWEN_BASE_MODEL_PATH`.

Like FLUX, `encode_pair` only takes ONE sample at a time -- this adapter
loops internally and reuses `_combine` (ported from the original
`collate_batch`, minus the device move, which `phases/*.py` does itself
via `move_batch_to_device`) for both `encode_sample` and `collate_cached`.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Sequence

import torch

from ..config import PrecisionConfig
from ..datasets.llvip import LLVIPPairsDataset
from ..dtypes import TORCH_DTYPES
from .base import FrozenComponents, PipelineAdapter

DEFAULT_PROMPT = "Convert this image to a thermal infrared image, showing heat signatures in false color."
DEFAULT_BASE_MODEL = "OmniGen2/OmniGen2"
DEFAULT_TEXT_ENCODER = "Qwen/Qwen2.5-VL-3B-Instruct"

LORA_TARGET_MODULES = ["to_k", "to_q", "to_v", "to_out.0"]
SYSTEM_PROMPT = "You are a helpful assistant that generates high-quality images based on user instructions."

MAX_OUTPUT_PIXELS = 1024 * 1024
MAX_SIDE_LENGTH = 2048


def _require_omnigen2_repo() -> None:
    repo_path = os.environ.get("OMNIGEN2_REPO_PATH")
    if not repo_path:
        raise RuntimeError(
            "OMNIGEN2_REPO_PATH is not set -- the OmniGen2 adapter needs it to point at a "
            "checkout of the OmniGen2 repo (the `omnigen2` package isn't on PyPI)."
        )
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)


def _apply_chat_template(tokenizer, instruction: str) -> str:
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]
    return tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=False)


@torch.no_grad()
def _encode_vae(vae, img: torch.Tensor) -> torch.Tensor:
    z0 = vae.encode(img.to(dtype=vae.dtype)).latent_dist.sample()
    if vae.config.shift_factor is not None:
        z0 = z0 - vae.config.shift_factor
    if vae.config.scaling_factor is not None:
        z0 = z0 * vae.config.scaling_factor
    return z0.to(dtype=vae.dtype)


@torch.no_grad()
def _encode_pair(encoders: Dict[str, Any], source_pil, target_pil, prompt: str, device, dtype,
                  max_pixels: int = MAX_OUTPUT_PIXELS, max_side_length: int = MAX_SIDE_LENGTH) -> Dict[str, Any]:
    vae = encoders["vae"]
    tokenizer = encoders["tokenizer"]
    text_encoder = encoders["text_encoder"]
    image_processor = encoders["image_processor"]

    instruction = _apply_chat_template(tokenizer, prompt)
    text_inputs = tokenizer([instruction], padding="longest", max_length=888, truncation=True, return_tensors="pt")
    text_ids = text_inputs.input_ids.to(device)
    text_mask = text_inputs.attention_mask.to(device)
    text_hidden_states = text_encoder(
        input_ids=text_ids, attention_mask=text_mask, output_hidden_states=False,
    ).last_hidden_state

    source_tensor = image_processor.preprocess(source_pil, max_pixels=max_pixels, max_side_length=max_side_length)
    target_tensor = image_processor.preprocess(target_pil, max_pixels=max_pixels, max_side_length=max_side_length)

    ref_latent = _encode_vae(vae, source_tensor.to(device)).squeeze(0)
    output_latent = _encode_vae(vae, target_tensor.to(device)).squeeze(0)

    return {
        "text_hidden_states": text_hidden_states.to(dtype=dtype),
        "text_attention_mask": text_mask,
        "ref_image_hidden_states": ref_latent.to(dtype=dtype),
        "output_latents": output_latent.to(dtype=dtype),
    }


class OmniGen2Adapter(PipelineAdapter):
    name = "omnigen2"

    def __init__(self):
        self._transport = None
        self._freqs_cis = None

    def supported_precisions(self) -> Sequence[str]:
        return ("none", "nf4")

    def needs_two_gpus_for_live(self) -> bool:
        # ~3.97B DiT (bf16, ~8GB) + Qwen2.5-VL-3B text encoder (~6-7GB) + VAE
        # add up to more than one 16GB card can hold together (the original
        # finetune_live_multigpu.py always splits them across 2 GPUs on
        # PopOS-IA, same as FLUX's -- no single-GPU live variant exists for
        # this pipeline, unlike DeepGen's). Confirmed empirically during
        # this port's own verification (see project_diffcache_framework
        # memory). Revisit for BULL's 32GB V100s later.
        return True

    def build_dataset(self, dataset_root: str, max_samples: int, split: str = "train_val",
                       shard_start: int = 0, shard_count: int | None = None) -> LLVIPPairsDataset:
        return LLVIPPairsDataset(
            dataset_root, DEFAULT_PROMPT, max_samples,
            shard_start=shard_start, shard_count=shard_count,
        )

    def load_frozen(self, device: str, dtype: torch.dtype) -> FrozenComponents:
        _require_omnigen2_repo()
        from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
        from transformers import AutoTokenizer
        from transformers import Qwen2_5_VLModel as TextEncoder

        from omnigen2.pipelines.image_processor import OmniGen2ImageProcessor

        vae = AutoencoderKL.from_pretrained(DEFAULT_BASE_MODEL, subfolder="vae", torch_dtype=dtype)
        vae.to(device).eval()
        vae.requires_grad_(False)

        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_TEXT_ENCODER)
        tokenizer.padding_side = "right"

        text_encoder = TextEncoder.from_pretrained(DEFAULT_TEXT_ENCODER, torch_dtype=dtype)
        text_encoder.to(device).eval()
        text_encoder.requires_grad_(False)

        img_scale_num = (2 ** (len(vae.config.block_out_channels) - 1)) * 2
        image_processor = OmniGen2ImageProcessor(vae_scale_factor=img_scale_num, do_resize=True)

        return FrozenComponents({
            "vae": vae, "tokenizer": tokenizer, "text_encoder": text_encoder, "image_processor": image_processor,
        })

    def load_trainable(self, device: str, precision: PrecisionConfig, lora_rank: int, lora_alpha: int,
                        lora_dropout: float, gradient_checkpointing: bool) -> torch.nn.Module:
        _require_omnigen2_repo()
        from peft import LoraConfig

        from omnigen2.models.transformers.repo import OmniGen2RotaryPosEmbed
        from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel

        compute_dtype = TORCH_DTYPES[precision.resolved_compute_dtype()]

        if precision.quant_type == "nf4":
            from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig

            quant_config = DiffusersBitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=compute_dtype,
            )
            model = OmniGen2Transformer2DModel.from_pretrained(
                DEFAULT_BASE_MODEL, subfolder="transformer",
                quantization_config=quant_config, torch_dtype=compute_dtype, device_map={"": device},
            )
        elif precision.quant_type == "none":
            # device_map pinned directly here too (not a plain .to(device)
            # afterward) -- see the FLUX adapter's load_trainable comment
            # for why that matters on a contended machine.
            model = OmniGen2Transformer2DModel.from_pretrained(
                DEFAULT_BASE_MODEL, subfolder="transformer", torch_dtype=compute_dtype, device_map={"": device},
            )
        else:
            raise ValueError(
                f"quant_type={precision.quant_type!r} not supported by this adapter "
                f"(supported: {self.supported_precisions()})"
            )

        model.requires_grad_(False)
        lora_config = LoraConfig(
            r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            init_lora_weights="gaussian", target_modules=LORA_TARGET_MODULES,
        )
        model.add_adapter(lora_config)
        if gradient_checkpointing:
            model.enable_gradient_checkpointing()

        # Tied to the model's own static config (axes_dim_rope/axes_lens),
        # not to a particular sample -- computed once here and reused by
        # every forward_loss call against this adapter instance, same
        # pattern as Qwen's _scheduler_cache(). NOT moved to device here:
        # it's a List[Tensor], and the model's own rope embedder moves each
        # axis to the right device internally (see module docstring).
        self._freqs_cis = OmniGen2RotaryPosEmbed.get_freqs_cis(
            model.config.axes_dim_rope, model.config.axes_lens, theta=10000,
        )
        return model

    def load_scheduler(self) -> Any:
        _require_omnigen2_repo()
        from omnigen2.transport import create_transport

        seq_len = MAX_OUTPUT_PIXELS // 16 // 16
        return create_transport(
            "Linear", "velocity", None, None, None,
            snr_type="lognorm", do_shift=True, seq_len=seq_len,
            dynamic_time_shift=True, time_shift_version="v1",
        )

    @staticmethod
    def _unwrap_to_tensor(value: Any) -> Any:
        """`output_latents`/`ref_image_hidden_states` are bare tensors coming
        straight out of `_encode_pair` (one raw sample, `encode_sample`'s own
        loop) -- but each gets wrapped in an extra length-1 list level by
        every `_combine` call it passes through afterward (`output_latents`
        once, `ref_image_hidden_states` twice: once for "this sample's ref
        images", once for "this batch's samples"). A sample saved to a
        `.pt` file by `precompute.py` has already been through one
        `_combine` call (treated as a "batch of 1"), so it arrives back
        here still wrapped. Peeling off every length-1 list level
        regardless of how many there are lets one `_combine` implementation
        handle both the raw-encode and disk-reload call sites without
        `precompute.py` needing to know this pipeline's internal
        representation."""
        while isinstance(value, list):
            assert len(value) == 1, f"expected a length-1 wrapper, got length {len(value)}"
            value = value[0]
        return value

    def _combine(self, items: List[Dict[str, Any]], dtype: torch.dtype) -> Dict[str, Any]:
        latents = [self._unwrap_to_tensor(it["output_latents"]) for it in items]
        refs = [self._unwrap_to_tensor(it["ref_image_hidden_states"]) for it in items]

        text_seq_lens = {it["text_hidden_states"].shape[1] for it in items}
        latent_shapes = {tuple(t.shape) for t in latents}
        if len(text_seq_lens) > 1 or len(latent_shapes) > 1:
            raise ValueError(
                f"collate requires uniform text/latent shapes across the batch "
                f"(seq_lens={text_seq_lens}, latent_shapes={latent_shapes}); shouldn't happen "
                "with a fixed prompt and LLVIP's fixed aspect ratio."
            )
        return {
            "text_hidden_states": torch.cat([it["text_hidden_states"] for it in items], dim=0).to(dtype=dtype),
            "text_attention_mask": torch.cat([it["text_attention_mask"] for it in items], dim=0),
            "ref_image_hidden_states": [[r.to(dtype=dtype)] for r in refs],
            "output_latents": [t.to(dtype=dtype) for t in latents],
        }

    def encode_sample(self, frozen: FrozenComponents, prompts: List[str], source_pils: List[Any],
                       target_pils: List[Any], device: str, dtype: torch.dtype) -> Dict[str, Any]:
        items = [
            _encode_pair(frozen.modules, src, tgt, prompt, device, dtype)
            for prompt, src, tgt in zip(prompts, source_pils, target_pils)
        ]
        return self._combine(items, dtype)

    def cache_index(self, cached_items: List[Dict[str, Any]]) -> Any:
        return None  # samples are already uniform, no padding needed

    def collate_cached(self, cached_items: List[Dict[str, Any]], cache_index: Any,
                        dtype: torch.dtype) -> Dict[str, Any]:
        return self._combine(cached_items, dtype)

    def forward_loss(self, model: torch.nn.Module, batch: Dict[str, Any], device: str,
                      compute_dtype: torch.dtype, seed: int) -> torch.Tensor:
        # NOTE: unlike Qwen/FLUX/DeepGen, OmniGen2's own transport.training_losses
        # doesn't expose a generator/seed hook -- `seed` is accepted (per the
        # shared ABC signature) but can't be wired into this pipeline's
        # randomness, matching the original compute_loss exactly.
        transport = self._scheduler_cache()
        ref_image_hidden_states = [
            [t.to(device=device, dtype=compute_dtype) for t in sample] if sample is not None else None
            for sample in batch["ref_image_hidden_states"]
        ]
        output_latents = [t.to(device=device, dtype=compute_dtype) for t in batch["output_latents"]]
        model_kwargs = dict(
            text_hidden_states=batch["text_hidden_states"].to(device=device, dtype=compute_dtype),
            text_attention_mask=batch["text_attention_mask"].to(device=device),
            ref_image_hidden_states=ref_image_hidden_states,
            freqs_cis=self._freqs_cis,
        )
        num_tokens = sum(latent.numel() for latent in output_latents)
        loss_dict = transport.training_losses(model, output_latents, model_kwargs, reduction="sum")
        return loss_dict["loss"].sum() / num_tokens

    def _scheduler_cache(self):
        if self._transport is None:
            self._transport = self.load_scheduler()
        return self._transport
