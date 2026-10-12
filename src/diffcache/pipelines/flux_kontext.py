"""FLUX.1-Kontext-dev adapter -- ported from
`Flux.1/precompute_live_zero2_bench/flux_bench_utils.py` on PopOS-IA (first
verified on this machine's 2x RTX 5070Ti; BULL/Volta verification is a
later phase, see the project plan).

Architecture (RGB->thermal task on LLVIP, same as the other 2 new
pipelines): CLIP-L/14 + T5-v1.1-XXL + FLUX VAE, all frozen; guidance-
distilled Rectified-Flow DiT (11.9B real size), QLoRA-trainable, NF4-only
(doesn't fit bf16 on a 16GB consumer GPU, and likely not on a 32GB V100
either -- 11.9B*2 bytes + ~10GB frozen > 32GB). Source image is VAE-encoded
and concatenated to the target latent along the token sequence.

Unlike Qwen's VLM, FLUX's `encode_pair` only takes ONE sample at a time --
this adapter loops over the batch internally and reuses `_combine` (the
same "stack per-sample dicts into one batch" logic as the original script's
`collate()`) for both `encode_sample`'s own loop and `collate_cached`'s
disk-read path, per the pattern documented in `base.py`.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

import torch
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from diffusers import FluxKontextPipeline, FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux_kontext import PREFERRED_KONTEXT_RESOLUTIONS
from peft import LoraConfig, get_peft_model

from ..config import PrecisionConfig
from ..datasets.llvip import LLVIPPairsDataset
from ..dtypes import TORCH_DTYPES
from .base import FrozenComponents, PipelineAdapter

DEFAULT_PROMPT = "Convert this image to a thermal infrared image, showing heat signatures in false color."
DEFAULT_BASE_MODEL = "black-forest-labs/FLUX.1-Kontext-dev"

LORA_TARGET_MODULES = [
    "to_q", "to_k", "to_v", "to_out.0",
    "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",
]

MAX_SEQUENCE_LENGTH = 512
GUIDANCE_SCALE = 3.5  # guidance-distilled model's fixed training target, matches the original script's default


def _get_target_resolution(pil_img) -> tuple:
    w, h = pil_img.size
    aspect_ratio = w / h
    _, width, height = min(
        (abs(aspect_ratio - pw / ph), pw, ph) for pw, ph in PREFERRED_KONTEXT_RESOLUTIONS
    )
    return height, width


def _pil_to_vae_tensor(pipeline, pil_img, height, width, device, dtype):
    resized = pipeline.image_processor.resize(pil_img, height, width)
    tensor = pipeline.image_processor.preprocess(resized, height, width)
    return tensor.to(device=device, dtype=dtype)


@torch.no_grad()
def _encode_pair(pipeline, source_pil, target_pil, prompt: str, device, dtype,
                  max_sequence_length: int = MAX_SEQUENCE_LENGTH) -> Dict[str, Any]:
    vae_dtype = next(pipeline.vae.parameters()).dtype
    height, width = _get_target_resolution(target_pil)

    t_vae = _pil_to_vae_tensor(pipeline, target_pil, height, width, device, dtype=vae_dtype)
    target_enc = pipeline._encode_vae_image(t_vae, generator=None)
    B, C, H_lat, W_lat = target_enc.shape
    target_latents_packed = pipeline._pack_latents(target_enc, B, C, H_lat, W_lat)

    s_vae = _pil_to_vae_tensor(pipeline, source_pil, height, width, device, dtype=vae_dtype)
    source_enc = pipeline._encode_vae_image(s_vae, generator=None)
    source_latents_packed = pipeline._pack_latents(source_enc, B, C, H_lat, W_lat)

    prompt_embeds, pooled_prompt_embeds, _ = pipeline.encode_prompt(
        prompt=[prompt], device=device, num_images_per_prompt=1, max_sequence_length=max_sequence_length,
    )

    return {
        "target_latents_packed": target_latents_packed.to(dtype=dtype),
        "source_latents_packed": source_latents_packed.to(dtype=dtype),
        "prompt_embeds": prompt_embeds.to(dtype=dtype),
        "pooled_prompt_embeds": pooled_prompt_embeds.to(dtype=dtype),
        "latent_height": H_lat,
        "latent_width": W_lat,
    }


def _prepare_latent_image_ids(patch_height, patch_width, device, dtype):
    latent_image_ids = torch.zeros(patch_height, patch_width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(patch_height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(patch_width)[None, :]
    h, w, c = latent_image_ids.shape
    return latent_image_ids.reshape(h * w, c).to(device=device, dtype=dtype)


def _build_ids(prompt_seq_len, latent_height, latent_width, device, dtype):
    patch_h, patch_w = latent_height // 2, latent_width // 2
    latent_ids = _prepare_latent_image_ids(patch_h, patch_w, device, dtype)
    image_ids = _prepare_latent_image_ids(patch_h, patch_w, device, dtype)
    image_ids[..., 0] = 1
    img_ids = torch.cat([latent_ids, image_ids], dim=0)
    text_ids = torch.zeros(prompt_seq_len, 3, device=device, dtype=dtype)
    return img_ids, text_ids


class FluxKontextAdapter(PipelineAdapter):
    name = "flux_kontext"

    def supported_precisions(self) -> Sequence[str]:
        return ("nf4",)

    def needs_two_gpus_for_live(self) -> bool:
        # NF4 DiT (~7GB) + frozen CLIP/T5/VAE (~10GB) add up to ~17GB, which
        # does NOT fit a 16GB card (confirmed empirically: OOM on a single
        # RTX 5070Ti) -- this is why the original finetune_live_multigpu.py
        # always split them across 2 GPUs on PopOS-IA, even though the two
        # components together DO fit a single 24GB/32GB card (A30/V100).
        # True here since this round targets PopOS-IA's 16GB cards; revisit
        # once the BULL phase (32GB V100) actually measures it.
        return True

    def build_dataset(self, dataset_root: str, max_samples: int, split: str = "train_val",
                       shard_start: int = 0, shard_count: int | None = None) -> LLVIPPairsDataset:
        return LLVIPPairsDataset(
            dataset_root, DEFAULT_PROMPT, max_samples,
            shard_start=shard_start, shard_count=shard_count,
        )

    def load_frozen(self, device: str, dtype: torch.dtype) -> FrozenComponents:
        pipeline = FluxKontextPipeline.from_pretrained(DEFAULT_BASE_MODEL, torch_dtype=dtype)
        del pipeline.transformer
        pipeline.transformer = None
        torch.cuda.empty_cache()
        pipeline.vae.to(device)
        pipeline.text_encoder.to(device)
        pipeline.text_encoder_2.to(device)
        pipeline.vae.eval()
        pipeline.text_encoder.eval()
        pipeline.text_encoder_2.eval()
        return FrozenComponents({"pipeline": pipeline})

    def load_trainable(self, device: str, precision: PrecisionConfig, lora_rank: int, lora_alpha: int,
                        lora_dropout: float, gradient_checkpointing: bool) -> torch.nn.Module:
        if precision.quant_type != "nf4":
            raise ValueError(
                f"quant_type={precision.quant_type!r} not supported by this adapter "
                f"(supported: {self.supported_precisions()})"
            )
        compute_dtype = TORCH_DTYPES[precision.resolved_compute_dtype()]

        quant_config = DiffusersBitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=compute_dtype,
        )
        # device_map pins the load directly to `device` -- the original
        # load_trainable_dit loads with no device_map (defaults to whatever
        # torch.cuda.current_device() is, i.e. GPU 0) and relies on a
        # trailing transformer.to(device) to relocate it. That two-step
        # dance transiently needs headroom on GPU 0 too (confirmed: OOM'd
        # there on a contended machine even though the target GPU 1 had
        # plenty of room) -- pinning the device upfront, like
        # OmniGen2/DeepGen's own NF4 branch already does, avoids that
        # entirely without changing where the model ends up.
        transformer = FluxTransformer2DModel.from_pretrained(
            DEFAULT_BASE_MODEL, subfolder="transformer",
            quantization_config=quant_config, torch_dtype=compute_dtype,
            device_map={"": device},
        )
        lora_config = LoraConfig(
            r=lora_rank, lora_alpha=lora_alpha, target_modules=LORA_TARGET_MODULES,
            lora_dropout=0.0,  # matches the original load_trainable_dit, which has no lora_dropout param
            bias="none",
        )
        transformer = get_peft_model(transformer, lora_config)
        if gradient_checkpointing:
            transformer.enable_gradient_checkpointing()
        transformer.print_trainable_parameters()
        return transformer

    def load_scheduler(self) -> Any:
        # FLUX's own compute_loss samples timesteps manually (logit-normal
        # sigmoid(randn)), never looks up a scheduler's num_train_timesteps --
        # nothing to load here, unlike Qwen.
        return None

    def _combine(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        heights = {it["latent_height"] for it in items}
        widths = {it["latent_width"] for it in items}
        if len(heights) > 1 or len(widths) > 1:
            raise ValueError(
                f"Batch has samples with different latent resolutions ({heights}x{widths}) -- "
                "use batch_size=1 or a dataset with uniform aspect ratio (matches the original "
                "finetune_precomputed_bench.py::collate's own check)."
            )
        combined: Dict[str, Any] = {
            key: torch.cat([it[key] for it in items], dim=0)
            for key in ("target_latents_packed", "source_latents_packed", "prompt_embeds", "pooled_prompt_embeds")
        }
        combined["latent_height"] = items[0]["latent_height"]
        combined["latent_width"] = items[0]["latent_width"]
        return combined

    def encode_sample(self, frozen: FrozenComponents, prompts: List[str], source_pils: List[Any],
                       target_pils: List[Any], device: str, dtype: torch.dtype) -> Dict[str, Any]:
        items = [
            _encode_pair(frozen.pipeline, src, tgt, prompt, device, dtype)
            for prompt, src, tgt in zip(prompts, source_pils, target_pils)
        ]
        return self._combine(items)

    def cache_index(self, cached_items: List[Dict[str, Any]]) -> Any:
        return None  # samples are already uniform (fixed prompt + LLVIP's fixed aspect ratio), no padding needed

    def collate_cached(self, cached_items: List[Dict[str, Any]], cache_index: Any,
                        dtype: torch.dtype) -> Dict[str, Any]:
        return self._combine(cached_items)

    def forward_loss(self, model: torch.nn.Module, batch: Dict[str, Any], device: str,
                      compute_dtype: torch.dtype, seed: int) -> torch.Tensor:
        target_latents = batch["target_latents_packed"].to(device=device, dtype=compute_dtype)
        source_latents = batch["source_latents_packed"].to(device=device, dtype=compute_dtype)
        prompt_embeds = batch["prompt_embeds"].to(device=device, dtype=compute_dtype)
        pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(device=device, dtype=compute_dtype)
        B = target_latents.shape[0]

        generator = torch.Generator(device=device).manual_seed(seed)
        noise = torch.randn(target_latents.shape, generator=generator, device=device, dtype=compute_dtype)
        u = torch.randn(B, generator=generator, device=device)
        t = torch.sigmoid(u)

        t_expanded = t.view(B, 1, 1).to(compute_dtype)
        noisy_latents = (1.0 - t_expanded) * target_latents + t_expanded * noise
        velocity_target = noise - target_latents

        hidden_states = torch.cat([noisy_latents, source_latents], dim=1)
        img_ids, text_ids = _build_ids(
            prompt_embeds.shape[1], batch["latent_height"], batch["latent_width"], device, compute_dtype
        )
        guidance = torch.full((B,), GUIDANCE_SCALE, device=device, dtype=torch.float32)

        noise_pred = model(
            hidden_states=hidden_states,
            timestep=t.to(compute_dtype),
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=img_ids,
            return_dict=False,
        )[0][:, : target_latents.size(1)]

        return torch.nn.functional.mse_loss(noise_pred.float(), velocity_target.float())
