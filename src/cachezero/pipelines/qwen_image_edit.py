"""Qwen-Image-Edit-Angles adapter -- the only pipeline fully ported for this
first framework version (it already lives in this repo; FLUX.1 Kontext,
OmniGen2 and DeepGen live in sibling projects and are still stubs, see
`flux_kontext.py`/`omnigen2.py`/`deepgen.py`).

Ports, unchanged in substance, the functions already verified across every
`training_from_jupyter/*.py` benchmark script: `qwen_bench_utils.py`'s
`build_vlm_and_vae`/`build_trainable_dit`/`encode_prompt_live`/
`encode_vae_image_live`/`pack_latents`, and `model.py`'s
`QwenSingleGPUWrapper`. Those original files are left untouched -- this is a
copy adapted to the `PipelineAdapter` interface, not an import, so the
package has no path dependency on `training_from_jupyter/`.

Model paths come from `QWEN_BASE_MODEL_PATH`/`QWEN_FUSED_TRANSFORMER_PATH`
(same env vars `run_multigpu_inference.py` and every bench script already
use) -- required, not defaulted to a relative path, since this package is
meant to be installable independently of where `Qwen_JoS/base_model` happens
to sit on a given machine.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.checkpoint import checkpoint

from ..config import PrecisionConfig
from ..datasets.multiview import MultiViewRotationDataset
from ..dtypes import TORCH_DTYPES
from .base import FrozenComponents, PipelineAdapter

logger = logging.getLogger(__name__)

LORA_TARGET_MODULES = [
    "to_q", "to_k", "to_v", "to_out.0",
    "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",
]

PROMPT_TEMPLATE = (
    "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, "
    "texture, objects, background), then explain how the user's text instruction should alter "
    "or modify the image. Generate a new image that meets the user's requirements while "
    "maintaining consistency with the original input where appropriate.<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)
PROMPT_TEMPLATE_START_IDX = 64


def _require_path(env_var: str) -> str:
    path = os.environ.get(env_var)
    if not path:
        raise RuntimeError(
            f"{env_var} is not set -- the Qwen adapter needs it to locate the model weights "
            "(same convention as training_from_jupyter/qwen_bench_utils.py)."
        )
    return path


def _pil_to_vae_tensor(pil_img, resolution: int, device, dtype) -> torch.Tensor:
    transform = T.Compose([
        T.Resize((resolution, resolution), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),
    ])
    tensor = transform(pil_img).unsqueeze(0).unsqueeze(2)
    return tensor.to(device=device, dtype=dtype)


def _pil_to_condition_pil(pil_img, condition_resolution: int):
    return pil_img.resize((condition_resolution, condition_resolution), Image.BILINEAR)


def _extract_masked_hidden(hidden_states: torch.Tensor, mask: torch.Tensor):
    bool_mask = mask.bool()
    valid_lengths = bool_mask.sum(dim=1)
    selected = hidden_states[bool_mask]
    return torch.split(selected, valid_lengths.tolist(), dim=0)


@torch.no_grad()
def _encode_prompt_live(text_encoder, processor, prompts, condition_images, device, dtype):
    img_prompt_template = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
    base_img_prompt = img_prompt_template.format(1)
    txt = [PROMPT_TEMPLATE.format(base_img_prompt + p) for p in prompts]
    model_inputs = processor(text=txt, images=condition_images, padding=True, return_tensors="pt").to(device)
    outputs = text_encoder(
        input_ids=model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        pixel_values=model_inputs.pixel_values,
        image_grid_thw=model_inputs.image_grid_thw,
        output_hidden_states=True,
    )
    hidden_states = outputs.hidden_states[-1]
    split_hidden_states = _extract_masked_hidden(hidden_states, model_inputs.attention_mask)
    split_hidden_states = [e[PROMPT_TEMPLATE_START_IDX:] for e in split_hidden_states]
    attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
    max_seq_len = max(e.size(0) for e in split_hidden_states)
    prompt_embeds = torch.stack(
        [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
    )
    encoder_attention_mask = torch.stack(
        [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
    )
    return prompt_embeds.to(dtype=dtype, device=device), encoder_attention_mask.to(device=device)


@torch.no_grad()
def _encode_vae_image_live(vae, image_tensor: torch.Tensor) -> torch.Tensor:
    latents = vae.encode(image_tensor).latent_dist.mode()
    z_dim = vae.config.z_dim
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, z_dim, 1, 1, 1).to(latents.device, latents.dtype)
    latents_std = torch.tensor(vae.config.latents_std).view(1, z_dim, 1, 1, 1).to(latents.device, latents.dtype)
    return (latents - latents_mean) / latents_std


def _pack_latents(latents: torch.Tensor, batch_size: int, num_channels_latents: int,
                   height: int, width: int) -> torch.Tensor:
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)


class QwenSingleGPUWrapper(nn.Module):
    """Port of `training_from_jupyter/model.py::QwenSingleGPUWrapper`,
    unchanged: reimplements the DiT's forward block-by-block so every
    transformer block can be wrapped in `torch.utils.checkpoint.checkpoint`
    unconditionally during training."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
            self.inner_model = model.base_model.model
        else:
            self.inner_model = model
        total_layers = len(self.inner_model.transformer_blocks)
        logger.info("QwenSingleGPUWrapper: %d transformer blocks (no split).", total_layers)

    def _block_checkpoint(self, block, h, e, mask, temb, r0, r1):
        def _fwd(h, e, mask, temb, r0, r1):
            if r0.numel() > r1.numel():
                img_rot, txt_rot = r0, r1
            else:
                img_rot, txt_rot = r1, r0
            return block(
                hidden_states=h, encoder_hidden_states=e, encoder_hidden_states_mask=mask,
                temb=temb, image_rotary_emb=(img_rot, txt_rot),
            )
        return checkpoint(_fwd, h, e, mask, temb, r0, r1, use_reentrant=False)

    def forward(self, hidden_states, encoder_hidden_states, encoder_hidden_states_mask, timestep):
        hidden_states = self.inner_model.img_in(hidden_states)
        timestep = timestep.to(hidden_states.dtype)
        encoder_hidden_states = self.inner_model.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.inner_model.txt_in(encoder_hidden_states)

        temb = self.inner_model.time_text_embed(timestep, hidden_states)
        temb = temb.to(dtype=hidden_states.dtype)

        B = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]
        grid_sq = int(seq_len ** 0.5)
        if grid_sq * grid_sq == seq_len:
            img_shapes = [[(1, grid_sq, grid_sq)]] * B
        else:
            half = seq_len // 2
            grid_sq = int(half ** 0.5)
            img_shapes = [[(2, grid_sq, grid_sq)]] * B

        full_len = encoder_hidden_states.shape[1]
        image_rotary_emb = self.inner_model.pos_embed(img_shapes, max_txt_seq_len=full_len, device=hidden_states.device)
        r0, r1 = image_rotary_emb

        for block in self.inner_model.transformer_blocks:
            if self.training:
                if not hidden_states.requires_grad:
                    hidden_states.requires_grad_(True)
                if not encoder_hidden_states.requires_grad:
                    encoder_hidden_states.requires_grad_(True)
                encoder_hidden_states, hidden_states = self._block_checkpoint(
                    block, hidden_states, encoder_hidden_states, encoder_hidden_states_mask, temb, r0, r1,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=encoder_hidden_states_mask, temb=temb,
                    image_rotary_emb=(r0, r1),
                )

        hidden_states = self.inner_model.norm_out(hidden_states, temb)
        hidden_states = self.inner_model.proj_out(hidden_states)
        return hidden_states


class QwenImageEditAdapter(PipelineAdapter):
    name = "qwen_image_edit"

    RESOLUTION = 1024
    CONDITION_RESOLUTION = 384

    def supported_precisions(self) -> Sequence[str]:
        # "none" (unquantized, ~38GiB) only via the ZeRO-3/device_map
        # baselines, which this first framework version doesn't expose yet
        # (see project_qwen_bull_benchmarks: ZeRO-3 abandoned, device_map
        # kept as a baseline script, not yet ported here). The data_parallel
        # phase here always expects a quantized trainable component.
        return ("nf4", "int8")

    def needs_two_gpus_for_live(self) -> bool:
        return True  # 20.4B DiT + 8.3B VLM don't both fit one 32GB V100 (see project_jsc_paper memory)

    def build_dataset(self, dataset_root: str, max_samples: int, split: str = "train_val",
                       shard_start: int = 0, shard_count: int | None = None) -> MultiViewRotationDataset:
        return MultiViewRotationDataset(
            dataset_root, split=split, max_samples=max_samples,
            shard_start=shard_start, shard_count=shard_count,
        )

    def load_frozen(self, device: str, dtype: torch.dtype) -> FrozenComponents:
        from diffusers import AutoencoderKLQwenImage
        from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLProcessor

        base_model_path = _require_path("QWEN_BASE_MODEL_PATH")

        logger.info("Loading Qwen2.5-VL on %s (%s)...", device, dtype)
        text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_path, subfolder="text_encoder", torch_dtype=dtype, local_files_only=True
        ).to(device)
        text_encoder.requires_grad_(False)
        text_encoder.eval()

        processor = Qwen2VLProcessor.from_pretrained(base_model_path, subfolder="processor", local_files_only=True)

        logger.info("Loading VAE (encode) on %s (%s)...", device, dtype)
        vae = AutoencoderKLQwenImage.from_pretrained(
            base_model_path, subfolder="vae", torch_dtype=dtype, local_files_only=True
        ).to(device)
        vae.requires_grad_(False)
        vae.eval()

        return FrozenComponents({"text_encoder": text_encoder, "processor": processor, "vae": vae})

    def load_trainable(self, device: str, precision: PrecisionConfig, lora_rank: int, lora_alpha: int,
                        lora_dropout: float, gradient_checkpointing: bool) -> torch.nn.Module:
        from diffusers import BitsAndBytesConfig, QwenImageTransformer2DModel
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        fused_transformer_path = _require_path("QWEN_FUSED_TRANSFORMER_PATH")
        compute_dtype = TORCH_DTYPES[precision.resolved_compute_dtype()]

        if precision.quant_type == "nf4":
            logger.info("Loading fused DiT (NF4, compute_dtype=%s) on %s...", compute_dtype, device)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=True,
            )
        elif precision.quant_type == "int8":
            logger.info("Loading fused DiT (int8/LLM.int8(), torch_dtype=%s) on %s...", compute_dtype, device)
            bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
        else:
            raise ValueError(
                f"quant_type={precision.quant_type!r} not supported by this adapter yet "
                f"(supported: {self.supported_precisions()})"
            )

        transformer = QwenImageTransformer2DModel.from_pretrained(
            fused_transformer_path, quantization_config=bnb_config, torch_dtype=compute_dtype,
            device_map=device, attn_implementation="sdpa", local_files_only=True,
        )
        if gradient_checkpointing:
            # No-op under QwenSingleGPUWrapper (see its docstring), kept for
            # compatibility if the base model is ever used unwrapped.
            transformer.enable_gradient_checkpointing()
        transformer = prepare_model_for_kbit_training(transformer, use_gradient_checkpointing=False)
        lora_config = LoraConfig(
            r=lora_rank, lora_alpha=lora_alpha, init_lora_weights=True,
            target_modules=LORA_TARGET_MODULES, lora_dropout=lora_dropout,
        )
        transformer = get_peft_model(transformer, lora_config)
        transformer.print_trainable_parameters()
        return QwenSingleGPUWrapper(transformer)

    def load_scheduler(self) -> Any:
        from diffusers import FlowMatchEulerDiscreteScheduler
        base_model_path = _require_path("QWEN_BASE_MODEL_PATH")
        return FlowMatchEulerDiscreteScheduler.from_pretrained(base_model_path, subfolder="scheduler")

    def encode_sample(self, frozen: FrozenComponents, prompts: List[str], source_pils: List[Any],
                       target_pils: List[Any], device: str, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
        condition_pils = [_pil_to_condition_pil(p, self.CONDITION_RESOLUTION) for p in source_pils]
        prompt_embeds, prompt_embeds_mask = _encode_prompt_live(
            frozen.text_encoder, frozen.processor, prompts, condition_pils, device, dtype
        )

        target_latents_list, source_latents_list = [], []
        for src_pil, tgt_pil in zip(source_pils, target_pils):
            src_t = _pil_to_vae_tensor(src_pil, self.RESOLUTION, device, dtype)
            tgt_t = _pil_to_vae_tensor(tgt_pil, self.RESOLUTION, device, dtype)
            source_latents_list.append(_encode_vae_image_live(frozen.vae, src_t))
            target_latents_list.append(_encode_vae_image_live(frozen.vae, tgt_t))
        source_latents_5d = torch.cat(source_latents_list, dim=0)
        target_latents_5d = torch.cat(target_latents_list, dim=0)
        B, C, _, H_lat, W_lat = source_latents_5d.shape
        source_latents_packed = _pack_latents(source_latents_5d.squeeze(2), B, C, H_lat, W_lat)
        target_latents_packed = _pack_latents(target_latents_5d.squeeze(2), B, C, H_lat, W_lat)

        # Left on `device` deliberately, NOT moved to CPU here: this same
        # dict is used both by the live baseline (which transfers it
        # straight GPU->GPU to the training device) and by precompute.py
        # (which does its own .cpu() right before torch.save). Doing the
        # CPU round-trip inside this method would add a hop the original
        # live benchmark never paid, skewing the live-vs-cached comparison.
        return {
            "target_latents_packed": target_latents_packed.to(dtype),
            "source_latents_packed": source_latents_packed.to(dtype),
            "prompt_embeds": prompt_embeds.to(dtype),
            "prompt_embeds_mask": prompt_embeds_mask,
        }

    def cache_max_seq_len(self, cached_items: List[Dict[str, torch.Tensor]]) -> int:
        return max(item["prompt_embeds"].shape[1] for item in cached_items)

    def collate_cached(self, cached_items: List[Dict[str, torch.Tensor]], max_seq_len: int,
                        dtype: torch.dtype) -> Dict[str, torch.Tensor]:
        target = torch.cat([it["target_latents_packed"] for it in cached_items], dim=0)
        source = torch.cat([it["source_latents_packed"] for it in cached_items], dim=0)

        padded_embeds, padded_masks = [], []
        for it in cached_items:
            pe, pm = it["prompt_embeds"], it["prompt_embeds_mask"]
            curr_len = pe.shape[1]
            if curr_len < max_seq_len:
                pad_len = max_seq_len - curr_len
                pe = F.pad(pe, (0, 0, 0, pad_len), value=0.0)
                pm = F.pad(pm, (0, pad_len), value=0)
            padded_embeds.append(pe)
            padded_masks.append(pm)

        return {
            "target_latents_packed": target.to(dtype),
            "source_latents_packed": source.to(dtype),
            "prompt_embeds": torch.cat(padded_embeds, dim=0).to(dtype),
            "prompt_embeds_mask": torch.cat(padded_masks, dim=0),
        }

    def forward_loss(self, model: torch.nn.Module, batch: Dict[str, torch.Tensor], device: str,
                      compute_dtype: torch.dtype, seed: int) -> torch.Tensor:
        scheduler = self._scheduler_cache()
        target_d = batch["target_latents_packed"]
        source_d = batch["source_latents_packed"]
        prompt_embeds_d = batch["prompt_embeds"]
        prompt_embeds_mask_d = batch["prompt_embeds_mask"]

        bsz = target_d.shape[0]
        gen = torch.Generator(device=device).manual_seed(seed)
        timesteps = torch.randint(
            0, scheduler.config.num_train_timesteps, (bsz,), generator=gen, device=device
        ).long()
        noise = torch.randn(target_d.shape, generator=gen, device=device, dtype=compute_dtype)

        t_norm = (timesteps.float() / scheduler.config.num_train_timesteps).to(compute_dtype).view(-1, 1, 1)
        noisy = (1.0 - t_norm) * target_d + t_norm * noise
        velocity_target = noise - target_d
        timestep_norm = (timesteps.float() / 1000.0).to(compute_dtype)

        latent_model_input = torch.cat([noisy, source_d], dim=1)
        v_pred = model(latent_model_input, prompt_embeds_d, prompt_embeds_mask_d, timestep_norm)
        v_pred = v_pred[:, : velocity_target.shape[1]]

        return F.mse_loss(v_pred.float(), velocity_target.float())

    def _scheduler_cache(self):
        if not hasattr(self, "_scheduler"):
            self._scheduler = self.load_scheduler()
        return self._scheduler
