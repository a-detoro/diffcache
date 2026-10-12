"""DeepGen 1.0 adapter -- ported from
`DeepGen/precompute_live_zero2_bench/deepgen_bench_utils.py` on PopOS-IA.

Architecture (RGB->thermal task on LLVIP, same prompt/dataset as the other
2 new pipelines): 3B VLM (Qwen2.5-VL-3B, frozen) + SCB connector (0.4324B,
frozen, bridges multi-layer VLM hidden states + learnable "meta queries" to
DiT conditioning) + SD3 DiT ("UniPic2-SD3.5M-Kontext-2B", 2.2432B,
QLoRA-trainable) + small frozen VAE. Loss is the same manual logit-normal
rectified-flow scheme as FLUX's (`FlowMatchEulerDiscreteScheduler`-style),
NOT OmniGen2's `transport` module. Resolution is fixed 512x512 (DeepGen's
own native default), unlike the other 2 pipelines' ~1280x1024.

Requires the `DEEPGEN_REPO_PATH` env var (a checkout of the
`deepgenteam/deepgen` repo containing `deepgen_pipeline.py`, e.g.
`/home/antonio/Proyectos/DeepGen`) -- `SD3Transformer2DModel`/
`DeepGenConnector` are custom classes defined there, not in `diffusers`.

Like FLUX/OmniGen2, `encode_pair` only takes ONE sample at a time -- this
adapter loops internally and reuses `_combine`. Two of its batch fields
(`cond_latents`/`target_latents`) use DIFFERENT key names in the raw
per-sample form (`cond_latent`/`target_latent`, singular) than in the
combined batch form (plural) -- `_latent_field` below tries both.
"""
from __future__ import annotations

import json
import os
import sys
from functools import partial
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image

from ..config import PrecisionConfig
from ..datasets.llvip import LLVIPPairsDataset
from ..dtypes import TORCH_DTYPES
from .base import FrozenComponents, PipelineAdapter

DEFAULT_PROMPT = "Convert this image to a thermal infrared image, showing heat signatures in false color."
DEFAULT_MODEL_PATH = "deepgenteam/DeepGen-1.0-diffusers"

LORA_TARGET_MODULES = ["to_q", "to_k", "to_v", "to_out.0"]

HEIGHT = 512
WIDTH = 512


def _require_deepgen_repo() -> None:
    repo_path = os.environ.get("DEEPGEN_REPO_PATH")
    if not repo_path:
        raise RuntimeError(
            "DEEPGEN_REPO_PATH is not set -- the DeepGen adapter needs it to point at a "
            "checkout of the DeepGen repo containing deepgen_pipeline.py."
        )
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)


def _resolve_local_path(model_path: str) -> str:
    if os.path.isdir(model_path):
        return model_path
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=model_path)


def _pil_to_model_tensor(img: Image.Image, height: int, width: int, dtype, device) -> torch.Tensor:
    img = img.convert("RGB").resize((width, height))
    pv = torch.from_numpy(np.array(img)).float() / 255.0
    pv = 2 * pv - 1
    pv = rearrange(pv, "h w c -> c h w")
    return pv.to(dtype=dtype, device=device)


def _multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


@torch.no_grad()
def _get_semantic_features(encoders: Dict[str, Any], pixel_values: torch.Tensor, resize: bool = True):
    lmm = encoders["lmm"]
    device = encoders["device"]
    vit_mean, vit_std = encoders["vit_mean"], encoders["vit_std"]

    pixel_values = (pixel_values + 1.0) / 2
    pixel_values = pixel_values - vit_mean.view(1, 3, 1, 1)
    pixel_values = pixel_values / vit_std.view(1, 3, 1, 1)
    if resize:
        pixel_values = F.interpolate(pixel_values, size=(448, 448), mode="bilinear")
    b, c, h, w = pixel_values.shape

    patch_size = lmm.config.vision_config.patch_size
    spatial_merge_size = lmm.config.vision_config.spatial_merge_size
    temporal_patch_size = lmm.config.vision_config.temporal_patch_size

    pixel_values = pixel_values[:, None].expand(b, temporal_patch_size, c, h, w)
    grid_t = 1
    grid_h, grid_w = h // patch_size, w // patch_size

    pixel_values = pixel_values.reshape(
        b, grid_t, temporal_patch_size, c,
        grid_h // spatial_merge_size, spatial_merge_size, patch_size,
        grid_w // spatial_merge_size, spatial_merge_size, patch_size)
    pixel_values = rearrange(pixel_values, "b t tp c h m p w n q -> (b t h w m n) (c tp p q)")

    image_grid_thw = torch.tensor([(grid_t, grid_h, grid_w)] * b).to(device).long()
    image_embeds = lmm.model.visual(pixel_values, grid_thw=image_grid_thw).pooler_output
    image_embeds = rearrange(image_embeds, "(b l) d -> b l d", b=b)
    return image_embeds, image_grid_thw


@torch.no_grad()
def _get_semantic_features_dynamic(encoders: Dict[str, Any], pixel_values_list):
    pixel_values_list = [F.interpolate(p[None], scale_factor=28 / 32, mode="bilinear") for p in pixel_values_list]
    image_embeds, image_grid_thw = _multi_apply(
        lambda p: _get_semantic_features(encoders, p, resize=False), pixel_values_list)
    image_embeds = [x[0] for x in image_embeds]
    image_grid_thw = torch.cat(image_grid_thw, dim=0)
    return image_embeds, image_grid_thw


def _prepare_image2image_prompts(encoders: Dict[str, Any], texts, num_refs, ref_lens):
    tokenizer = encoders["tokenizer"]
    prompt_template = encoders["prompt_template"]
    device = encoders["device"]

    prompts = []
    cnt = 0
    for text, num_ref in zip(texts, num_refs):
        image_tokens = ""
        for _ in range(num_ref):
            image_tokens += (prompt_template["IMG_START_TOKEN"] +
                              prompt_template["IMG_CONTEXT_TOKEN"] * ref_lens[cnt] +
                              prompt_template["IMG_END_TOKEN"])
            cnt += 1
        prompts.append(prompt_template["INSTRUCTION"].format(input=f"{image_tokens}\n{text}"))
    return tokenizer(
        prompts, add_special_tokens=True, return_tensors="pt", padding=True, padding_side="left"
    ).to(device)


def _prepare_forward_input(encoders: Dict[str, Any], query_embeds, input_ids=None, image_embeds=None,
                            image_grid_thw=None, attention_mask=None):
    lmm = encoders["lmm"]
    llm = lmm.model.language_model
    image_token_id = encoders["image_token_id"]
    device = encoders["device"]
    dtype = encoders["dtype"]

    b, l, _ = query_embeds.shape
    attention_mask = attention_mask.to(device=device, dtype=torch.bool)
    input_ids = torch.cat([input_ids, input_ids.new_zeros(b, l)], dim=1)
    attention_mask = torch.cat([attention_mask, attention_mask.new_ones(b, l)], dim=1)

    mm_token_type_ids = (input_ids == image_token_id).to(torch.int)
    position_ids, _ = lmm.model.get_rope_index(
        input_ids=input_ids, mm_token_type_ids=mm_token_type_ids, image_grid_thw=image_grid_thw,
        video_grid_thw=None, second_per_grid_ts=None, attention_mask=attention_mask)

    input_ids_txt = input_ids[:, :-l]
    if image_embeds is None:
        inputs_embeds = llm.get_input_embeddings()(input_ids_txt)
    else:
        inputs_embeds = torch.zeros(*input_ids_txt.shape, llm.config.hidden_size, device=device, dtype=dtype)
        inputs_embeds[input_ids_txt == image_token_id] = image_embeds.contiguous().view(-1, llm.config.hidden_size)
        inputs_embeds[input_ids_txt != image_token_id] = \
            llm.get_input_embeddings()(input_ids_txt[input_ids_txt != image_token_id])
    inputs_embeds = torch.cat([inputs_embeds, query_embeds], dim=1)

    return dict(inputs_embeds=inputs_embeds, attention_mask=attention_mask, position_ids=position_ids)


@torch.no_grad()
def _pixels_to_latents(encoders: Dict[str, Any], x: torch.Tensor) -> torch.Tensor:
    vae = encoders["vae"]
    z = vae.encode(x.to(dtype=next(vae.parameters()).dtype)).latent_dist.sample()
    z = (z - vae.config.shift_factor) * vae.config.scaling_factor
    return z


@torch.no_grad()
def _encode_pair(encoders: Dict[str, Any], source_pil, target_pil, prompt: str, device, dtype,
                  height: int = HEIGHT, width: int = WIDTH) -> Dict[str, Any]:
    src_tensor = _pil_to_model_tensor(source_pil, height, width, dtype, device)
    tgt_tensor = _pil_to_model_tensor(target_pil, height, width, dtype, device)

    image_embeds, image_grid_thw = _get_semantic_features_dynamic(encoders, [src_tensor])
    ref_lens = [len(x) for x in image_embeds]

    text_inputs = _prepare_image2image_prompts(encoders, [prompt], num_refs=[1], ref_lens=ref_lens)

    connector_module = encoders["connector_module"]
    query_embeds = connector_module.meta_queries[None].expand(1, connector_module.num_queries, -1)
    inputs = _prepare_forward_input(
        encoders, query_embeds=query_embeds,
        input_ids=text_inputs["input_ids"], attention_mask=text_inputs["attention_mask"],
        image_embeds=torch.cat(image_embeds), image_grid_thw=image_grid_thw,
    )

    lmm = encoders["lmm"]
    output = lmm.model.language_model(**inputs, return_dict=True, output_hidden_states=True)

    hidden_states = output.hidden_states
    num_layers = len(hidden_states) - 1
    selected_layers = list(range(num_layers - 1, 0, -6))
    merged_hidden = torch.cat([hidden_states[i] for i in selected_layers], dim=-1)
    pooled_out, seq_out = connector_module.llm2dit(merged_hidden)

    cond_latent = _pixels_to_latents(encoders, src_tensor[None])[0]
    target_latent = _pixels_to_latents(encoders, tgt_tensor[None])[0]

    return {
        "seq_out": seq_out[0].to(dtype=dtype),
        "pooled_out": pooled_out[0].to(dtype=dtype),
        "cond_latent": cond_latent.to(dtype=dtype),
        "target_latent": target_latent.to(dtype=dtype),
    }


class DeepGenAdapter(PipelineAdapter):
    name = "deepgen"

    def supported_precisions(self) -> Sequence[str]:
        return ("none", "nf4")

    def needs_two_gpus_for_live(self) -> bool:
        # Confirmed empirically (project_deepgen_popos_bench memory): DeepGen's
        # ~5B total (VLM+connector+DiT+VAE) fits one 16GB card at ~14.7GB peak
        # (bf16, batch=1, gradient checkpointing) -- unlike FLUX/OmniGen2's
        # stubs, this one was already correct.
        return False

    def build_dataset(self, dataset_root: str, max_samples: int, split: str = "train_val",
                       shard_start: int = 0, shard_count: int | None = None) -> LLVIPPairsDataset:
        return LLVIPPairsDataset(
            dataset_root, DEFAULT_PROMPT, max_samples,
            shard_start=shard_start, shard_count=shard_count,
        )

    def load_frozen(self, device: str, dtype: torch.dtype) -> FrozenComponents:
        _require_deepgen_repo()
        from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
        from safetensors.torch import load_file
        from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

        from deepgen_pipeline import IMAGE_MEAN, IMAGE_STD, DeepGenConnector

        path = _resolve_local_path(DEFAULT_MODEL_PATH)

        vae = AutoencoderKL.from_pretrained(path, subfolder="vae", torch_dtype=dtype)
        vae.to(device).eval()
        vae.requires_grad_(False)

        vlm_path = os.path.join(path, "vlm")
        try:
            lmm = Qwen2_5_VLForConditionalGeneration.from_pretrained(vlm_path, torch_dtype=dtype, attn_implementation="sdpa")
        except Exception:
            lmm = Qwen2_5_VLForConditionalGeneration.from_pretrained(vlm_path, torch_dtype=dtype)
        lmm.to(device).eval()
        lmm.requires_grad_(False)

        tokenizer_path = os.path.join(path, "tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, padding_side="right")

        connector_dir = os.path.join(path, "connector")
        with open(os.path.join(connector_dir, "config.json")) as f:
            connector_cfg = json.load(f)
        conn_cfg = connector_cfg["connector"].copy()
        conn_cfg["_attn_implementation"] = "sdpa"
        connector_module = DeepGenConnector(
            connector_config=conn_cfg,
            num_queries=connector_cfg["num_queries"],
            llm_hidden_size=connector_cfg["llm_hidden_size"],
            projector_1_in=connector_cfg["projector_1_in"], projector_1_out=connector_cfg["projector_1_out"],
            projector_2_in=connector_cfg["projector_2_in"], projector_2_out=connector_cfg["projector_2_out"],
            projector_3_in=connector_cfg["projector_3_in"], projector_3_out=connector_cfg["projector_3_out"],
        )
        connector_state = load_file(os.path.join(connector_dir, "model.safetensors"))
        connector_module.load_state_dict(connector_state, strict=True)
        connector_module = connector_module.to(device=device, dtype=dtype).eval()
        connector_module.requires_grad_(False)

        with open(os.path.join(path, "prompt_template.json")) as f:
            prompt_template = json.load(f)
        image_token_id = tokenizer.convert_tokens_to_ids(prompt_template["IMG_CONTEXT_TOKEN"])

        return FrozenComponents({
            "vae": vae, "lmm": lmm, "tokenizer": tokenizer, "connector_module": connector_module,
            "prompt_template": prompt_template, "image_token_id": image_token_id,
            "vit_mean": torch.tensor(IMAGE_MEAN, device=device), "vit_std": torch.tensor(IMAGE_STD, device=device),
            "device": device, "dtype": dtype,
        })

    def load_trainable(self, device: str, precision: PrecisionConfig, lora_rank: int, lora_alpha: int,
                        lora_dropout: float, gradient_checkpointing: bool) -> torch.nn.Module:
        _require_deepgen_repo()
        from peft import LoraConfig

        from deepgen_pipeline import SD3Transformer2DModel

        compute_dtype = TORCH_DTYPES[precision.resolved_compute_dtype()]
        path = _resolve_local_path(DEFAULT_MODEL_PATH)

        if precision.quant_type == "nf4":
            from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig

            quant_config = DiffusersBitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=compute_dtype,
            )
            model = SD3Transformer2DModel.from_pretrained(
                path, subfolder="transformer",
                quantization_config=quant_config, torch_dtype=compute_dtype, device_map={"": device},
            )
        elif precision.quant_type == "none":
            model = SD3Transformer2DModel.from_pretrained(
                path, subfolder="transformer", torch_dtype=compute_dtype, device_map={"": device},
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
        return model

    def load_scheduler(self) -> Any:
        # Manual logit-normal flow-matching (same family as FLUX's) -- no
        # external scheduler object is consulted at loss time.
        return None

    @staticmethod
    def _unwrap_to_tensor(value: Any) -> Any:
        """See the identical helper in `omnigen2.py` for why this is
        needed: a disk-reloaded sample has already been through one
        `_combine` call (as a "batch of 1"), so its list-valued fields
        arrive here still wrapped."""
        while isinstance(value, list):
            assert len(value) == 1, f"expected a length-1 wrapper, got length {len(value)}"
            value = value[0]
        return value

    @staticmethod
    def _latent_field(item: Dict[str, Any], singular_key: str, plural_key: str) -> torch.Tensor:
        """`cond_latent`/`target_latent` (singular) in a raw `_encode_pair`
        result vs. `cond_latents`/`target_latents` (plural) in an
        already-combined batch dict -- the original `deepgen_bench_utils.py`
        genuinely uses different key names for the two representations."""
        value = item[singular_key] if singular_key in item else item[plural_key]
        return DeepGenAdapter._unwrap_to_tensor(value)

    @staticmethod
    def _to_batched(t: torch.Tensor, unbatched_ndim: int) -> torch.Tensor:
        """`seq_out`/`pooled_out` are stacked (not list-wrapped) -- a raw
        `_encode_pair` result has no batch dim yet (ndim == unbatched_ndim),
        a disk-reloaded one already does (ndim == unbatched_ndim + 1)."""
        return t if t.dim() == unbatched_ndim + 1 else t.unsqueeze(0)

    def _combine(self, items: List[Dict[str, Any]], dtype: torch.dtype) -> Dict[str, Any]:
        seq_out = torch.cat([self._to_batched(it["seq_out"], 2) for it in items], dim=0).to(dtype=dtype)
        pooled_out = torch.cat([self._to_batched(it["pooled_out"], 1) for it in items], dim=0).to(dtype=dtype)
        cond_latents = [[self._latent_field(it, "cond_latent", "cond_latents").to(dtype=dtype)] for it in items]
        target_latents = [self._latent_field(it, "target_latent", "target_latents").to(dtype=dtype) for it in items]
        return {
            "seq_out": seq_out, "pooled_out": pooled_out,
            "cond_latents": cond_latents, "target_latents": target_latents,
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
        target_latents = [t.to(device=device, dtype=compute_dtype) for t in batch["target_latents"]]
        cond_latents = [[t.to(device=device, dtype=compute_dtype) for t in refs] for refs in batch["cond_latents"]]
        seq_out = batch["seq_out"].to(device=device, dtype=compute_dtype)
        pooled_out = batch["pooled_out"].to(device=device, dtype=compute_dtype)
        B = len(target_latents)

        generator = torch.Generator(device=device).manual_seed(seed)
        noise = [torch.randn(t.shape, generator=generator, device=device, dtype=compute_dtype) for t in target_latents]
        u = torch.randn(B, generator=generator, device=device)
        t = torch.sigmoid(u).to(dtype=compute_dtype)

        noisy_latents = [(1.0 - t[i]) * target_latents[i] + t[i] * noise[i] for i in range(B)]
        velocity_target = torch.stack([noise[i] - target_latents[i] for i in range(B)])
        timestep = t * 1000.0  # FlowMatchEulerDiscreteScheduler uses sigmas in [0, 1000]

        noise_pred = model(
            hidden_states=noisy_latents, cond_hidden_states=cond_latents,
            encoder_hidden_states=seq_out, pooled_projections=pooled_out,
            timestep=timestep, return_dict=False,
        )[0]
        if isinstance(noise_pred, list):
            noise_pred = torch.stack(noise_pred)

        return F.mse_loss(noise_pred.float(), velocity_target.float())
