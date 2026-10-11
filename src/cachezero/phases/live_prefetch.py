"""Baseline: "live with prefetch" -- same as `live.py` but the next sample's
encode overlaps the current step's DiT training via `prefetch.EncoderPrefetcher`
(double-buffered across CUDA streams, one dedicated producer thread). This is
the "second baseline" referenced in project_jsc_tex_update_20260924's naming
convention. Always needs 2 distinct GPUs -- that's the entire point (each
GPU gets its own uninterrupted stream of work); unlike `live.py`, there's no
single-GPU variant of this phase.
"""
from __future__ import annotations

import itertools
import logging
import os
import random
import time

import torch

from ..config import RunConfig
from ..datasets.base import collate_pils
from ..dtypes import TORCH_DTYPES
from ..hardware import apply_quirks
from ..pipelines import get_adapter
from ..prefetch import EncoderPrefetcher, StepRecorder
from ..report import write_report
from ..timing import mean_excluding_first
from ._cache_io import move_batch_to_device, resolve_gradient_checkpointing

logger = logging.getLogger(__name__)


def run(run_config: RunConfig, device_encode: str = "cuda:0", device_train: str = "cuda:1") -> dict:
    if device_encode == device_train:
        raise ValueError("phase=live_prefetch requires two distinct GPUs (device_encode != device_train)")

    adapter = get_adapter(run_config.pipeline)
    apply_quirks(run_config.precision, device_train)
    dtype = TORCH_DTYPES[run_config.precision.storage_dtype]
    compute_dtype = TORCH_DTYPES[run_config.precision.resolved_compute_dtype()]

    if not run_config.dataset_root:
        raise ValueError("RunConfig.dataset_root is required for phase=live_prefetch")
    dataset = adapter.build_dataset(run_config.dataset_root, run_config.num_samples)

    use_gc = resolve_gradient_checkpointing(run_config)
    frozen = adapter.load_frozen(device_encode, dtype)
    model = adapter.load_trainable(
        device_train, run_config.precision, run_config.lora_rank, run_config.lora_alpha,
        run_config.lora_dropout, use_gc,
    )
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=run_config.learning_rate, eps=1e-6)

    rng = random.Random(42)
    pool = list(range(len(dataset)))
    rng.shuffle(pool)
    pool_cycle = itertools.cycle(pool)
    schedule = [(0, [next(pool_cycle) for _ in range(run_config.batch_size)]) for _ in range(run_config.steps)]

    def load_fn(indices):
        return collate_pils([dataset[i] for i in indices])

    def encode_fn(samples):
        return adapter.encode_sample(frozen, samples["prompts"], samples["source_pils"], samples["target_pils"],
                                      device_encode, dtype)

    prefetcher = EncoderPrefetcher(schedule, load_fn, encode_fn, device_encode, device_train, depth=1)
    recorder = StepRecorder(num_epochs=1)

    logger.info("encoders -> %s (prefetching) | trainable DiT -> %s", device_encode, device_train)
    logger.info("Steps: %d | batch_size=%d", run_config.steps, run_config.batch_size)

    torch.cuda.reset_peak_memory_stats(device_encode)
    torch.cuda.reset_peak_memory_stats(device_train)
    model.train()
    recorder.start()

    for step, (epoch, batch, encode_ms, copy_ms, wait_ms) in enumerate(prefetcher):
        batch_device = move_batch_to_device(batch, device_train, compute_dtype)

        t0 = time.perf_counter()
        optimizer.zero_grad()
        loss = adapter.forward_loss(model, batch_device, device_train, compute_dtype, seed=42 + step)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
        torch.cuda.synchronize(device_train)
        t_train_ms = (time.perf_counter() - t0) * 1000.0

        recorder.add(epoch, t_train_ms, encode_ms, copy_ms, wait_ms, loss.item())
        logger.info(
            "[step %d] loss=%.6f | train=%.0fms encode=%.0fms copy=%.0fms wait=%.0fms",
            step, loss.item(), t_train_ms, encode_ms, copy_ms, wait_ms,
        )

    def epoch_stats(step_wall, losses, epoch):
        return {
            "epoch": epoch, "steps": len(step_wall),
            "mean_step_wall_ms_excl_first": mean_excluding_first(step_wall),
            "mean_loss": sum(losses) / len(losses),
        }

    epochs, timing, losses = recorder.report(epoch_stats)

    report_base = run_config.report_path or os.path.join(
        os.getcwd(), "reports", f"live_prefetch_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    return write_report(
        report_base,
        title=f"cachezero -- live baseline with prefetch ({run_config.pipeline})",
        config={
            "pipeline": run_config.pipeline, "steps": run_config.steps, "batch_size": run_config.batch_size,
            "device_encode": device_encode, "device_train": device_train,
            "quantize": run_config.precision.quant_type, "gradient_checkpointing": use_gc,
            "dtype": run_config.precision.storage_dtype,
            "compute_dtype": run_config.precision.resolved_compute_dtype(),
        },
        timing_ms={**timing, "epoch_stats": epochs},
        vram_gb={
            "peak_encode_allocated": torch.cuda.max_memory_allocated(device_encode) / 1e9,
            "peak_train_allocated": torch.cuda.max_memory_allocated(device_train) / 1e9,
        },
        per_step=[{"step": i, "loss": losses[i]} for i in range(len(losses))],
    )
