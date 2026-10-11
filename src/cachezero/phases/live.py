"""Baseline: "live", no caching -- every step re-runs the frozen encoders.
Pipeline-agnostic port of `training_from_jupyter/finetune_live_multigpu.py`.

Whether this needs 1 or 2 GPUs is decided by the adapter
(`needs_two_gpus_for_live()`), not a flag the caller has to get right: Qwen's
20.4B DiT + 7B VLM don't fit one 32GB V100 together, so it always needs 2;
DeepGen's ~5B total fits one 16GB card, so it can run on 1 (see
project_deepgen_popos_bench memory for the single-GPU live arm that
motivated this distinction).
"""
from __future__ import annotations

import logging
import os
import time

import torch
from torch.utils.data import DataLoader

from ..config import RunConfig
from ..datasets.base import collate_pils
from ..dtypes import TORCH_DTYPES
from ..hardware import apply_quirks
from ..pipelines import get_adapter
from ..report import write_report
from ..timing import mean_excluding_first
from ._cache_io import move_batch_to_device, resolve_gradient_checkpointing

logger = logging.getLogger(__name__)


def run(run_config: RunConfig, device_encode: str = "cuda:0", device_train: str | None = None) -> dict:
    adapter = get_adapter(run_config.pipeline)
    two_gpu = adapter.needs_two_gpus_for_live()
    device_train = device_train or ("cuda:1" if two_gpu else device_encode)

    apply_quirks(run_config.precision, device_train)
    dtype = TORCH_DTYPES[run_config.precision.storage_dtype]
    compute_dtype = TORCH_DTYPES[run_config.precision.resolved_compute_dtype()]

    if not run_config.dataset_root:
        raise ValueError("RunConfig.dataset_root is required for phase=live")

    dataset = adapter.build_dataset(run_config.dataset_root, run_config.num_samples)
    dataloader = DataLoader(
        dataset, batch_size=run_config.batch_size, shuffle=True, drop_last=True,
        collate_fn=collate_pils, num_workers=2,
    )

    use_gc = resolve_gradient_checkpointing(run_config)
    frozen = adapter.load_frozen(device_encode, dtype)
    model = adapter.load_trainable(
        device_train, run_config.precision, run_config.lora_rank, run_config.lora_alpha,
        run_config.lora_dropout, use_gc,
    )
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=run_config.learning_rate, eps=1e-6)

    logger.info("encoders -> %s | trainable DiT -> %s (two_gpu=%s)", device_encode, device_train, two_gpu)
    logger.info("Steps: %d | batch_size=%d", run_config.steps, run_config.batch_size)

    torch.cuda.reset_peak_memory_stats(device_encode)
    torch.cuda.reset_peak_memory_stats(device_train)

    model.train()
    data_iter = iter(dataloader)
    timings_ms, vram_samples = [], []

    for step in range(run_config.steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        t_step0 = time.perf_counter()

        torch.cuda.synchronize(device_encode)
        t0 = time.perf_counter()
        encoded = adapter.encode_sample(frozen, batch["prompts"], batch["source_pils"], batch["target_pils"],
                                         device_encode, dtype)
        torch.cuda.synchronize(device_encode)
        t_encode = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        batch_device = move_batch_to_device(encoded, device_train, compute_dtype)
        torch.cuda.synchronize(device_train)
        t_transfer = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        optimizer.zero_grad()
        loss = adapter.forward_loss(model, batch_device, device_train, compute_dtype, seed=42 + step)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
        torch.cuda.synchronize(device_train)
        t_fwd_bwd = (time.perf_counter() - t0) * 1000.0

        t_total = (time.perf_counter() - t_step0) * 1000.0
        timings_ms.append((t_encode, t_transfer, t_fwd_bwd, t_total))
        vram_samples.append((
            torch.cuda.memory_allocated(device_encode) / 1e9, torch.cuda.memory_reserved(device_encode) / 1e9,
            torch.cuda.memory_allocated(device_train) / 1e9, torch.cuda.memory_reserved(device_train) / 1e9,
        ))

        logger.info(
            "[step %d] loss=%.6f | encode=%.0fms transfer=%.0fms fwd_bwd=%.0fms TOTAL=%.0fms",
            step, loss.item(), t_encode, t_transfer, t_fwd_bwd, t_total,
        )

    n = len(timings_ms)
    totals = [sum(t[i] for t in timings_ms) for i in range(4)]
    tails = [[t[i] for t in timings_ms] for i in range(4)]
    vram_avg = [sum(v[i] for v in vram_samples) / n for i in range(4)]

    report_base = run_config.report_path or os.path.join(
        os.getcwd(), "reports", f"live_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    return write_report(
        report_base,
        title=f"cachezero -- live baseline ({run_config.pipeline}, two_gpu={two_gpu})",
        config={
            "pipeline": run_config.pipeline, "steps": n, "batch_size": run_config.batch_size,
            "device_encode": device_encode, "device_train": device_train,
            "quantize": run_config.precision.quant_type, "gradient_checkpointing": use_gc,
            "dtype": run_config.precision.storage_dtype,
            "compute_dtype": run_config.precision.resolved_compute_dtype(),
        },
        timing_ms={
            "encode_avg": totals[0] / n, "transfer_avg": totals[1] / n,
            "fwd_bwd_avg": totals[2] / n, "total_step_avg": totals[3] / n,
            "total_step_avg_excl_first": mean_excluding_first(tails[3]),
            "total_run_ms": totals[3],
        },
        vram_gb={
            "avg_encode_allocated": vram_avg[0], "avg_encode_reserved": vram_avg[1],
            "avg_train_allocated": vram_avg[2], "avg_train_reserved": vram_avg[3],
            "peak_encode_allocated": torch.cuda.max_memory_allocated(device_encode) / 1e9,
            "peak_train_allocated": torch.cuda.max_memory_allocated(device_train) / 1e9,
        },
        per_step=[
            {"step": i, "encode_ms": tails[0][i], "transfer_ms": tails[1][i], "fwd_bwd_ms": tails[2][i],
             "total_ms": tails[3][i]}
            for i in range(n)
        ],
    )
