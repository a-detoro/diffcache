"""Phase 2: single-GPU reference, training on the cache `precompute.py`
wrote. Pipeline-agnostic port of
`training_from_jupyter/finetune_precomputed_bench.py` -- no frozen
encoders loaded here, only the trainable DiT+LoRA, which is exactly the
VRAM saving this methodology's Step 1 is meant to buy.
"""
from __future__ import annotations

import logging
import os
import time

import torch

from ..config import RunConfig
from ..dtypes import TORCH_DTYPES
from ..hardware import apply_quirks
from ..pipelines import get_adapter
from ..report import write_report
from ..timing import mean_excluding_first
from ._cache_io import load_and_collate_batch, load_precomputed_index, move_batch_to_device, resolve_gradient_checkpointing

logger = logging.getLogger(__name__)


def run(run_config: RunConfig, device: str = "cuda:0") -> dict:
    adapter = get_adapter(run_config.pipeline)
    apply_quirks(run_config.precision, device)
    dtype = TORCH_DTYPES[run_config.precision.storage_dtype]
    compute_dtype = TORCH_DTYPES[run_config.precision.resolved_compute_dtype()]

    precomputed_dir = run_config.precomputed_dir or os.path.join(os.getcwd(), "precomputed_bench")
    paths, cache_index = load_precomputed_index(adapter, precomputed_dir)
    n_samples = len(paths)

    use_gc = resolve_gradient_checkpointing(run_config)
    model = adapter.load_trainable(
        device, run_config.precision, run_config.lora_rank, run_config.lora_alpha,
        run_config.lora_dropout, use_gc,
    )
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=run_config.learning_rate, eps=1e-6)

    logger.info("DiT (quantize=%s, 1 GPU, no frozen encoders resident) -> %s", run_config.precision.quant_type, device)
    logger.info("Steps: %d | batch_size=%d | precomputed samples=%d", run_config.steps, run_config.batch_size, n_samples)

    torch.cuda.reset_peak_memory_stats(device)
    model.train()
    timings_ms, vram_alloc, vram_reserved = [], [], []

    for step in range(run_config.steps):
        batch_paths = [paths[(step * run_config.batch_size + i) % n_samples] for i in range(run_config.batch_size)]

        t_step0 = time.perf_counter()
        batch_cpu = load_and_collate_batch(adapter, batch_paths, cache_index, dtype)
        t_disk_read = (time.perf_counter() - t_step0) * 1000.0

        t0 = time.perf_counter()
        batch_device = move_batch_to_device(batch_cpu, device, compute_dtype)
        torch.cuda.synchronize(device)
        t_to_gpu = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        optimizer.zero_grad()
        loss = adapter.forward_loss(model, batch_device, device, compute_dtype, seed=42 + step)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
        torch.cuda.synchronize(device)
        t_fwd_bwd = (time.perf_counter() - t0) * 1000.0

        t_total = (time.perf_counter() - t_step0) * 1000.0
        timings_ms.append((t_disk_read, t_to_gpu, t_fwd_bwd, t_total))
        vram_alloc.append(torch.cuda.memory_allocated(device) / 1e9)
        vram_reserved.append(torch.cuda.memory_reserved(device) / 1e9)

        logger.info(
            "[step %d] loss=%.6f | disk_read=%.0fms to_gpu=%.0fms fwd_bwd=%.0fms TOTAL=%.0fms | VRAM %.2f/%.2fGB",
            step, loss.item(), t_disk_read, t_to_gpu, t_fwd_bwd, t_total, vram_alloc[-1], vram_reserved[-1],
        )

    n = len(timings_ms)
    totals = [sum(t[i] for t in timings_ms) for i in range(4)]
    tails = [[t[i] for t in timings_ms] for i in range(4)]

    report_base = run_config.report_path or os.path.join(
        os.getcwd(), "reports", f"finetune_cached_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    return write_report(
        report_base,
        title=f"cachezero -- finetune_cached, 1 GPU ({run_config.pipeline}, quantize={run_config.precision.quant_type})",
        config={
            "pipeline": run_config.pipeline, "steps": n, "batch_size": run_config.batch_size,
            "precomputed_samples": n_samples, "device_name": torch.cuda.get_device_name(device),
            "quantize": run_config.precision.quant_type, "gradient_checkpointing": use_gc,
            "dtype": run_config.precision.storage_dtype,
            "compute_dtype": run_config.precision.resolved_compute_dtype(),
        },
        timing_ms={
            "disk_read_avg": totals[0] / n, "to_gpu_avg": totals[1] / n,
            "fwd_bwd_avg": totals[2] / n, "total_step_avg": totals[3] / n,
            "total_step_avg_excl_first": mean_excluding_first(tails[3]),
            "total_run_ms": totals[3],
        },
        vram_gb={
            "avg_allocated": sum(vram_alloc) / n, "avg_reserved": sum(vram_reserved) / n,
            "peak_allocated": torch.cuda.max_memory_allocated(device) / 1e9,
            "peak_reserved": torch.cuda.max_memory_reserved(device) / 1e9,
        },
        per_step=[
            {"step": i, "disk_read_ms": tails[0][i], "to_gpu_ms": tails[1][i], "fwd_bwd_ms": tails[2][i],
             "total_ms": tails[3][i], "allocated_gb": vram_alloc[i], "reserved_gb": vram_reserved[i]}
            for i in range(n)
        ],
    )
