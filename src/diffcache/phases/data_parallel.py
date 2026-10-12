"""Phase 3: the proposed methodology -- real data parallelism across every
GPU the precompute step frees, training on the cache `precompute.py` wrote.
Pipeline-agnostic port of `training_from_jupyter/finetune_zero_bench.py`,
generalized over the backend (`parallel.zero.ZeroBackend` by default,
`parallel.ddp.DDPBackend` as the alternative) via `ParallelBackend`/
`ParallelEngine` -- see `parallel/base.py`.

Launch with the launcher matching `run_config.parallel.backend`:
    deepspeed --num_gpus=N -m diffcache.cli run --parallel zero2 ...
    torchrun --nproc_per_node=N -m diffcache.cli run --parallel ddp ...
Both set RANK/WORLD_SIZE/LOCAL_RANK, which this phase and both backends read
from the environment.
"""
from __future__ import annotations

import logging
import os
import time

import torch

from ..config import RunConfig
from ..dtypes import TORCH_DTYPES
from ..hardware import apply_quirks
from ..parallel import get_backend
from ..pipelines import get_adapter
from ..report import write_report
from ..timing import mean_excluding_first
from ._cache_io import load_and_collate_batch, load_precomputed_index, move_batch_to_device, resolve_gradient_checkpointing

logger = logging.getLogger(__name__)


def run(run_config: RunConfig) -> dict:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(device)

    adapter = get_adapter(run_config.pipeline)
    apply_quirks(run_config.precision, device)
    dtype = TORCH_DTYPES[run_config.precision.storage_dtype]
    compute_dtype = TORCH_DTYPES[run_config.precision.resolved_compute_dtype()]

    precomputed_dir = run_config.precomputed_dir or os.path.join(os.getcwd(), "precomputed_bench")
    paths, max_seq_len = load_precomputed_index(adapter, precomputed_dir)
    n_samples = len(paths)

    use_gc = resolve_gradient_checkpointing(run_config)
    model = adapter.load_trainable(
        device, run_config.precision, run_config.lora_rank, run_config.lora_alpha,
        run_config.lora_dropout, use_gc,
    )
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    model.train()

    backend = get_backend(run_config.parallel.backend)
    engine = backend.setup(model, trainable_params, run_config)

    if engine.is_main:
        logger.info(
            "backend=%s | world_size=%d | batch_size_per_gpu=%d (global=%d)",
            run_config.parallel.backend, engine.world_size, run_config.parallel.batch_size_per_gpu,
            run_config.parallel.batch_size_per_gpu * engine.world_size,
        )
        logger.info("Steps: %d | precomputed samples: %d", run_config.steps, n_samples)

    torch.cuda.reset_peak_memory_stats(device)
    timings_ms, vram_alloc, vram_reserved = [], [], []

    for step in range(run_config.steps):
        batch_paths = backend.shard_batch(paths, step, engine.rank, engine.world_size,
                                           run_config.parallel.batch_size_per_gpu)

        t_step0 = time.perf_counter()
        batch_cpu = load_and_collate_batch(adapter, batch_paths, max_seq_len, dtype)
        t_disk_read = (time.perf_counter() - t_step0) * 1000.0

        t0 = time.perf_counter()
        batch_device = move_batch_to_device(batch_cpu, device, compute_dtype)
        torch.cuda.synchronize(device)
        t_to_gpu = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        engine.zero_grad()
        seed = 42 + step * engine.world_size + engine.rank
        loss = adapter.forward_loss(engine, batch_device, device, compute_dtype, seed=seed)
        engine.backward(loss)
        engine.step()
        torch.cuda.synchronize(device)
        t_fwd_bwd = (time.perf_counter() - t0) * 1000.0

        t_total = (time.perf_counter() - t_step0) * 1000.0
        timings_ms.append((t_disk_read, t_to_gpu, t_fwd_bwd, t_total))
        vram_alloc.append(torch.cuda.memory_allocated(device) / 1e9)
        vram_reserved.append(torch.cuda.memory_reserved(device) / 1e9)

        if engine.is_main:
            logger.info(
                "[step %d] loss(rank0)=%.6f | disk_read=%.0fms to_gpu=%.0fms fwd_bwd=%.0fms TOTAL=%.0fms | "
                "VRAM rank0=%.2f/%.2fGB",
                step, loss.item(), t_disk_read, t_to_gpu, t_fwd_bwd, t_total, vram_alloc[-1], vram_reserved[-1],
            )

    if not engine.is_main:
        return {}

    n = len(timings_ms)
    totals = [sum(t[i] for t in timings_ms) for i in range(4)]
    tails = [[t[i] for t in timings_ms] for i in range(4)]

    report_base = run_config.report_path or os.path.join(
        os.getcwd(), "reports", f"data_parallel_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    return write_report(
        report_base,
        title=f"diffcache -- data_parallel ({run_config.pipeline}, backend={run_config.parallel.backend}, "
              f"world_size={engine.world_size})",
        config={
            "pipeline": run_config.pipeline, "steps": n,
            "backend": run_config.parallel.backend, "zero_stage": run_config.parallel.zero_stage,
            "batch_size_per_gpu": run_config.parallel.batch_size_per_gpu,
            "batch_size_global": run_config.parallel.batch_size_per_gpu * engine.world_size,
            "world_size": engine.world_size, "quantize": run_config.precision.quant_type,
            "gradient_checkpointing": use_gc, "device_name": torch.cuda.get_device_name(device),
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
            "avg_allocated_rank0": sum(vram_alloc) / n, "avg_reserved_rank0": sum(vram_reserved) / n,
            "peak_allocated_rank0": torch.cuda.max_memory_allocated(device) / 1e9,
            "peak_reserved_rank0": torch.cuda.max_memory_reserved(device) / 1e9,
        },
        per_step=[
            {"step": i, "disk_read_ms": tails[0][i], "to_gpu_ms": tails[1][i], "fwd_bwd_ms": tails[2][i],
             "total_ms": tails[3][i], "allocated_gb_rank0": vram_alloc[i], "reserved_gb_rank0": vram_reserved[i]}
            for i in range(n)
        ],
        extra_lines=[
            "NOTE: VRAM reported is rank 0 only. Each rank holds its own replica of the "
            f"trainable DiT+LoRA; with backend={run_config.parallel.backend!r}, only the "
            "trainable component's optimizer state/gradients are sharded (ZeRO-2) or "
            "all-reduced (DDP) -- the frozen-quantized weights are replicated per rank either way.",
        ],
    )
