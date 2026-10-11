"""Phase 1: precompute+cache. Pipeline-agnostic port of
`training_from_jupyter/precompute_bench.py` -- runs every frozen component
once over N samples via the adapter's `encode_sample`, saves one `.pt` per
sample. Needs only 1 GPU: in the precompute approach the trainable DiT
doesn't need to be loaded during this phase at all.
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

logger = logging.getLogger(__name__)


def run(run_config: RunConfig, device: str = "cuda:0") -> dict:
    adapter = get_adapter(run_config.pipeline)
    apply_quirks(run_config.precision, device)
    dtype = TORCH_DTYPES[run_config.precision.storage_dtype]

    if not run_config.dataset_root:
        raise ValueError("RunConfig.dataset_root is required for phase=precompute")
    output_dir = run_config.precomputed_dir or os.path.join(os.getcwd(), "precomputed_bench")
    os.makedirs(output_dir, exist_ok=True)

    dataset = adapter.build_dataset(run_config.dataset_root, run_config.num_samples)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_pils)

    frozen = adapter.load_frozen(device, dtype)
    torch.cuda.reset_peak_memory_stats(device)

    logger.info("Precomputing %d samples -> %s", len(dataset), output_dir)

    timings_ms, vram_alloc, vram_reserved = [], [], []
    t_run0 = time.perf_counter()

    for idx, batch in enumerate(dataloader):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        cached = adapter.encode_sample(frozen, batch["prompts"], batch["source_pils"], batch["target_pils"], device, dtype)
        torch.cuda.synchronize(device)
        t_encode = (time.perf_counter() - t0) * 1000.0

        save_path = os.path.join(output_dir, f"sample_{idx:04d}.pt")
        torch.save({k: v.cpu() for k, v in cached.items()}, save_path)

        timings_ms.append(t_encode)
        vram_alloc.append(torch.cuda.memory_allocated(device) / 1e9)
        vram_reserved.append(torch.cuda.memory_reserved(device) / 1e9)
        logger.info("[%d/%d] encode=%.0fms | VRAM %.2f/%.2fGB", idx + 1, len(dataset), t_encode,
                    vram_alloc[-1], vram_reserved[-1])

    t_total = time.perf_counter() - t_run0
    n = len(timings_ms)

    report_base = run_config.report_path or os.path.join(
        os.getcwd(), "reports", f"precompute_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    return write_report(
        report_base,
        title=f"cachezero -- precompute ({run_config.pipeline}, dtype={run_config.precision.storage_dtype})",
        config={
            "pipeline": run_config.pipeline, "samples": n,
            "device": device, "device_name": torch.cuda.get_device_name(device),
            "dtype": run_config.precision.storage_dtype, "output_dir": output_dir,
        },
        timing_ms={
            "encode_avg_ms": sum(timings_ms) / n,
            "encode_avg_ms_excl_first": mean_excluding_first(timings_ms),
            "precompute_total_s": t_total,
        },
        vram_gb={
            "avg_allocated": sum(vram_alloc) / n, "avg_reserved": sum(vram_reserved) / n,
            "peak_allocated": torch.cuda.max_memory_allocated(device) / 1e9,
            "peak_reserved": torch.cuda.max_memory_reserved(device) / 1e9,
        },
        per_step=[
            {"idx": i, "encode_ms": timings_ms[i], "allocated_gb": vram_alloc[i], "reserved_gb": vram_reserved[i]}
            for i in range(n)
        ],
    )
