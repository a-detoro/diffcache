"""Timing/VRAM measurement helpers, shared by every phase so numbers are
comparable across pipelines and backends (same convention the original
benchmark scripts used by hand in every `*_bench.py`)."""
from __future__ import annotations

import time
from contextlib import contextmanager

import torch


@contextmanager
def cuda_timer(device: str | torch.device):
    """Measures wall-clock time for a block, synchronizing `device` on both
    ends so the measurement includes queued GPU work, not just host dispatch
    time. Access the elapsed seconds via the yielded object's `.seconds`
    after the `with` block exits."""

    class _Result:
        seconds: float = 0.0

    result = _Result()
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    try:
        yield result
    finally:
        torch.cuda.synchronize(device)
        result.seconds = time.perf_counter() - t0


def vram_snapshot(device: str | torch.device) -> dict[str, float]:
    """Instantaneous (not peak) allocated/reserved VRAM in GB."""
    return {
        "allocated_gb": torch.cuda.memory_allocated(device) / 1e9,
        "reserved_gb": torch.cuda.memory_reserved(device) / 1e9,
    }


def vram_peak(device: str | torch.device) -> dict[str, float]:
    """Peak allocated/reserved VRAM since the last `torch.cuda.reset_peak_memory_stats`."""
    return {
        "peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9,
        "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / 1e9,
    }


def mean_excluding_first(values: list[float]) -> float:
    """Step 0 is consistently ~0.5% slower (first CUDA kernel launches / cuDNN
    algo selection, see project_qwen_bull_benchmarks memory) -- excluded from
    the mean by convention throughout this project's benchmarks."""
    tail = values[1:] if len(values) > 1 else values
    return sum(tail) / len(tail)
