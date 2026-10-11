"""Dispatches a `RunConfig` to the matching `phases/*.py` module. The one
place that knows "phase name" -> "which function runs it" -- everything else
(CLI, tests, a notebook) goes through this instead of importing `phases.*`
directly.
"""
from __future__ import annotations

from .config import RunConfig


def run(run_config: RunConfig) -> dict:
    if run_config.phase == "precompute":
        from .phases import precompute
        return precompute.run(run_config)
    if run_config.phase == "finetune_cached":
        from .phases import finetune_cached
        return finetune_cached.run(run_config)
    if run_config.phase == "live":
        from .phases import live
        return live.run(run_config)
    if run_config.phase == "live_prefetch":
        from .phases import live_prefetch
        return live_prefetch.run(run_config)
    if run_config.phase == "data_parallel":
        from .phases import data_parallel
        return data_parallel.run(run_config)
    raise ValueError(f"Unknown phase: {run_config.phase!r}")
