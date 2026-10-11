"""Common JSON+txt report writer -- ports `qwen_bench_utils.write_report`
verbatim (same shape used by every pipeline's `*_bench_utils.py` across
Qwen_JoS/Flux.1/OmniGen2/DeepGen), decoupled from any Qwen-specific field so
every phase in this framework can call it the same way regardless of
pipeline.
"""
from __future__ import annotations

import json
import logging
import os
import platform
import socket
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)


def write_report(
    report_base: str,
    title: str,
    config: dict[str, Any],
    timing_ms: dict[str, Any],
    vram_gb: dict[str, Any],
    per_step: list[dict[str, Any]],
    extra_lines: Optional[list[str]] = None,
) -> dict[str, Any]:
    os.makedirs(os.path.dirname(report_base) or ".", exist_ok=True)

    lines = [
        "=" * 70,
        title,
        f"Date: {datetime.now().isoformat(timespec='seconds')}  |  Host: {socket.gethostname()}",
    ]
    for k, v in config.items():
        lines.append(f"{k}: {v}")
    lines.append("-" * 70)
    lines.append("TIMINGS:")
    for k, v in timing_ms.items():
        lines.append(f"  {k}: {v}")
    lines.append("-" * 70)
    lines.append("VRAM (GB):")
    for k, v in vram_gb.items():
        lines.append(f"  {k}: {v}")
    if extra_lines:
        lines.append("-" * 70)
        lines.extend(extra_lines)
    lines.append("=" * 70)

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "config": config,
        "timing_ms": timing_ms,
        "vram_gb": vram_gb,
        "per_step": per_step,
    }

    with open(report_base + ".json", "w") as f:
        json.dump(report, f, indent=2)
    with open(report_base + ".txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    for line in lines:
        logger.info(line)
    logger.info("Report saved to %s.json and %s.txt", report_base, report_base)
    return report
