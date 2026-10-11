"""Run configuration: what to run (pipeline), how (precision/parallel backend), and on what (phase).

These dataclasses are the single source of truth the CLI builds and every
`phases/*.py` function consumes -- keeping phase code pipeline-agnostic and
precision-agnostic.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Literal, Optional

QuantType = Literal["none", "nf4", "int8"]
StorageDtype = Literal["bf16", "fp16", "fp32"]
ParallelBackendName = Literal["zero2", "ddp", "none"]
Phase = Literal["precompute", "finetune_cached", "live", "live_prefetch", "data_parallel"]

DTYPE_CHOICES: tuple[StorageDtype, ...] = ("bf16", "fp16", "fp32")


@dataclass
class PrecisionConfig:
    """Numeric precision for the trainable DiT and for the frozen encoders.

    `quant_type` only applies to the trainable component (the DiT/MMDiT) --
    frozen encoders are never quantized in this methodology, only cast to
    `storage_dtype`. `compute_dtype` is the dtype NF4/int8 matmuls are
    actually computed in (`bnb_4bit_compute_dtype` for NF4); when left as
    None it defaults to `storage_dtype`, except a hardware quirk (see
    `hardware.py`) may override it, e.g. Volta forces fp32 compute for NF4.
    """

    quant_type: QuantType = "nf4"
    storage_dtype: StorageDtype = "bf16"
    compute_dtype: Optional[StorageDtype] = None

    def resolved_compute_dtype(self) -> StorageDtype:
        return self.compute_dtype or self.storage_dtype


@dataclass
class ParallelConfig:
    """Data-parallel strategy for the `data_parallel` phase.

    Default is ZeRO-2, matching every experiment measured for the paper so
    far (see feedback_bull_volta_quirks / project_qwen_bull_benchmarks
    memories) -- ZeRO-3 was tried and abandoned (doesn't compose with
    bitsandbytes' Params4bit), so it is intentionally not an option here.
    """

    backend: ParallelBackendName = "zero2"
    zero_stage: int = 2  # only meaningful when backend == "zero2"; 1 or 2 (not 3, see above)
    num_gpus: int = 2
    batch_size_per_gpu: int = 1


@dataclass
class RunConfig:
    pipeline: str
    phase: Phase
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)

    steps: int = 100
    batch_size: int = 2  # used by phases that are not data_parallel (which uses parallel.batch_size_per_gpu)
    learning_rate: float = 1e-4
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    gradient_checkpointing: Literal["auto", "on", "off"] = "auto"

    num_samples: int = 20
    dataset_root: Optional[str] = None
    precomputed_dir: Optional[str] = None

    disable_cudnn: bool = False  # normally set automatically by hardware.apply_quirks(), can be forced
    report_path: Optional[str] = None


def add_run_config_args(parser: argparse.ArgumentParser) -> None:
    """Registers every RunConfig field as a CLI flag, shared by `cli.py`."""
    parser.add_argument("--pipeline", required=True,
                         choices=["qwen_image_edit", "flux_kontext", "omnigen2", "deepgen"])
    parser.add_argument("--phase", required=True,
                         choices=["precompute", "finetune_cached", "live", "live_prefetch", "data_parallel"])
    parser.add_argument("--precision", default="nf4", choices=["none", "nf4", "int8"],
                         help="Quantization of the trainable DiT (storage/compute dtype are separate flags).")
    parser.add_argument("--dtype", default="bf16", choices=list(DTYPE_CHOICES))
    parser.add_argument("--compute-dtype", default=None, choices=list(DTYPE_CHOICES))
    parser.add_argument("--parallel", default="zero2", choices=["zero2", "ddp", "none"])
    parser.add_argument("--zero-stage", type=int, default=2, choices=[1, 2])
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--batch-size-per-gpu", type=int, default=1)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--gradient-checkpointing", default="auto", choices=["auto", "on", "off"])
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--precomputed-dir", default=None)
    parser.add_argument("--disable-cudnn", action="store_true")
    parser.add_argument("--report-path", default=None)


def run_config_from_args(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        pipeline=args.pipeline,
        phase=args.phase,
        precision=PrecisionConfig(
            quant_type=args.precision,
            storage_dtype=args.dtype,
            compute_dtype=args.compute_dtype,
        ),
        parallel=ParallelConfig(
            backend=args.parallel,
            zero_stage=args.zero_stage,
            num_gpus=args.gpus,
            batch_size_per_gpu=args.batch_size_per_gpu,
        ),
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        gradient_checkpointing=args.gradient_checkpointing,
        num_samples=args.num_samples,
        dataset_root=args.dataset_root,
        precomputed_dir=args.precomputed_dir,
        disable_cudnn=args.disable_cudnn,
        report_path=args.report_path,
    )
