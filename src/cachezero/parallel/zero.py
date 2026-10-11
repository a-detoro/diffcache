"""DeepSpeed ZeRO-1/2 backend -- the default data-parallel strategy, ported
from `training_from_jupyter/finetune_zero_bench.py`'s `ds_config` construction
(same optimizer block, same "no fp16/bf16 block" choice so DeepSpeed's own
mixed-precision machinery never competes with the hardware-quirk dtype
already applied by `hardware.apply_quirks`).

ZeRO-3 is intentionally not supported here: it was tried for this
methodology and abandoned (see feedback_bull_volta_quirks / the ZeRO-3
section of project_qwen_bull_benchmarks) -- its low-level parameter
partitioning doesn't compose with bitsandbytes' custom `Params4bit` tensors
used by NF4 quantization, which is the trainable component's normal state in
this framework. `config.ParallelConfig.zero_stage` only accepts 1 or 2.

Launch with `deepspeed --num_gpus=N --module cachezero.cli ...` -- the
deepspeed launcher needs its own `--module` flag (not Python's bare `-m`)
to run a module; see the package README.
"""
from __future__ import annotations

import os
from typing import Any, List

import torch

from .base import ParallelBackend, ParallelEngine


class ZeroEngine(ParallelEngine):
    def __init__(self, model_engine: Any, rank: int, world_size: int):
        self.model_engine = model_engine
        self.rank = rank
        self.world_size = world_size

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.model_engine(*args, **kwargs)

    def zero_grad(self) -> None:
        pass  # DeepSpeed's model_engine.step() already zeroes grads after applying them.

    def backward(self, loss: Any) -> None:
        self.model_engine.backward(loss)

    def step(self) -> None:
        self.model_engine.step()


class ZeroBackend(ParallelBackend):
    def setup(self, model: Any, trainable_params: List[Any], run_config: Any) -> ZeroEngine:
        import deepspeed

        deepspeed.init_distributed()

        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(f"cuda:{local_rank}")

        zero_stage = run_config.parallel.zero_stage
        if zero_stage not in (1, 2):
            raise ValueError(
                f"zero_stage={zero_stage} not supported -- ZeRO-3 doesn't compose with "
                "bitsandbytes' NF4/int8 quantized tensors for this methodology (see module "
                "docstring). Use zero_stage 1 or 2."
            )

        ds_config = {
            "train_micro_batch_size_per_gpu": run_config.parallel.batch_size_per_gpu,
            "gradient_accumulation_steps": 1,
            "steps_per_print": 10_000_000,  # caller does its own logging
            "zero_optimization": {
                "stage": zero_stage,
                # DeepSpeed's own defaults (500_000_000 elements each,
                # ~1.9GiB fp32) size these allreduce/allgather buckets for
                # models with hundreds of millions of trainable parameters.
                # Every pipeline in this framework trains a LoRA adapter
                # (5-47M params total across every tensor) -- confirmed
                # OOMing on a 16GB card from this buffer alone, with the
                # actual model nowhere near the memory limit otherwise.
                # 20M elements (~80MB fp32) comfortably covers the largest
                # LoRA adapter here with room to grow.
                "reduce_bucket_size": 20_000_000,
                "allgather_bucket_size": 20_000_000,
            },
            "gradient_clipping": 1.0,
            "optimizer": {
                "type": "AdamW",
                # torch_adam=True forces plain torch.optim.AdamW instead of
                # DeepSpeed's fused CUDA kernel, which JIT-compiles via
                # ninja+nvcc on first use -- fails on GPU generations the
                # installed CUDA toolkit doesn't target yet (confirmed:
                # "nvcc fatal: Unsupported gpu architecture 'compute_120'"
                # on Blackwell with a 12.6 toolkit). Numerically identical,
                # just not fused; with only a few million LoRA params this
                # workload never exercised FusedAdam's actual speed
                # advantage anyway. Unconditional, not hardware-detected:
                # it's a portability default, not a correctness fix for one
                # specific GPU generation.
                "params": {"lr": run_config.learning_rate, "eps": 1e-6, "torch_adam": True},
            },
        }

        model_engine, _optimizer, _, _ = deepspeed.initialize(
            model=model, model_parameters=trainable_params, config=ds_config,
        )
        return ZeroEngine(model_engine, rank=rank, world_size=world_size)
