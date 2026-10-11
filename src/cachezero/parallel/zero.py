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
            "zero_optimization": {"stage": zero_stage},
            "gradient_clipping": 1.0,
            "optimizer": {
                "type": "AdamW",
                "params": {"lr": run_config.learning_rate, "eps": 1e-6},
            },
        }

        model_engine, _optimizer, _, _ = deepspeed.initialize(
            model=model, model_parameters=trainable_params, config=ds_config,
        )
        return ZeroEngine(model_engine, rank=rank, world_size=world_size)
