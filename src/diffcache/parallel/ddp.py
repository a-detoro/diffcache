"""DistributedDataParallel backend -- NEW, not present in any of the
original benchmark scripts (only ZeRO-1/2 were ever measured for the paper;
see project_qwen_bull_benchmarks). Offered as an alternative to ZeRO-2 for
users who don't want the DeepSpeed dependency, or want a reference point:
with only a few million trainable LoRA parameters, ZeRO-2's distinguishing
feature (optimizer-state/gradient sharding) barely matters (see the
batch-size-confound correction in project_qwen_bull_benchmarks) -- plain DDP
is expected to land close to ZeRO-2 in practice for this workload, which is
worth letting users verify themselves rather than asserting.

Launch with `torchrun --nproc_per_node=N -m diffcache.cli ...`, which sets
RANK/WORLD_SIZE/LOCAL_RANK the same way `deepspeed --num_gpus=N` does for
`ZeroBackend` -- see the package README.
"""
from __future__ import annotations

import os
from typing import Any, List

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .base import ParallelBackend, ParallelEngine


class DDPEngine(ParallelEngine):
    def __init__(self, ddp_model: DistributedDataParallel, optimizer: torch.optim.Optimizer,
                 trainable_params: List[Any], rank: int, world_size: int):
        self.ddp_model = ddp_model
        self.optimizer = optimizer
        self.trainable_params = trainable_params
        self.rank = rank
        self.world_size = world_size

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.ddp_model(*args, **kwargs)

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def backward(self, loss: Any) -> None:
        loss.backward()

    def step(self) -> None:
        # Same gradient-clipping convention as every other training script in
        # this project (finetune_precomputed_bench.py, finetune_zero_bench.py).
        torch.nn.utils.clip_grad_norm_(self.trainable_params, max_norm=1.0)
        self.optimizer.step()


class DDPBackend(ParallelBackend):
    def setup(self, model: Any, trainable_params: List[Any], run_config: Any) -> DDPEngine:
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

        model = model.to(device)
        # find_unused_parameters=True: the trainable component here is a
        # custom per-block checkpointed forward (QwenSingleGPUWrapper and
        # equivalents), not a vanilla nn.Module.forward -- DDP can't always
        # statically prove every requires_grad=True LoRA parameter
        # participates in every single step's graph, and guessing wrong
        # raises a hard error mid-training rather than just costing a little
        # extra bookkeeping.
        ddp_model = DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=True,
        )
        optimizer = torch.optim.AdamW(trainable_params, lr=run_config.learning_rate, eps=1e-6)

        return DDPEngine(ddp_model, optimizer, trainable_params, rank=rank, world_size=world_size)
