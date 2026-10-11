"""Data-parallel backend abstraction for the `data_parallel` phase.

Both concrete backends (`zero.ZeroBackend`, `ddp.DDPBackend`) must process the
exact same volume of work per step -- `batch_size_per_gpu` samples on each of
`world_size` ranks, a DIFFERENT sub-batch per rank -- so a timing comparison
between them is fair. `shard_batch` is the one piece of logic that decides
"which samples does rank R train on at step S", shared so neither backend can
accidentally drift from the other (this is exactly the mistake documented in
project_omnigen2_popos_bench's memory: comparing ms/step without accounting
for samples-processed-per-step gives a wrong speedup number).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List


class ParallelEngine(ABC):
    """Uniform wrapper around either a DeepSpeed `model_engine` or a plain
    `(DistributedDataParallel, optimizer)` pair, so `phases/data_parallel.py`
    doesn't need an `if backend == ...` branch in its training loop."""

    rank: int
    world_size: int

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass through the (possibly wrapped) trainable model."""

    @abstractmethod
    def zero_grad(self) -> None:
        ...

    @abstractmethod
    def backward(self, loss: Any) -> None:
        ...

    @abstractmethod
    def step(self) -> None:
        ...


class ParallelBackend(ABC):
    @abstractmethod
    def setup(self, model: Any, trainable_params: List[Any], run_config: Any) -> ParallelEngine:
        """Wraps `model` for data parallelism across `run_config.parallel.num_gpus`
        ranks and returns a ready-to-use `ParallelEngine`. Must be called
        under the launcher appropriate for the backend (`deepspeed
        --num_gpus=N` for ZeRO, `torchrun --nproc_per_node=N` for DDP) -- see
        the package README."""

    @staticmethod
    def shard_batch(paths: List[str], step: int, rank: int, world_size: int,
                     batch_size_per_gpu: int) -> List[str]:
        """Which `batch_size_per_gpu` samples rank `rank` trains on at `step`,
        out of the cyclic pool `paths`. Ported unchanged from
        `finetune_zero_bench.py`'s inline sharding formula."""
        n = len(paths)
        base = step * batch_size_per_gpu * world_size + rank * batch_size_per_gpu
        return [paths[(base + i) % n] for i in range(batch_size_per_gpu)]
