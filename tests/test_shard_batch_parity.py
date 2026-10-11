"""`ParallelBackend.shard_batch` is the one piece of logic both the ZeRO-2 and
DDP backends share so they process identical work-per-step -- see the
module docstring in parallel/base.py and the OmniGen2 "samples-per-step"
lesson in project_omnigen2_popos_bench. These tests don't need a GPU or
deepspeed installed: both backend classes only import their heavy
dependencies lazily, inside `setup()`.
"""
from cachezero.parallel.ddp import DDPBackend
from cachezero.parallel.zero import ZeroBackend

PATHS = [f"sample_{i:04d}.pt" for i in range(12)]


def test_shard_batch_is_identical_between_zero_and_ddp_backends():
    for step in range(5):
        for rank in range(2):
            zero_shard = ZeroBackend.shard_batch(PATHS, step, rank, world_size=2, batch_size_per_gpu=1)
            ddp_shard = DDPBackend.shard_batch(PATHS, step, rank, world_size=2, batch_size_per_gpu=1)
            assert zero_shard == ddp_shard


def test_shard_batch_covers_global_batch_without_overlap_within_a_step():
    step, world_size, batch_size_per_gpu = 3, 2, 1
    shards = [
        ZeroBackend.shard_batch(PATHS, step, rank, world_size, batch_size_per_gpu)
        for rank in range(world_size)
    ]
    flattened = [p for shard in shards for p in shard]
    assert len(flattened) == len(set(flattened)) == world_size * batch_size_per_gpu


def test_shard_batch_advances_deterministically_across_steps():
    rank, world_size, batch_size_per_gpu = 0, 2, 1
    seen = {
        step: ZeroBackend.shard_batch(PATHS, step, rank, world_size, batch_size_per_gpu)[0]
        for step in range(len(PATHS))
    }
    # Same rank, consecutive steps should not repeat a sample until the pool wraps.
    assert len(set(seen.values())) == len(PATHS) // world_size
