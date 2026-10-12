# DiffCache

Precompute+cache the frozen encoders of an image-editing diffusion pipeline,
quantize the trainable DiT via QLoRA, and reclaim the GPU that caching freed
for data-parallel training (ZeRO-2 by default, DDP as an alternative) --
packaged as a reusable framework instead of one `*_bench.py` script per
pipeline/precision/GPU-count combination.

This is the methodology behind a Journal of Supercomputing submission on
Qwen-Image-Edit-Angles, FLUX.1-Kontext-dev, OmniGen2 and DeepGen 1.0. It
lets you (a) fine-tune your own model with this methodology, or (b)
reproduce the paper's measured wall-clock/VRAM numbers on your own
hardware.

## Status

**This branch (`master`): Qwen-Image-Edit-Angles only**, fully implemented
and verified against the original `*_bench.py` scripts (timing/VRAM parity
within normal run-to-run noise) on BULL (2x Tesla V100-PCIE-32GB, Volta,
SLURM cluster) -- `precompute`, `finetune_cached`, `live`, `live_prefetch`
and `data_parallel` (both ZeRO-2 and DDP) all pass.

FLUX.1-Kontext-dev, OmniGen2 and DeepGen 1.0 exist here only as adapter
*stubs* (`src/diffcache/pipelines/{flux_kontext,omnigen2,deepgen}.py`) --
calling them raises `NotImplementedError`. **All 3 are fully implemented
and verified (both PopOS-IA/Blackwell and BULL/Volta, all 5 phases) on the
`pipelines/flux-omnigen2-deepgen` branch**, not yet merged here.

## Install

```bash
pip install -e .                 # core framework + Qwen adapter
pip install -e ".[zero2]"        # + DeepSpeed, for --parallel zero2 (the default)
pip install -e ".[dev]"          # + pytest
```

Requires `QWEN_BASE_MODEL_PATH` and `QWEN_FUSED_TRANSFORMER_PATH` set (same
convention as `training_from_jupyter/qwen_bench_utils.py` in the original
`Qwen_JoS` repo this was extracted from) to use the Qwen adapter.

**Known environment gotchas** (hit and fixed during verification, see the
`pipelines/flux-omnigen2-deepgen` branch's README for the full writeup):
a duplicate `nvidia-nccl-cu12`/`nvidia-nccl-cu13` install can silently
break `--parallel ddp`/`zero2`, and DeepSpeed's `FusedAdam` optimizer can
fail to JIT-compile on a GPU generation the installed CUDA toolkit doesn't
target yet (worked around internally in `parallel/zero.py` via
`torch_adam=True`, nothing to configure).

## Usage

Every run is one `phase` of one `pipeline`, configured via flags (see
`diffcache run --help`):

```bash
# Phase 1: precompute + cache the frozen VLM/VAE to disk (1 GPU)
diffcache run --pipeline qwen_image_edit --phase precompute \
    --dataset-root /path/to/MultiViewVisibleThermalImagesHPE --num-samples 50 \
    --precomputed-dir ./precomputed_bench --dtype bf16

# Phase 2: single-GPU reference, training on the cache above
diffcache run --pipeline qwen_image_edit --phase finetune_cached \
    --precomputed-dir ./precomputed_bench --precision nf4 --steps 100

# Baseline: live, no caching (2 GPUs for Qwen -- decided automatically by the adapter)
diffcache run --pipeline qwen_image_edit --phase live \
    --dataset-root /path/to/MultiViewVisibleThermalImagesHPE --num-samples 20 --steps 10

# Baseline: live with prefetch (always 2 GPUs)
diffcache run --pipeline qwen_image_edit --phase live_prefetch \
    --dataset-root /path/to/MultiViewVisibleThermalImagesHPE --num-samples 20 --steps 10

# The proposed methodology: data-parallel training on the cache, ZeRO-2 (default)
# Note: deepspeed's launcher needs --module (not python's own -m) to run a module.
deepspeed --num_gpus=2 --module diffcache.cli run --pipeline qwen_image_edit \
    --phase data_parallel --parallel zero2 --precomputed-dir ./precomputed_bench --steps 100

# ...or DDP instead of ZeRO-2 (same phase, different data-parallel backend)
torchrun --nproc_per_node=2 -m diffcache.cli run --pipeline qwen_image_edit \
    --phase data_parallel --parallel ddp --precomputed-dir ./precomputed_bench --steps 100
```

On Volta GPUs (e.g. BULL's V100s) `--disable_cudnn`/`fp16`/`fp32 NF4 compute`
are applied automatically (`hardware.apply_quirks`) -- no flags to remember,
unlike the original benchmark scripts. Pass `--dtype`/`--compute-dtype`
explicitly if you want to override that for a deliberate experiment (an
explicit flag is never silently clobbered).

Every run writes `<report>.json` + `<report>.txt` (same shape as the
original `*_bench.py` scripts' reports) to `--report-path`, or
`./reports/<phase>_<timestamp>` by default.

## Design

- **One adapter class per pipeline** (`pipelines/*.py`), not a universal
  model wrapper -- each architecture encodes its prompt/images and computes
  its flow-matching loss too differently for that to work. Everything that
  *is* common (caching, the live/precompute/data-parallel control flow,
  timing, VRAM accounting, hardware quirks) lives in `phases/*.py` and
  `hardware.py`, written once against the small `PipelineAdapter` interface.
- **ZeRO-2 is the default data-parallel backend** (`parallel/zero.py`,
  matching every experiment measured for the paper); **DDP**
  (`parallel/ddp.py`) is a simpler alternative with no DeepSpeed
  dependency. Both share `ParallelBackend.shard_batch` so a comparison
  between them is always apples-to-apples (same samples-per-step on each
  side).
- **Standalone repo, no dependency on the paper's own experiment scripts**
  (`Qwen_JoS/training_from_jupyter/`, `Flux.1/`, `OmniGen2/`,
  `DeepGen/precompute_live_zero2_bench/`, etc.) -- each adapter copies and
  adapts the already-measured logic from those sibling projects rather than
  importing it, so neither repo depends on the other and the paper's
  existing results stay reproducible exactly as measured.

## Tests

```bash
pytest tests/   # no GPU/DeepSpeed required -- these are unit tests of the
                 # config/hardware-quirk/sharding logic, not end-to-end runs
```

End-to-end verification (needs a GPU) is: run `precompute`+`finetune_cached`
and diff the resulting `.json` report against one produced by the original
`precompute_bench.py`/`finetune_precomputed_bench.py` with the same
arguments.
