# cachezero

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

**All 4 pipelines are implemented**: Qwen-Image-Edit-Angles, FLUX.1-Kontext,
OmniGen2 and DeepGen 1.0. All four have `precompute`/`finetune_cached`
verified on real hardware against their original `*_bench.py` scripts
(timing/VRAM parity within normal run-to-run noise); `live` is verified for
Qwen/FLUX/OmniGen2 and `data_parallel` (ZeRO-2 and DDP) for Qwen.

Verified so far on **PopOS-IA** (2x RTX 5070Ti) and **BULL** (2x Tesla
V100, Qwen only). FLUX.1-Kontext/OmniGen2/DeepGen have not yet been run on
BULL/Volta -- that is a planned follow-up phase, since none of their
numeric behavior on Volta hardware has ever been tested (unlike Qwen,
which needed real fp16/cuDNN workarounds there, see `hardware.py`).

## Install

```bash
pip install -e .                 # core framework + Qwen adapter
pip install -e ".[zero2]"        # + DeepSpeed, for --parallel zero2 (the default)
pip install -e ".[dev]"          # + pytest
```

Each adapter needs one or two env vars pointing at model weights or a repo
checkout (the `omnigen2`/`deepgen_pipeline` packages aren't on PyPI, so
their repos must be cloned separately):

| Pipeline | Env var(s) |
|---|---|
| `qwen_image_edit` | `QWEN_BASE_MODEL_PATH`, `QWEN_FUSED_TRANSFORMER_PATH` |
| `flux_kontext` | none (pulls `black-forest-labs/FLUX.1-Kontext-dev` from HF, gated -- needs an authorized HF token) |
| `omnigen2` | `OMNIGEN2_REPO_PATH` (checkout of the `OmniGen2` repo) |
| `deepgen` | `DEEPGEN_REPO_PATH` (checkout of the `deepgenteam/deepgen` repo) |

**Known environment gotcha for `--parallel ddp`**: some `pip` environments
have both `nvidia-nccl-cu12` and `nvidia-nccl-cu13` installed, which install
to the identical path (`nvidia/nccl/lib/libnccl.so.2`) and silently
overwrite each other -- surfaces as `NCCL error ... Cuda failure 'named
symbol not found'` the first time anything opens a real
`torch.distributed` process group (confirmed on BULL's `~/envs/qwen`,
2026-10-11). Fix: `pip uninstall nvidia-nccl-cu13` then
`pip install --force-reinstall --no-deps nvidia-nccl-cu12==<version torch
was built against>`. `--parallel zero2` can still hit the same bug under
DeepSpeed, for the same reason.

## Usage

Every run is one `phase` of one `pipeline`, configured via flags (see
`cachezero run --help`):

```bash
# Phase 1: precompute + cache the frozen VLM/VAE to disk (1 GPU)
cachezero run --pipeline qwen_image_edit --phase precompute \
    --dataset-root /path/to/MultiViewVisibleThermalImagesHPE --num-samples 50 \
    --precomputed-dir ./precomputed_bench --dtype bf16

# Phase 2: single-GPU reference, training on the cache above
cachezero run --pipeline qwen_image_edit --phase finetune_cached \
    --precomputed-dir ./precomputed_bench --precision nf4 --steps 100

# Baseline: live, no caching (2 GPUs for Qwen -- decided automatically by the adapter)
cachezero run --pipeline qwen_image_edit --phase live \
    --dataset-root /path/to/MultiViewVisibleThermalImagesHPE --num-samples 20 --steps 10

# Baseline: live with prefetch (always 2 GPUs)
cachezero run --pipeline qwen_image_edit --phase live_prefetch \
    --dataset-root /path/to/MultiViewVisibleThermalImagesHPE --num-samples 20 --steps 10

# The proposed methodology: data-parallel training on the cache, ZeRO-2 (default)
# Note: deepspeed's launcher needs --module (not python's own -m) to run a module.
deepspeed --num_gpus=2 --module cachezero.cli run --pipeline qwen_image_edit \
    --phase data_parallel --parallel zero2 --precomputed-dir ./precomputed_bench --steps 100

# ...or DDP instead of ZeRO-2 (same phase, different data-parallel backend)
torchrun --nproc_per_node=2 -m cachezero.cli run --pipeline qwen_image_edit \
    --phase data_parallel --parallel ddp --precomputed-dir ./precomputed_bench --steps 100

# FLUX.1-Kontext/OmniGen2/DeepGen all use the same LLVIP RGB->thermal task
# and dataset layout -- --dataset-root points at the LLVIP root (containing
# either visible_train/infrared_train or visible/train + infrared/train).
cachezero run --pipeline flux_kontext --phase precompute \
    --dataset-root /path/to/LLVIP --num-samples 50 --precomputed-dir ./flux_precomputed
cachezero run --pipeline omnigen2 --phase finetune_cached \
    --precomputed-dir ./omnigen2_precomputed --precision none --steps 20 --gradient-checkpointing on
cachezero run --pipeline deepgen --phase live \
    --dataset-root /path/to/LLVIP --num-samples 20 --steps 10 --gradient-checkpointing on
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
  (`parallel/ddp.py`) is a new, simpler alternative with no DeepSpeed
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

End-to-end verification (needs a GPU) is described in the project's plan
file, not automated here yet: run `precompute`+`finetune_cached` and diff
the resulting `.json` report against one produced by the original
`precompute_bench.py`/`finetune_precomputed_bench.py` with the same
arguments.
