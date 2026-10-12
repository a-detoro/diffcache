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

**All 4 pipelines, all 5 phases** (`precompute`, `finetune_cached`, `live`,
`live_prefetch`, `data_parallel` with both ZeRO-2 and DDP) are implemented
and verified against the original `*_bench.py` scripts (timing/VRAM parity
within normal run-to-run noise), on two different machines/GPU generations:

- **PopOS-IA** (2x RTX 5070Ti, Blackwell, sm_120)
- **BULL** (2x Tesla V100-PCIE-32GB, Volta, sm_70, SLURM cluster)

No pipeline needed its own hardware-specific code beyond what already
existed for Qwen -- the Volta quirk table (`hardware.py`) generalized to
FLUX.1-Kontext/OmniGen2/DeepGen without changes.

## Install

**One environment per machine, not per pipeline.** All 4 pipelines share
the same core dependency versions (`transformers`, `peft`, `bitsandbytes`,
`accelerate` stay identical across every pipeline and every GPU generation
tested so far) -- the only thing that actually forces a different
environment is the **GPU generation** (which `torch`/CUDA build it needs),
not which pipeline you want to run. Pipeline-specific extras (`einops`,
`torchdiffeq`, `deepspeed`) are small, optional, and stack on top of one
shared base install via pip extras.

### Step 1 -- pick the `torch` build for your hardware

| GPU generation | `torch` | Notes |
|---|---|---|
| Volta / Ampere / Ada (most data-center and consumer cards, incl. V100, A30, A100, RTX 30/40-series) | stable release from `https://download.pytorch.org/whl/cu121` (verified: `2.5.1+cu121`) | Works out of the box |
| Blackwell (RTX 50-series, sm_120) | a nightly/dev build from the `cu128` nightly index (verified: `2.12.0.dev+cu128`) at the time this was tested -- check PyTorch's current release notes, stable support for sm_120 may have landed since | Also needed a `diffusers` install built close to HEAD rather than the latest PyPI tag, for the same reason; re-check before assuming you need this today |

```bash
# Volta/Ampere/Ada:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# Blackwell (if stable torch still doesn't support sm_120 when you read this):
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

### Step 2 -- install DiffCache with the extras for the pipelines you want

```bash
pip install -e .                 # core framework + Qwen adapter (no extras needed)
pip install -e ".[omnigen2]"     # + einops, torchdiffeq
pip install -e ".[deepgen]"      # + einops, safetensors
pip install -e ".[zero2]"        # + DeepSpeed, for --parallel zero2 (the default data-parallel backend)
pip install -e ".[all]"          # every extra above
pip install -e ".[dev]"          # + pytest
```

`deepspeed`'s own version is hardware-sensitive too, same as `torch`: pin
`deepspeed==0.14.5` on a stable `torch==2.5.1` stack (newer DeepSpeed
releases fail to import there -- `torch.library.infer_schema` doesn't
recognize a type annotation one of their newer modules uses); a
dev/nightly `torch` build tolerated a newer DeepSpeed fine (`0.19.6`,
verified). If `pip install -e ".[zero2]"` pulls a DeepSpeed that fails to
import, pin it down to `0.14.5` explicitly.

### Step 3 -- per-pipeline model weights / repo checkouts

Each adapter needs one or two env vars pointing at model weights or a repo
checkout (the `omnigen2`/`deepgen_pipeline` packages aren't on PyPI, so
their repos must be cloned separately):

| Pipeline | Env var(s) |
|---|---|
| `qwen_image_edit` | `QWEN_BASE_MODEL_PATH`, `QWEN_FUSED_TRANSFORMER_PATH` |
| `flux_kontext` | none (pulls `black-forest-labs/FLUX.1-Kontext-dev` from HF, gated -- needs an authorized HF token) |
| `omnigen2` | `OMNIGEN2_REPO_PATH` (checkout of the `OmniGen2` repo) |
| `deepgen` | `DEEPGEN_REPO_PATH` (checkout of the `deepgenteam/deepgen` repo, only `deepgen_pipeline.py` is actually needed) |

### Known environment gotchas

- **`--parallel ddp`/`--parallel zero2` NCCL error** (`Cuda failure 'named
  symbol not found'`): some environments end up with both
  `nvidia-nccl-cu12` and `nvidia-nccl-cu13` installed, which both install
  to the identical path (`nvidia/lib/libnccl.so.2`) and silently overwrite
  each other -- surfaces the first time anything opens a real
  `torch.distributed` process group. Fix: `pip uninstall nvidia-nccl-cu13`
  then `pip install --force-reinstall --no-deps nvidia-nccl-cu12==<version
  torch was built against>`. Building the venv fresh (rather than
  installing from an old frozen requirements file that lists both) avoids
  this from the start.
- **`--parallel zero2` `FusedAdam` build failure** (`nvcc fatal:
  Unsupported gpu architecture 'compute_XXX'`): DeepSpeed's optimizer
  JIT-compiles a CUDA kernel on first use, which fails on a GPU generation
  the installed CUDA toolkit doesn't target yet. Already worked around
  inside `parallel/zero.py` (`torch_adam=True`, falls back to plain
  `torch.optim.AdamW`) -- nothing to do, just documented here in case you
  see the error from somewhere else.

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

# FLUX.1-Kontext/OmniGen2/DeepGen all use the same LLVIP RGB->thermal task
# and dataset layout -- --dataset-root points at the LLVIP root (containing
# either visible_train/infrared_train or visible/train + infrared/train).
diffcache run --pipeline flux_kontext --phase precompute \
    --dataset-root /path/to/LLVIP --num-samples 50 --precomputed-dir ./flux_precomputed
diffcache run --pipeline omnigen2 --phase finetune_cached \
    --precomputed-dir ./omnigen2_precomputed --precision none --steps 20 --gradient-checkpointing on
diffcache run --pipeline deepgen --phase live \
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
  (`parallel/ddp.py`) is a simpler alternative with no DeepSpeed
  dependency. Both share `ParallelBackend.shard_batch` so a comparison
  between them is always apples-to-apples (same samples-per-step on each
  side), and both have their gradient-bucket/optimizer defaults tuned for
  LoRA-sized trainable sets (a few million parameters), not DeepSpeed's own
  defaults sized for hundreds of millions.
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
arguments. Every pipeline/phase/backend combination listed in Status above
has been verified this way at least once.
