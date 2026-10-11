from cachezero.config import (
    ParallelConfig,
    PrecisionConfig,
    RunConfig,
    add_run_config_args,
    run_config_from_args,
)
import argparse


def test_precision_config_resolved_compute_dtype_defaults_to_storage_dtype():
    precision = PrecisionConfig(storage_dtype="bf16")
    assert precision.resolved_compute_dtype() == "bf16"


def test_precision_config_explicit_compute_dtype_wins():
    precision = PrecisionConfig(storage_dtype="bf16", compute_dtype="fp32")
    assert precision.resolved_compute_dtype() == "fp32"


def test_parallel_config_defaults_to_zero2():
    parallel = ParallelConfig()
    assert parallel.backend == "zero2"
    assert parallel.zero_stage == 2


def test_run_config_from_args_round_trip():
    parser = argparse.ArgumentParser()
    add_run_config_args(parser)
    args = parser.parse_args([
        "--pipeline", "qwen_image_edit",
        "--phase", "data_parallel",
        "--parallel", "ddp",
        "--gpus", "2",
        "--precision", "nf4",
        "--dtype", "fp16",
    ])
    run_config = run_config_from_args(args)

    assert isinstance(run_config, RunConfig)
    assert run_config.pipeline == "qwen_image_edit"
    assert run_config.phase == "data_parallel"
    assert run_config.parallel.backend == "ddp"
    assert run_config.parallel.num_gpus == 2
    assert run_config.precision.quant_type == "nf4"
    assert run_config.precision.storage_dtype == "fp16"
