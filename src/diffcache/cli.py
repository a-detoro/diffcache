"""`diffcache run ...` -- builds a RunConfig from CLI flags and dispatches it
via `runner.run`. For `--phase data_parallel`, launch under the matching
distributed launcher (see parallel/zero.py, parallel/ddp.py docstrings):

    deepspeed --num_gpus=2 --module diffcache.cli run --pipeline qwen_image_edit \\
        --phase data_parallel --parallel zero2 --precomputed-dir ./precomputed_bench

    torchrun --nproc_per_node=2 -m diffcache.cli run --pipeline qwen_image_edit \\
        --phase data_parallel --parallel ddp --precomputed-dir ./precomputed_bench
"""
from __future__ import annotations

import argparse
import logging
import sys

from .config import add_run_config_args, run_config_from_args
from .runner import run as run_phase


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="diffcache")
    # `deepspeed`'s launcher injects `--local_rank=N` ahead of the
    # subcommand for every spawned process (the old torch.distributed.launch
    # convention) -- accepted and ignored here, same as every original
    # *_bench.py script in this project does, since LOCAL_RANK is read from
    # the environment instead (see parallel/zero.py, parallel/ddp.py).
    parser.add_argument("--local_rank", type=int, default=-1, help=argparse.SUPPRESS)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run one phase of one pipeline with a given configuration.")
    add_run_config_args(run_parser)

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    if args.command == "run":
        run_config = run_config_from_args(args)
        run_phase(run_config)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
