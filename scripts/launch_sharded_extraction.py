#!/usr/bin/env python3
"""
Helper for launching multiple Slurm jobs that each process a non-overlapping shard of the dataset.

Example:

    python scripts/launch_sharded_extraction.py \
        --num-shards 4 \
        --base-output-dir /scratch/x3069a10/auto_interp/output_openweb_sharded \
        --sbatch-script scripts/extract_neuron_records.sbatch \
        --env MODEL_ID=meta-llama/Llama-3.1-8B-Instruct \
        --env DATASET_NAME=/scratch/x3069a10/datasets/openwebtext_local \
        --env DATASET_CONFIG=default \
        --env ALL_LAYERS=1 --env ALL_NEURONS=1 --env MAX_SEQUENCES=0

Each shard receives its own DATASET_SPLIT (e.g. train[:25%], train[25%:50%], …) and
OUTPUT_DIR=<base-output-dir>/shard_<idx>. The rest of the environment variables are forwarded to
the Slurm script unchanged.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Dict, List


def parse_env_overrides(pairs: List[str]) -> Dict[str, str]:
    env: Dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Expected KEY=VALUE for --env, got '{pair}'")
        key, value = pair.split("=", maxsplit=1)
        env[key.strip()] = value.strip()
    return env


def fmt_percent(value: float, *, omit_zero: bool, omit_hundred: bool) -> str:
    if (omit_zero and abs(value) < 1e-9) or (omit_hundred and abs(value - 100.0) < 1e-9):
        return ""
    if abs(value - round(value)) < 1e-6:
        return f"{int(round(value))}%"
    return f"{value:.4f}%".rstrip("0").rstrip(".")


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch sbatch jobs over dataset shards.")
    parser.add_argument(
        "--sbatch-script",
        default="scripts/extract_neuron_records.sbatch",
        help="Path to the sbatch script to invoke (default: scripts/extract_neuron_records.sbatch).",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        required=True,
        help="Number of equal-sized dataset shards to launch.",
    )
    parser.add_argument(
        "--base-split",
        default="train",
        help="Base datasets split to subdivide (default: train).",
    )
    parser.add_argument(
        "--base-output-dir",
        required=True,
        help="Directory where shard_<idx> subfolders will be created.",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="Additional KEY=VALUE env overrides forwarded to sbatch. Repeatable.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the sbatch commands without executing them.",
    )
    args = parser.parse_args()

    if args.num_shards <= 0:
        raise SystemExit("--num-shards must be > 0")

    base_env = parse_env_overrides(args.env)
    base_output = Path(args.base_output_dir).resolve()
    base_output.mkdir(parents=True, exist_ok=True)

    shard_width = 100.0 / args.num_shards
    for shard_idx in range(args.num_shards):
        start_pct = shard_idx * shard_width
        end_pct = (shard_idx + 1) * shard_width
        start_token = fmt_percent(start_pct, omit_zero=True, omit_hundred=False)
        end_token = fmt_percent(
            end_pct,
            omit_zero=False,
            omit_hundred=True and shard_idx == args.num_shards - 1,
        )
        split_expr = (
            f"{args.base_split}[{start_token}:{end_token}]"
            if start_token or end_token
            else args.base_split
        )

        shard_env = base_env.copy()
        shard_env["DATASET_SPLIT"] = split_expr
        shard_output = base_output / f"shard_{shard_idx:02d}"
        shard_env["OUTPUT_DIR"] = str(shard_output)
        shard_env["SHARD_INDEX"] = str(shard_idx)
        shard_env["NUM_SHARDS"] = str(args.num_shards)

        export_arg = "ALL," + ",".join(f"{k}={v}" for k, v in shard_env.items())
        cmd = ["sbatch", "--export", export_arg, args.sbatch_script]
        print(" ".join(cmd))
        if not args.dry_run:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(result.stdout.strip())


if __name__ == "__main__":
    main()
