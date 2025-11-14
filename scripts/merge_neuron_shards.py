#!/usr/bin/env python3
"""
Merge multiple shard outputs (neurons/<layer>/<neuron>.json trees) into a single directory.

The script re-aggregates the activation samples from each shard using the same ActivationStats
helper as the extractor. This keeps the top-k and random reservoir sizes bounded while combining
data from every shard.
"""

from __future__ import annotations

import argparse
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

from neuron_explainer.activations.activations import ActivationRecord, NeuronId, NeuronRecord
from neuron_explainer.fast_dataclasses import loads, dumps
from neuron_explainer.preprocessing.extract_neuron_records import ActivationStats


def load_neuron_record(path: Path) -> NeuronRecord:
    with path.open("rb") as handle:
        data = handle.read()
    return loads(data)


def activation_records(record: NeuronRecord) -> Iterable[ActivationRecord]:
    yield from record.most_positive_activation_records
    yield from record.random_sample
    if record.random_sample_by_quantile:
        for bucket in record.random_sample_by_quantile:
            yield from bucket


def merge_records(
    records: List[NeuronRecord],
    *,
    top_k: int,
    random_sample_size: int,
) -> NeuronRecord:
    stats = ActivationStats(top_k=top_k, random_sample_size=random_sample_size)
    for record in records:
        for activation in activation_records(record):
            stats.add(activation.tokens, activation.activations)
    neuron_id = records[0].neuron_id
    return NeuronRecord(
        neuron_id=NeuronId(layer_index=neuron_id.layer_index, neuron_index=neuron_id.neuron_index),
        random_sample=stats.random_records(),
        random_sample_by_quantile=None,
        quantile_boundaries=None,
        mean=stats.mean,
        variance=stats.variance,
        skewness=math.nan,
        kurtosis=math.nan,
        most_positive_activation_records=stats.top_records(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge neuron shard outputs into a single tree.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Destination directory for the merged neurons/<layer>/<neuron>.json tree.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of top activation records to keep per neuron (default: 100).",
    )
    parser.add_argument(
        "--random-sample-size",
        type=int,
        default=50,
        help="Reservoir size for random samples when merging (default: 50).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for the merging reservoir sampling (default: 1234).",
    )
    parser.add_argument(
        "shard_dirs",
        nargs="+",
        help="One or more shard output directories to merge.",
    )
    args = parser.parse_args()

    if args.top_k <= 0:
        raise SystemExit("--top-k must be positive.")
    if args.random_sample_size < 0:
        raise SystemExit("--random-sample-size must be >= 0.")

    random.seed(args.seed)
    shards = [Path(p).resolve() for p in args.shard_dirs]
    for shard in shards:
        if not (shard / "neurons").exists():
            raise SystemExit(f"{shard} does not contain a neurons/ directory.")

    grouped: Dict[str, List[NeuronRecord]] = defaultdict(list)
    for shard in shards:
        for neuron_file in (shard / "neurons").rglob("*.json"):
            rel_key = str(neuron_file.relative_to(shard / "neurons"))
            grouped[rel_key].append(load_neuron_record(neuron_file))

    output_dir = Path(args.output_dir).resolve()
    neurons_dir = output_dir / "neurons"
    neurons_dir.mkdir(parents=True, exist_ok=True)

    for rel_key, records in grouped.items():
        merged = merge_records(
            records,
            top_k=args.top_k,
            random_sample_size=args.random_sample_size,
        )
        dest = neurons_dir / rel_key
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as handle:
            handle.write(dumps(merged))
        print(f"wrote {dest}")


if __name__ == "__main__":
    main()
