"""Utilities for building `NeuronRecord` blobs from Hugging Face causal language models.

This script lets you reproduce the activation datasets that ship with the paper for your own
models/corpora. It streams text from a Hugging Face dataset, runs the target model, captures the
post-MLP activations for a specified layer, and writes JSON files that match the `NeuronRecord`
schema consumed by the rest of this repo.

Example:

```
python -m neuron_explainer.preprocessing.extract_neuron_records \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset openwebtext --split train --text-column text \
    --layer-index 15 --neuron-indices 42 123 \
    --sequence-length 128 --max-sequences 5000 \
    --output-dir ./llama3_neuron_records
```

This depends on optional packages (`torch`, `transformers`, `datasets`, `tqdm`) that are not part
of the minimal `neuron_explainer` install_requires list.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_explainer.activations.activations import ActivationRecord, NeuronId, NeuronRecord
from neuron_explainer.fast_dataclasses import dumps


DEFAULT_PROJECTION_TEMPLATES: Dict[str, str] = {
    "gpt2": "transformer.h[{layer}].mlp.c_proj",
    "llama": "model.layers[{layer}].mlp.down_proj",
    "mistral": "model.layers[{layer}].mlp.down_proj",
    "qwen2": "model.layers[{layer}].mlp.down_proj",
    "qwen2_moe": "model.layers[{layer}].mlp.down_proj",
}

DTYPE_LOOKUP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate NeuronRecord JSON files from a Hugging Face model."
    )
    parser.add_argument("--model", required=True, help="Hugging Face model id or local path.")
    parser.add_argument(
        "--tokenizer",
        help="Tokenizer id to use (defaults to the same as --model).",
    )
    parser.add_argument(
        "--dataset",
        default="openwebtext",
        help=(
            "Dataset name/path passed to datasets.load_dataset (default: openwebtext). "
            "Set to 'valueeval' to read the Touché23-ValueEval TSV files."
        ),
    )
    parser.add_argument(
        "--dataset-config",
        default=None,
        help="Optional dataset config passed to load_dataset.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split passed to load_dataset (default: train).",
    )
    parser.add_argument(
        "--valueeval-dir",
        type=Path,
        default=Path.home() / "data" / "valueeval",
        help="Directory containing the Touché23-ValueEval TSV files (used when --dataset valueeval).",
    )
    parser.add_argument(
        "--valueeval-splits",
        nargs="+",
        default=[
            "arguments-training.tsv",
            "arguments-validation.tsv",
            "arguments-validation-zhihu.tsv",
            "arguments-test.tsv",
            "arguments-test-nahjalbalagha.tsv",
        ],
        help=(
            "List of TSV filenames (relative to --valueeval-dir) to include when --dataset valueeval "
            "(default: arguments-training/validation/test variants)."
        ),
    )
    parser.add_argument(
        "--valueeval-text-column",
        default="Premise",
        help="Column to read from the ValueEval TSV files when --dataset valueeval (default: Premise).",
    )
    parser.add_argument(
        "--text-column",
        default="text",
        help="Column inside the dataset that contains raw text.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=128,
        help="Number of tokens per activation record (default: 128).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help=(
            "Stride (in tokens) between successive chunks. Defaults to sequence_length (no overlap)."
        ),
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=1000,
        help="Maximum number of token sequences to score (<=0 means no limit, default: 1000).",
    )
    parser.add_argument(
        "--expected-sequences",
        type=int,
        default=None,
        help=(
            "Optional expected number of sequences for progress reporting when --max-sequences <= 0. "
            "Set to the total window count if you want tqdm to display a concrete denominator."
        ),
    )
    parser.add_argument(
        "--layer-index",
        type=int,
        help="Index of a single transformer block to capture (legacy flag).",
    )
    parser.add_argument(
        "--layer-indices",
        type=int,
        nargs="+",
        help="Optional list of layer indices to capture.",
    )
    parser.add_argument(
        "--all-layers",
        action="store_true",
        help="Capture activations for every transformer block in the model.",
    )
    neuron_group = parser.add_mutually_exclusive_group(required=True)
    neuron_group.add_argument(
        "--neuron-indices",
        type=int,
        nargs="+",
        help="Specific neuron indices to record.",
    )
    neuron_group.add_argument(
        "--all-neurons",
        action="store_true",
        help="If set, records every neuron in the chosen layer (can be large).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="How many top-activating sequences to keep for each neuron.",
    )
    parser.add_argument(
        "--random-sample-size",
        type=int,
        default=50,
        help="How many random sequences to reservoir-sample for each neuron.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where a neurons/<layer>/<neuron>.json tree will be written.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device string (default: cuda if available else cpu).",
    )
    parser.add_argument(
        "--dtype",
        choices=sorted(DTYPE_LOOKUP.keys()),
        default="float16",
        help="Torch dtype for model weights.",
    )
    parser.add_argument(
        "--mlp-projection-template",
        help=(
            "Optional dotted path template (with '{layer}') pointing to the module whose "
            "forward pre-hook receives the per-neuron activations. "
            "Defaults to a known template for gpt2/llama/mistral/qwen2."
        ),
    )
    parser.add_argument(
        "--max-position-embeddings",
        type=int,
        default=None,
        help="Override the model's maximum supported context length for validation purposes.",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="If set, loads the dataset in streaming mode.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to AutoModel/Tokenizer (needed for some newer models).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for reservoir sampling.",
    )
    parser.add_argument(
        "--no-fast-tokenizer",
        action="store_true",
        help="Use the slow tokenizer implementation.",
    )
    return parser.parse_args()


def dataset_text_iterator(dataset, text_column: str) -> Iterator[str]:
    for row in dataset:
        text = row.get(text_column) if isinstance(row, dict) else getattr(row, text_column, None)
        if isinstance(text, str) and text.strip():
            yield text


def valueeval_text_iterator(
    data_dir: Path,
    splits: Sequence[str],
    text_column: str,
) -> Iterator[str]:
    for split in splits:
        path = data_dir / split
        if not path.exists():
            raise FileNotFoundError(
                f"Could not find {path}. Download the Touché23-ValueEval TSVs first."
            )
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            if reader.fieldnames is None or text_column not in reader.fieldnames:
                raise ValueError(
                    f"Expected column '{text_column}' in {path}, found {reader.fieldnames}."
                )
            for row in reader:
                text = (row.get(text_column) or "").strip()
                if text:
                    yield text


def token_sequences_from_corpus(
    texts: Iterable[str],
    tokenizer,
    sequence_length: int,
    stride: Optional[int],
    max_sequences: Optional[int],
) -> Iterator[List[int]]:
    buffer: List[int] = []
    produced = 0
    stride = stride or sequence_length
    for text in texts:
        tokenized = tokenizer(
            text,
            add_special_tokens=False,
        )
        ids = tokenized["input_ids"]
        if not ids:
            continue
        buffer.extend(ids)
        while len(buffer) >= sequence_length:
            yield buffer[:sequence_length]
            produced += 1
            if max_sequences is not None and produced >= max_sequences:
                return
            buffer = buffer[stride:]


class LayerActivationHook:
    """Captures activations by registering a forward pre-hook on a projection module."""

    def __init__(self, module: torch.nn.Module):
        self.buffer: Optional[torch.Tensor] = None
        self.handle = module.register_forward_pre_hook(self._hook)

    def _hook(self, module: torch.nn.Module, inputs: Sequence[torch.Tensor]) -> None:
        self.buffer = inputs[0].detach().cpu()

    def pop(self) -> torch.Tensor:
        if self.buffer is None:
            raise RuntimeError("No activations captured in the previous forward pass.")
        tensor = self.buffer
        self.buffer = None
        return tensor

    def remove(self) -> None:
        self.handle.remove()


@dataclass
class ActivationStats:
    """Maintains streaming summaries, a top-k heap, and a reservoir sample."""

    top_k: int
    random_sample_size: int

    def __post_init__(self) -> None:
        # Heap stores (score, tie_breaker, record) so items stay comparable.
        self._top_heap: List[tuple[float, int, ActivationRecord]] = []
        self._reservoir: List[ActivationRecord] = []
        self._seen_records = 0
        self._count = 0
        self._mean = 0.0
        self._m2 = 0.0
        self._tie_counter = 0

    def add(self, tokens: Sequence[str], activations: Sequence[float]) -> None:
        activation_list = [float(x) for x in activations]
        record = ActivationRecord(tokens=list(tokens), activations=activation_list)
        score = max(activation_list) if activation_list else float("-inf")
        if self.top_k > 0:
            self._tie_counter += 1
            item = (score, self._tie_counter, record)
            if len(self._top_heap) < self.top_k:
                heapq.heappush(self._top_heap, item)
            else:
                if score > self._top_heap[0][0]:
                    heapq.heapreplace(self._top_heap, item)
        self._seen_records += 1
        if self.random_sample_size > 0:
            if len(self._reservoir) < self.random_sample_size:
                self._reservoir.append(record)
            else:
                idx = random.randint(0, self._seen_records - 1)
                if idx < self.random_sample_size:
                    self._reservoir[idx] = record
        for value in activation_list:
            self._count += 1
            delta = value - self._mean
            self._mean += delta / self._count
            delta2 = value - self._mean
            self._m2 += delta * delta2

    @property
    def mean(self) -> float:
        return self._mean if self._count > 0 else math.nan

    @property
    def variance(self) -> float:
        if self._count < 2:
            return math.nan
        return self._m2 / (self._count - 1)

    def top_records(self) -> List[ActivationRecord]:
        return [record for _, _, record in sorted(self._top_heap, key=lambda item: item[0], reverse=True)]

    def random_records(self) -> List[ActivationRecord]:
        return list(self._reservoir)


def resolve_projection_module(model: torch.nn.Module, template: str, layer_index: int) -> torch.nn.Module:
    path = template.format(layer=layer_index)
    current = model
    for chunk in path.split("."):
        if "[" in chunk and chunk.endswith("]"):
            attr, index_str = chunk[:-1].split("[", maxsplit=1)
            current = getattr(current, attr)
            index = int(index_str)
            current = current[index]
        else:
            current = getattr(current, chunk)
    if not isinstance(current, torch.nn.Module):
        raise ValueError(f"Resolved path '{path}' is not a torch.nn.Module.")
    return current


def determine_target_layers(
    num_layers: int,
    *,
    layer_index: Optional[int],
    layer_indices: Optional[Sequence[int]],
    all_layers: bool,
) -> List[int]:
    if all_layers:
        return list(range(num_layers))
    if layer_indices:
        unique = sorted(set(layer_indices))
        for idx in unique:
            if idx < 0 or idx >= num_layers:
                raise ValueError(f"Layer index {idx} is outside [0, {num_layers}).")
        return unique
    if layer_index is not None:
        if layer_index < 0 or layer_index >= num_layers:
            raise ValueError(f"layer_index {layer_index} is outside [0, {num_layers}).")
        return [layer_index]
    raise ValueError(
        "Specify --layer-index, --layer-indices, or --all-layers to choose which layers to capture."
    )


def determine_target_neurons(
    ff_dim: int, neuron_indices: Optional[Sequence[int]], all_neurons: bool
) -> List[int]:
    if all_neurons:
        return list(range(ff_dim))
    assert neuron_indices is not None
    selected = []
    for idx in neuron_indices:
        if idx < 0 or idx >= ff_dim:
            raise ValueError(f"Neuron index {idx} is out of bounds for dimension {ff_dim}.")
        selected.append(idx)
    return sorted(set(selected))


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    tokenizer_id = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id,
        use_fast=not args.no_fast_tokenizer,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=DTYPE_LOOKUP[args.dtype],
        trust_remote_code=args.trust_remote_code,
        device_map=None,
    )
    model.to(device)
    model.eval()
    torch.set_grad_enabled(False)

    num_layers = getattr(model.config, "n_layer", None) or getattr(
        model.config, "num_hidden_layers", None
    )
    if num_layers is None:
        raise ValueError("Could not infer the number of transformer layers for this model.")

    max_positions = (
        args.max_position_embeddings
        or getattr(model.config, "n_positions", None)
        or getattr(model.config, "max_position_embeddings", None)
    )
    if max_positions and args.sequence_length > max_positions:
        raise ValueError(
            f"sequence_length {args.sequence_length} exceeds the model limit ({max_positions})."
        )

    target_layers = determine_target_layers(
        num_layers,
        layer_index=args.layer_index,
        layer_indices=args.layer_indices,
        all_layers=args.all_layers,
    )

    template = args.mlp_projection_template or DEFAULT_PROJECTION_TEMPLATES.get(
        model.config.model_type
    )
    if template is None:
        raise ValueError(
            "Model type not recognized and no --mlp-projection-template provided. "
            "Specify a dotted path such as 'transformer.h[{layer}].mlp.c_proj'."
        )
    hooks: dict[int, LayerActivationHook] = {}
    for layer_idx in target_layers:
        projection_module = resolve_projection_module(model, template, layer_idx)
        hooks[layer_idx] = LayerActivationHook(projection_module)

    dataset_name = args.dataset.lower()
    if dataset_name == "valueeval":
        valueeval_dir = args.valueeval_dir.expanduser()
        text_iter = valueeval_text_iterator(
            valueeval_dir,
            args.valueeval_splits,
            args.valueeval_text_column,
        )
    else:
        dataset = load_dataset(
            args.dataset,
            args.dataset_config,
            split=args.split,
            streaming=args.streaming,
        )
        text_iter = dataset_text_iterator(dataset, args.text_column)
    stride = args.stride or args.sequence_length
    max_sequences = args.max_sequences if args.max_sequences > 0 else None
    sequence_iter = token_sequences_from_corpus(
        text_iter,
        tokenizer,
        args.sequence_length,
        stride,
        max_sequences,
    )

    aggregators: Dict[int, Dict[int, ActivationStats]] = {}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    expected_sequences = (
        args.expected_sequences if args.expected_sequences and args.expected_sequences > 0 else None
    )
    progress_total = max_sequences if max_sequences is not None else expected_sequences
    progress = tqdm(total=progress_total, desc="Sequences")
    layer_pbars = {
        layer_idx: tqdm(
            total=max_sequences if max_sequences is not None else None,
            leave=False,
            desc=f"Layer {layer_idx}",
        )
        for layer_idx in target_layers
    }
    sequences_processed = 0
    start_time = time.time()
    for token_ids in sequence_iter:
        input_ids = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
        sequences_processed += 1
        tokens_buffer = tokenizer.convert_ids_to_tokens(token_ids)
        model(input_ids=input_ids)
        for layer_idx, hook in hooks.items():
            activations = hook.pop().to(torch.float32).squeeze(0)
            layer_stats = aggregators.get(layer_idx)
            if layer_stats is None:
                ff_dim = activations.shape[-1]
                target_indices = determine_target_neurons(
                    ff_dim, args.neuron_indices, args.all_neurons
                )
                layer_stats = {
                    idx: ActivationStats(
                        top_k=args.top_k, random_sample_size=args.random_sample_size
                    )
                    for idx in target_indices
                }
                aggregators[layer_idx] = layer_stats
            for neuron_idx, stats in layer_stats.items():
                neuron_activations = activations[:, neuron_idx].tolist()
                stats.add(tokens_buffer, neuron_activations)
            layer_pbars[layer_idx].update(1)
        progress.update(1)
        if sequences_processed % 25 == 0 or (progress.total is None and sequences_processed <= 5):
            elapsed = max(time.time() - start_time, 1e-6)
            seq_per_min = sequences_processed / elapsed * 60.0
            progress.set_postfix(
                processed=sequences_processed,
                seq_per_min=f"{seq_per_min:.1f}",
                elapsed=f"{elapsed/60.0:.1f}m",
            )
    progress.close()
    for pbar in layer_pbars.values():
        pbar.close()
    for hook in hooks.values():
        hook.remove()

    if not aggregators:
        raise RuntimeError("No sequences were processed; nothing to write.")

    neurons_base = output_dir / "neurons"
    for layer_idx, neuron_map in aggregators.items():
        neurons_dir = neurons_base / str(layer_idx)
        neurons_dir.mkdir(parents=True, exist_ok=True)
        for neuron_idx, stats in neuron_map.items():
            neuron_record = NeuronRecord(
                neuron_id=NeuronId(layer_index=layer_idx, neuron_index=neuron_idx),
                random_sample=stats.random_records(),
                random_sample_by_quantile=None,
                quantile_boundaries=None,
                mean=stats.mean,
                variance=stats.variance,
                skewness=math.nan,
                kurtosis=math.nan,
                most_positive_activation_records=stats.top_records(),
            )
            output_path = neurons_dir / f"{neuron_idx}.json"
            with output_path.open("wb") as f:
                f.write(dumps(neuron_record))
    print(
        f"Wrote {sum(len(x) for x in aggregators.values())} neuron files "
        f"across {len(aggregators)} layer(s) to {neurons_base}. "
        f"Processed {sequences_processed} sequences."
    )


if __name__ == "__main__":
    main()
