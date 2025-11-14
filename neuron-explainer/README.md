# Neuron explainer

This directory contains a version of our code for generating, simulating and scoring explanations of
neuron behavior.

# Setup

```
pip install -e .
```

# Usage

For example usage, see the `demos` folder:

* [Generating and scoring activation-based explanations](demos/generate_and_score_explanation.ipynb)
* [Generating and scoring explanations based on tokens with high average activations](demos/generate_and_score_token_look_up_table_explanation.ipynb)
* [Generating explanations for human-written neuron puzzles](demos/explain_puzzles.ipynb)

## Building activation datasets

The GPT-2 release that ships with this repo was produced by running GPT-2 on the original WebText
("internet text") corpus. To support other architectures (e.g., Llama 3 or Qwen 2.5) you first need
matching activation blobs. The `extract_neuron_records` helper streams any Hugging Face dataset
through a model and writes `NeuronRecord` JSON files in the format expected by the explainer:

```
python -m neuron_explainer.preprocessing.extract_neuron_records \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset openwebtext --split train --text-column text \
    --layer-index 10 --neuron-indices 42 256 \
    --sequence-length 128 --max-sequences 5000 \
    --output-dir ./llama3_records
```

The script depends on optional packages (`torch`, `transformers`, `datasets`, `tqdm`) and writes
files to `<output>/neurons/<layer>/<neuron>.json`. Point `dataset_path` at that directory to reuse
all of the generation, simulation, and scoring utilities.

Add `--all-layers` to sweep every transformer block in one run, or pass `--layer-indices 0 1 2` to limit processing to a subset.

By default the CLI keeps the top 100 activation records per neuron (ordered strongest to weakest), which downstream tooling interprets in that same rank order.

If you run on a Slurm cluster, `scripts/extract_neuron_records.sbatch` wraps the same command and exposes the key arguments via environment overrides (e.g. `sbatch --export=ALL,MODEL_ID=... scripts/extract_neuron_records.sbatch`). Set `ALL_LAYERS=1` or `LAYER_INDICES="0 1 2"` before submission to mirror the CLI options.

