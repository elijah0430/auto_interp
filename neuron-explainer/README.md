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

When you want to stream the entire dataset, set `--max-sequences 0`. In that scenario the loader has no natural denominator, so you can optionally pass `--expected-sequences <N>` to tell `tqdm` how many fixed-size windows to anticipate. Even without a denominator the progress bar now shows elapsed time and throughput (sequences/minute) so you can extrapolate progress for long-running jobs.

### Touché23-ValueEval corpus

To use the dataset released with “Touché23-ValueEval: Identification of Human Values Behind Arguments”
(LREC-COLING 2024, DOI [10.5281/zenodo.6814563](https://doi.org/10.5281/zenodo.6814563)):

1. Download the TSV bundle into `~/data/valueeval` (or any directory you prefer):
   ```
   bash scripts/download_valueeval_dataset.sh  # optionally pass a custom target dir
   ```
2. Run the extractor with the `valueeval` corpus preset:
   ```
   python -m neuron_explainer.preprocessing.extract_neuron_records \
       --model meta-llama/Meta-Llama-3-8B-Instruct \
       --dataset valueeval \
       --valueeval-dir ~/data/valueeval \
       --valueeval-splits arguments-training.tsv arguments-validation.tsv \
       --all-layers --all-neurons \
       --sequence-length 128 --max-sequences 5000 \
       --output-dir ./llama3_valueeval_neurons
   ```
   The default split list already covers the main train/validation/test files, and the script reads
   the `Premise` column from each TSV (override via `--valueeval-text-column` if needed).
3. On Slurm, set the corresponding overrides when calling the sbatch wrapper:
   ```
   sbatch --export=ALL,DATASET_NAME=valueeval,VALUEEVAL_DIR=/home/$USER/data/valueeval,\
VALUEEVAL_SPLITS="arguments-training.tsv arguments-validation.tsv",ALL_LAYERS=1,ALL_NEURONS=1 \
       scripts/extract_neuron_records.sbatch
   ```

If you run on a Slurm cluster, `scripts/extract_neuron_records.sbatch` wraps the same command and exposes the key arguments via environment overrides (e.g. `sbatch --export=ALL,MODEL_ID=... scripts/extract_neuron_records.sbatch`). Set `ALL_LAYERS=1` or `LAYER_INDICES="0 1 2"` before submission to mirror the CLI options.

### Sharding long jobs on Slurm

Large corpora are easier to manage when you split them into non-overlapping slices and launch one job per slice. The helper below takes care of emitting sbatch commands with appropriate `DATASET_SPLIT` and `OUTPUT_DIR` overrides:

```
python scripts/launch_sharded_extraction.py \
    --num-shards 4 \
    --base-output-dir /scratch/$USER/auto_interp/output_openweb_sharded \
    --sbatch-script scripts/extract_neuron_records.sbatch \
    --env MODEL_ID=meta-llama/Llama-3.1-8B-Instruct \
    --env DATASET_NAME=/scratch/$USER/datasets/openwebtext_local \
    --env DATASET_CONFIG=default \
    --env ALL_LAYERS=1 \
    --env ALL_NEURONS=1 \
    --env MAX_SEQUENCES=0
```

Each shard submits an sbatch job targeting `train[start%:end%]` and writes into `<base-output-dir>/shard_XX`. After the jobs finish, combine the results:

```
python scripts/merge_neuron_shards.py \
    --output-dir /scratch/$USER/auto_interp/output_openweb_merged \
    /scratch/$USER/auto_interp/output_openweb_sharded/shard_00 \
    /scratch/$USER/auto_interp/output_openweb_sharded/shard_01 \
    /scratch/$USER/auto_interp/output_openweb_sharded/shard_02 \
    /scratch/$USER/auto_interp/output_openweb_sharded/shard_03
```

The merger reuses the extractor’s `ActivationStats` logic to keep top-k activations and random samples bounded while aggregating all shards into a single `neurons/<layer>/<neuron>.json` tree.
