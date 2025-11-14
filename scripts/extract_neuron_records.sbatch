#!/bin/bash
#SBATCH --job-name=neuron-extract
#SBATCH --output=logs/neuron-extract-%j.out
#SBATCH --error=logs/neuron-extract-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

# Override any of these via sbatch --export=ALL,VAR=value ...
CONDA_ENV=${CONDA_ENV:-neuron-explainer}
MODEL_ID=${MODEL_ID:-meta-llama/Meta-Llama-3-8B-Instruct}
TOKENIZER_ID=${TOKENIZER_ID:-$MODEL_ID}
DATASET_NAME=${DATASET_NAME:-openwebtext}
DATASET_CONFIG=${DATASET_CONFIG:-}
DATASET_SPLIT=${DATASET_SPLIT:-train}
TEXT_COLUMN=${TEXT_COLUMN:-text}
LAYER_INDEX=${LAYER_INDEX:-10}
SEQUENCE_LENGTH=${SEQUENCE_LENGTH:-128}
STRIDE=${STRIDE:-$SEQUENCE_LENGTH}
MAX_SEQUENCES=${MAX_SEQUENCES:-5000}
TOP_K=${TOP_K:-20}
RANDOM_SAMPLE_SIZE=${RANDOM_SAMPLE_SIZE:-50}
OUTPUT_DIR=${OUTPUT_DIR:-$SCRATCH/neuron_records}
DEVICE=${DEVICE:-cuda}
DTYPE=${DTYPE:-float16}
TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-0}
STREAMING=${STREAMING:-0}
ALL_NEURONS=${ALL_NEURONS:-0}
NEURON_INDICES=${NEURON_INDICES:-"42 256"}

mkdir -p logs "$OUTPUT_DIR"

source "$HOME/.bashrc"
conda activate "$CONDA_ENV"

extra_flags=()
if [[ -n "$DATASET_CONFIG" ]]; then
  extra_flags+=(--dataset-config "$DATASET_CONFIG")
fi
if [[ "$STREAMING" == "1" ]]; then
  extra_flags+=(--streaming)
fi
if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
  extra_flags+=(--trust-remote-code)
fi
if [[ "$STRIDE" != "$SEQUENCE_LENGTH" ]]; then
  extra_flags+=(--stride "$STRIDE")
fi
if [[ -n "$TOKENIZER_ID" && "$TOKENIZER_ID" != "$MODEL_ID" ]]; then
  extra_flags+=(--tokenizer "$TOKENIZER_ID")
fi

if [[ "$ALL_NEURONS" == "1" ]]; then
  neuron_flags=(--all-neurons)
else
  read -r -a neuron_array <<< "$NEURON_INDICES"
  neuron_flags=(--neuron-indices "${neuron_array[@]}")
fi

srun python -m neuron_explainer.preprocessing.extract_neuron_records \
  --model "$MODEL_ID" \
  --dataset "$DATASET_NAME" \
  --split "$DATASET_SPLIT" \
  --text-column "$TEXT_COLUMN" \
  --layer-index "$LAYER_INDEX" \
  --sequence-length "$SEQUENCE_LENGTH" \
  --max-sequences "$MAX_SEQUENCES" \
  --top-k "$TOP_K" \
  --random-sample-size "$RANDOM_SAMPLE_SIZE" \
  --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  "${neuron_flags[@]}" \
  "${extra_flags[@]}"
