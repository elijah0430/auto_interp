#!/bin/bash
#SBATCH -J neuron_extract
#SBATCH -p amd_a100nv_8
#SBATCH --comment python
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

set -eo pipefail

module load intel/19.1.2 cuda/11.4
module load python/3.7.1
export PS1=${PS1:-"(batch)"}
source activate ${CONDA_ENV:-neuron-explainer}
cd ${REPO_DIR:-$HOME/automated-interpretability}
export PYTHONPATH=${PYTHONPATH:-${REPO_DIR:-$HOME/automated-interpretability}}
if [ ! -f neuron-explainer/setup.py ]; then
  echo "Could not find neuron-explainer/setup.py in ${REPO_DIR:-$HOME/automated-interpretability}" >&2
  exit 1
fi
python -m pip install -e neuron-explainer >/dev/null


MODEL_ID=${MODEL_ID:-meta-llama/Meta-Llama-3-8B-Instruct}
TOKENIZER_ID=${TOKENIZER_ID:-$MODEL_ID}
DATASET_NAME=${DATASET_NAME:-openwebtext}
DATASET_CONFIG=${DATASET_CONFIG:-}
DATASET_SPLIT=${DATASET_SPLIT:-train}
TEXT_COLUMN=${TEXT_COLUMN:-text}
LAYER_INDEX=${LAYER_INDEX:-10}
LAYER_INDICES=${LAYER_INDICES:-}
ALL_LAYERS=${ALL_LAYERS:-0}
SEQUENCE_LENGTH=${SEQUENCE_LENGTH:-128}
STRIDE=${STRIDE:-$SEQUENCE_LENGTH}
MAX_SEQUENCES=${MAX_SEQUENCES:-5000}
TOP_K=${TOP_K:-100}
RANDOM_SAMPLE_SIZE=${RANDOM_SAMPLE_SIZE:-50}
OUTPUT_DIR=${OUTPUT_DIR:-$PWD/output_neuron_records}
DEVICE=${DEVICE:-cuda}
DTYPE=${DTYPE:-float16}
TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-0}
STREAMING=${STREAMING:-0}
ALL_NEURONS=${ALL_NEURONS:-0}
NEURON_INDICES=${NEURON_INDICES:-"42 256"}
VALUEEVAL_DIR=${VALUEEVAL_DIR:-$HOME/data/valueeval}
VALUEEVAL_SPLITS=${VALUEEVAL_SPLITS:-"arguments-training.tsv arguments-validation.tsv arguments-validation-zhihu.tsv arguments-test.tsv arguments-test-nahjalbalagha.tsv"}
VALUEEVAL_TEXT_COLUMN=${VALUEEVAL_TEXT_COLUMN:-Premise}
EXPECTED_SEQUENCES=${EXPECTED_SEQUENCES:-}
SHARD_COUNT=${SHARD_COUNT:-1}
SHARD_INDEX=${SHARD_INDEX:-0}
SHARD_PAD_WIDTH=${SHARD_PAD_WIDTH:-2}

base_output_dir="$OUTPUT_DIR"

if [ "$SHARD_COUNT" -gt 1 ]; then
  if [ "$SHARD_INDEX" -lt 0 ] || [ "$SHARD_INDEX" -ge "$SHARD_COUNT" ]; then
    echo "SHARD_INDEX ($SHARD_INDEX) must be within [0, SHARD_COUNT=$SHARD_COUNT)" >&2
    exit 1
  fi
  shard_dir=$(printf "%s/shard_%0${SHARD_PAD_WIDTH}d" "$base_output_dir" "$SHARD_INDEX")
  OUTPUT_DIR="$shard_dir"
  DATASET_SPLIT=$(
    python - "$DATASET_SPLIT" "$SHARD_INDEX" "$SHARD_COUNT" <<'PY'
import sys
base, shard_idx, shard_count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
width = 100.0 / shard_count
start = shard_idx * width
end = (shard_idx + 1) * width
def fmt(value, omit_zero=False, omit_hundred=False):
    if omit_zero and abs(value) < 1e-9:
        return ""
    if omit_hundred and abs(value - 100.0) < 1e-9:
        return ""
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value))}%"
    return f"{value:.6f}%".rstrip("0").rstrip(".")
start_token = fmt(start, omit_zero=True, omit_hundred=False)
end_token = fmt(end, omit_zero=False, omit_hundred=(shard_idx == shard_count - 1))
if start_token or end_token:
    print(f"{base}[{start_token}:{end_token}]")
else:
    print(base)
PY
  )
  echo "Shard ${SHARD_INDEX}/${SHARD_COUNT} -> split ${DATASET_SPLIT}, output ${OUTPUT_DIR}"
fi

mkdir -p "$OUTPUT_DIR"

extra_flags=()
if [ -n "$DATASET_CONFIG" ]; then
  extra_flags+=(--dataset-config "$DATASET_CONFIG")
fi
if [ "$STREAMING" = "1" ]; then
  extra_flags+=(--streaming)
fi
if [ "$TRUST_REMOTE_CODE" = "1" ]; then
  extra_flags+=(--trust-remote-code)
fi
if [ "$STRIDE" != "$SEQUENCE_LENGTH" ]; then
  extra_flags+=(--stride "$STRIDE")
fi
if [ -n "$TOKENIZER_ID" ] && [ "$TOKENIZER_ID" != "$MODEL_ID" ]; then
  extra_flags+=(--tokenizer "$TOKENIZER_ID")
fi
if [ "$DATASET_NAME" = "valueeval" ]; then
  extra_flags+=(--valueeval-dir "$VALUEEVAL_DIR")
  extra_flags+=(--valueeval-text-column "$VALUEEVAL_TEXT_COLUMN")
  read -r -a valueeval_split_array <<< "$VALUEEVAL_SPLITS"
  extra_flags+=(--valueeval-splits)
  extra_flags+=("${valueeval_split_array[@]}")
fi
if [ -n "$EXPECTED_SEQUENCES" ]; then
  extra_flags+=(--expected-sequences "$EXPECTED_SEQUENCES")
fi

if [ "$ALL_NEURONS" = "1" ]; then
  neuron_flags=(--all-neurons)
else
  read -r -a neuron_array <<< "$NEURON_INDICES"
  neuron_flags=(--neuron-indices "${neuron_array[@]}")
fi

if [ "$ALL_LAYERS" = "1" ]; then
  layer_flags=(--all-layers)
elif [ -n "$LAYER_INDICES" ]; then
  read -r -a layer_array <<< "$LAYER_INDICES"
  layer_flags=(--layer-indices "${layer_array[@]}")
else
  layer_flags=(--layer-index "$LAYER_INDEX")
fi

srun python -m neuron_explainer.preprocessing.extract_neuron_records \
  --model "$MODEL_ID" \
  --dataset "$DATASET_NAME" \
  --split "$DATASET_SPLIT" \
  --text-column "$TEXT_COLUMN" \
  "${layer_flags[@]}" \
  --sequence-length "$SEQUENCE_LENGTH" \
  --max-sequences "$MAX_SEQUENCES" \
  --top-k "$TOP_K" \
  --random-sample-size "$RANDOM_SAMPLE_SIZE" \
  --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  "${neuron_flags[@]}" \
  "${extra_flags[@]}"
