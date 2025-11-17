#!/bin/bash

# Local launcher for neuron record extraction without Slurm.
# Defaults to running two shards on CUDA devices 2 and 3.

set -eo pipefail

# CONDA_BASE=~/anaconda3/bin/activate
# if [ -f "${CONDA_BASE}/bin/activate" ]; then
#   # Old-style activation script works even if `conda` is not on PATH.
#   source "${CONDA_BASE}/bin/activate" "${CONDA_ENV:-jongwon}"
# else
#   source ~/.bashrc 2>/dev/null || true
#   conda activate ${CONDA_ENV:-jongwon}
# fi
# export PS1=${PS1:-"(batch)"}
# cd ${REPO_DIR:-/home/jongwonlim/automated-interpretability}
export PYTHONPATH=${PYTHONPATH:-${REPO_DIR:-/home/jongwonlim/automated-interpretability}}

if [ ! -f neuron-explainer/setup.py ]; then
  echo "Could not find neuron-explainer/setup.py in ${REPO_DIR:-/home/jongwonlim/automated-interpretability}" >&2
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
SHARD_COUNT=${SHARD_COUNT:-0}
SHARD_PAD_WIDTH=${SHARD_PAD_WIDTH:-2}
LOCAL_GPU_IDS=${LOCAL_GPU_IDS:-"2 3"}

IFS=' ' read -r -a GPU_ID_ARRAY <<< "$LOCAL_GPU_IDS"
GPU_COUNT=${#GPU_ID_ARRAY[@]}

if [ "$GPU_COUNT" -lt 1 ]; then
  echo "LOCAL_GPU_IDS resulted in zero GPUs." >&2
  exit 1
fi

if [ "$SHARD_COUNT" -le 0 ]; then
  SHARD_COUNT=$GPU_COUNT
fi

BASE_OUTPUT_DIR="$OUTPUT_DIR"
mkdir -p "$BASE_OUTPUT_DIR"

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

python - "$DATASET_SPLIT" "$SHARD_COUNT" <<'PY' > /tmp/split_ranges.txt
import sys

spec = sys.argv[1]
count = int(sys.argv[2])

def parse_spec(base_spec: str):
    if "[" not in base_spec:
        return base_spec, 0.0, 100.0
    base, _, bracket = base_spec.partition("[")
    bracket = bracket.rstrip("]")
    start_token, _, end_token = bracket.partition(":")
    def to_percent(token: str, default: float) -> float:
        if not token:
            return default
        token = token.rstrip("%")
        return float(token)
    return base, to_percent(start_token, 0.0), to_percent(end_token, 100.0)

def fmt(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value))}%"
    trimmed = f"{value:.6f}".rstrip("0").rstrip(".")
    return f"{trimmed}%"

base_name, base_start, base_end = parse_spec(spec)
total_width = base_end - base_start
width = total_width / count

splits = []
for idx in range(count):
    start = base_start + idx * width
    end = base_start + (idx + 1) * width
    end = min(end, base_end)
    splits.append(f"{base_name}[{fmt(start)}:{fmt(end)}]")

print("\\n".join(splits))
PY

mapfile -t SHARD_SPLITS < /tmp/split_ranges.txt
rm -f /tmp/split_ranges.txt

echo "Launching ${#SHARD_SPLITS[@]} shards across GPUs: ${LOCAL_GPU_IDS}"

for shard_idx in "${!SHARD_SPLITS[@]}"; do
  shard_split="${SHARD_SPLITS[$shard_idx]}"
  gpu_id="${GPU_ID_ARRAY[$((shard_idx % GPU_COUNT))]}"
  shard_output_dir=$(printf "%s/shard_%0${SHARD_PAD_WIDTH}d" "$BASE_OUTPUT_DIR" "$shard_idx")
  mkdir -p "$shard_output_dir"
  shard_extra_flags=("${extra_flags[@]}")
  if [ -n "$EXPECTED_SEQUENCES" ]; then
    per_shard=$((EXPECTED_SEQUENCES / SHARD_COUNT))
    if [ "$shard_idx" -eq "$((SHARD_COUNT - 1))" ]; then
      per_shard=$((EXPECTED_SEQUENCES - per_shard * (SHARD_COUNT - 1)))
    fi
    shard_extra_flags+=(--expected-sequences "$per_shard")
  fi

  echo "==== Shard $((shard_idx + 1))/${SHARD_COUNT} | split ${shard_split} | GPU ${gpu_id} ===="
  CUDA_VISIBLE_DEVICES="$gpu_id" python -m neuron_explainer.preprocessing.extract_neuron_records \
    --model "$MODEL_ID" \
    --dataset "$DATASET_NAME" \
    --split "$shard_split" \
    --text-column "$TEXT_COLUMN" \
    "${layer_flags[@]}" \
    --sequence-length "$SEQUENCE_LENGTH" \
    --max-sequences "$MAX_SEQUENCES" \
    --top-k "$TOP_K" \
    --random-sample-size "$RANDOM_SAMPLE_SIZE" \
    --output-dir "$shard_output_dir" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    "${neuron_flags[@]}" \
    "${shard_extra_flags[@]}" &
done

wait
echo "All shards completed."
