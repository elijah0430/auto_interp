#!/bin/bash
#SBATCH -J neuron_extract
#SBATCH -p amd_a100nv_8
#SBATCH --comment python
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:10
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
SHARD_COUNT=${SHARD_COUNT:-10}
SHARD_PAD_WIDTH=${SHARD_PAD_WIDTH:-2}
GPU_COUNT=${GPU_COUNT:-10}

if [ "$GPU_COUNT" -lt 1 ]; then
  echo "GPU_COUNT must be >= 1 (received $GPU_COUNT)" >&2
  exit 1
fi

BASE_OUTPUT_DIR="$OUTPUT_DIR"
mkdir -p "$BASE_OUTPUT_DIR"

gpu_list_source=${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-}}
if [ -n "$gpu_list_source" ]; then
  IFS=',' read -r -a ALLOCATED_GPU_IDS <<< "$gpu_list_source"
else
  ALLOCATED_GPU_IDS=()
  for ((gpu_idx=0; gpu_idx<GPU_COUNT; gpu_idx++)); do
    ALLOCATED_GPU_IDS+=("$gpu_idx")
  done
fi

if [ "${#ALLOCATED_GPU_IDS[@]}" -lt "$GPU_COUNT" ]; then
  echo "Only ${#ALLOCATED_GPU_IDS[@]} GPUs visible but GPU_COUNT=$GPU_COUNT; reducing GPU_COUNT." >&2
  GPU_COUNT=${#ALLOCATED_GPU_IDS[@]}
fi

if [ "$GPU_COUNT" -lt 1 ]; then
  echo "No GPUs detected for this job after inspection." >&2
  exit 1
fi

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
base = sys.argv[1]
count = int(sys.argv[2])
width = 100.0 / count
def fmt(v, omit_zero=False, omit_hundred=False):
    if omit_zero and abs(v) < 1e-9:
        return ""
    if omit_hundred and abs(v - 100.0) < 1e-9:
        return ""
    if abs(v - round(v)) < 1e-9:
        return f"{int(round(v))}%"
    return f"{v:.6f}%".rstrip("0").rstrip(".")
for idx in range(count):
    start = idx * width
    end = (idx + 1) * width
    start_token = fmt(start, omit_zero=True, omit_hundred=False)
    end_token = fmt(end, omit_zero=False, omit_hundred=(idx == count - 1))
    split = f"{base}[{start_token}:{end_token}]" if (start_token or end_token) else base
    print(split)
PY

mapfile -t SHARD_SPLITS < /tmp/split_ranges.txt
rm -f /tmp/split_ranges.txt

total_shards=${#SHARD_SPLITS[@]}
shard_start=0

while [ "$shard_start" -lt "$total_shards" ]; do
  remaining=$((total_shards - shard_start))
  if [ "$remaining" -lt "$GPU_COUNT" ]; then
    batch_size=$remaining
  else
    batch_size=$GPU_COUNT
  fi

  declare -a batch_pids=()

  for ((offset=0; offset<batch_size; offset++)); do
    shard_idx=$((shard_start + offset))
    shard_split="${SHARD_SPLITS[$shard_idx]}"
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
    gpu_id="${ALLOCATED_GPU_IDS[$offset]}"
    echo "==== Shard $((shard_idx + 1))/${SHARD_COUNT} | split ${shard_split} | output ${shard_output_dir} | GPU ${gpu_id} ===="
    CUDA_VISIBLE_DEVICES="$gpu_id" srun --exclusive -N1 -n1 --gres=gpu:1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-4}" \
      python -m neuron_explainer.preprocessing.extract_neuron_records \
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
    batch_pids+=($!)
  done

  for pid in "${batch_pids[@]}"; do
    wait "$pid"
  done

  shard_start=$((shard_start + batch_size))
done
