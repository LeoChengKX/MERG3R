#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash run.sh <dataset_path> <output_dir> [extra main.py args...]"
  echo "Example: bash run.sh data/tanks_and_temples/Barn/images outputs/barn/run01"
  exit 1
fi

DATASET="$1"
OUTPUT_DIR="$2"
shift 2

if [[ ! -d "${DATASET}" && ! -f "${DATASET}" ]]; then
  echo "Error: dataset path not found: ${DATASET}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

# Set the path to the dinov3 repo and the weights url prior to running the script
export MERG3R_DINOV3_REPO_DIR=./dinov3
export MERG3R_DINOV3_WEIGHTS_URL=https://...

python main.py \
  --dataset "${DATASET}" \
  --output_dir "${OUTPUT_DIR}" \
  --model pi3 \
  --sequence_type shortest_path \
  --alignment_type weighted_iterative \
  --subset_size 100 \
  --overlap 5 \
  --num_images 150 \
  --subsample 1 \
  --splitting_type interleave \
  --tracking_type graph \
  --global_ba \
  --lr 3e-3 \
  --epoch 300 \
  --max_reproj 8.0 \
  --stride 10 \
  --alpha 0.7 \
  --point_vis_threshold 20.0 \
  "$@"
