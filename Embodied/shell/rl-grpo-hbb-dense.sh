#!/usr/bin/env bash
# GRPO on small-dense tiles only (smallGT>=10), G=16, 2500 steps.
# Starts from 100k HBB geom LoRA. Single 4090, no DeepSpeed/KL.
set -euo pipefail

cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate

export HF_HOME="${HF_HOME:-/root/autodl-tmp/cache/}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"

MODEL_PATH=${MODEL_PATH:-work_dirs/dota_geom_lora_lda_2gpu_4k_100k_run1}
ANN=${ANN:-data/dota_v1_hbb_448_mix_v1/annotations/DOTA-v1.0_train_mix_hbb_448.jsonl}
DATA_ROOT=${DATA_ROOT:-data/dota_v1_hbb_448_mix_v1}
OUTPUT_DIR=${OUTPUT_DIR:-work_dirs/rl_grpo_hbb_dense_g16_2k5}
MAX_STEPS=${MAX_STEPS:-2500}
N=${N:-16}
LR=${LR:-1e-6}
SAVE_STEPS=${SAVE_STEPS:-250}
MIN_SMALL_GT=${MIN_SMALL_GT:-10}

mkdir -p "$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python scripts/rl/grpo_loop.py \
  --model-path "$MODEL_PATH" \
  --ann "$ANN" \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --max-steps "$MAX_STEPS" \
  --n "$N" \
  --lr "$LR" \
  --save-steps "$SAVE_STEPS" \
  --min-small-gt "$MIN_SMALL_GT"
