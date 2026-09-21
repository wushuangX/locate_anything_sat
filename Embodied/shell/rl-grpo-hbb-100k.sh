#!/usr/bin/env bash
# RL Phase B GRPO on HBB 100k geom LoRA（单卡 4090 / NTP / 无 DeepSpeed / 无 KL）
# 所有变量可环境变量覆盖。不用 recipe（无 META_PATH）。
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
OUTPUT_DIR=${OUTPUT_DIR:-work_dirs/rl_grpo_hbb_100k}
MAX_STEPS=${MAX_STEPS:-500}
N=${N:-8}
LR=${LR:-1e-6}
SAVE_STEPS=${SAVE_STEPS:-50}

mkdir -p "$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python scripts/rl/grpo_loop.py \
  --model-path "$MODEL_PATH" \
  --ann "$ANN" \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --max-steps "$MAX_STEPS" \
  --n "$N" \
  --lr "$LR" \
  --save-steps "$SAVE_STEPS"
