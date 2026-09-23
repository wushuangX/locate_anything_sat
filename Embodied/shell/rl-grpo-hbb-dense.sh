#!/usr/bin/env bash
# GRPO v2 on RL-single annotations (full single-class GT, reward v3 + reference KL).
# Starts from the 100k HBB geom LoRA. Single 4090, no DeepSpeed.
#一次采样、一次更新：group-normalized on-policy policy gradient + KL(β=0.02)。
set -euo pipefail

cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate

export HF_HOME="${HF_HOME:-/root/autodl-tmp/cache/}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"

MODEL_PATH=${MODEL_PATH:-work_dirs/dota_geom_lora_lda_2gpu_4k_100k_run1}
ANN=${ANN:-data/dota_v1_hbb_448_rl_v2/annotations/DOTA-v1.0_train_rl_single_hbb_448.jsonl}
DATA_ROOT=${DATA_ROOT:-data/dota_v1_hbb_448_rl_v2}
OUTPUT_DIR=${OUTPUT_DIR:-work_dirs/rl_grpo_hbb_dense_v2}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-}
MAX_STEPS=${MAX_STEPS:-250}
N=${N:-16}
LR=${LR:-5e-7}
SAVE_STEPS=${SAVE_STEPS:-50}
MIN_SMALL_GT=${MIN_SMALL_GT:-10}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
MAX_OUTPUT_BOXES=${MAX_OUTPUT_BOXES:-256}
MIN_REWARD_STD=${MIN_REWARD_STD:-0.02}
KL_BETA=${KL_BETA:-0.02}
LOSS_TOKEN_NORMALIZER=${LOSS_TOKEN_NORMALIZER:-512}

mkdir -p "$OUTPUT_DIR"

RESUME_ARGS=()
if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
  RESUME_ARGS=(--resume-from-checkpoint "$RESUME_FROM_CHECKPOINT")
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python scripts/rl/grpo_loop.py \
  --model-path "$MODEL_PATH" \
  --ann "$ANN" \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  "${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}" \
  --max-steps "$MAX_STEPS" \
  --n "$N" \
  --lr "$LR" \
  --save-steps "$SAVE_STEPS" \
  --min-small-gt "$MIN_SMALL_GT" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --max-output-boxes "$MAX_OUTPUT_BOXES" \
  --min-reward-std "$MIN_REWARD_STD" \
  --kl-beta "$KL_BETA" \
  --loss-token-normalizer "$LOSS_TOKEN_NORMALIZER"
