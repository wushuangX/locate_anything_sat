#!/usr/bin/env bash
# DOTA 几何增强长时程 LoRA+LDA 训练（2×4090 / SDPA / 4K token）
# 数据：五 key 几何 recipe（0°/rot90/rot180/rot270/hflip，全部 color_jitter）
# 默认 10 万步；所有变量可环境变量覆盖。
set -euo pipefail

unset CONDA_SHLVL
unset CONDA_EXE
unset _CE_CONDA
unset CONDA_PREFIX
unset CONDA_PROMPT_MODIFIER
unset CONDA_PYTHON_EXE
unset CONDA_DEFAULT_ENV
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v 'anaconda3' | paste -sd ':' -)
pip install hf_xet

export WANDB_PROJECT="${WANDB_PROJECT:-star-nemo}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-dota-geom-lora-lda-100k}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"
if [[ -z "${HF_TOKEN:-}" ]]; then
  HF_TOKEN=$(cat /root/.cache/huggingface/token 2>/dev/null || echo "")
fi
export HF_TOKEN

GPUS=${GPUS:-2}
NNODES=${1:-1}
OUTPUT_DIR=${2:-${OUTPUT_DIR:-"work_dirs/dota_geom_lora_lda_2gpu_4k_100k_run1"}}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

MODEL_PATH=${MODEL_PATH:-"nvidia/LocateAnything-3B"}
META_PATH=${META_PATH:-"data/dota_v1_hbb_448_mix_v1/recipes/dota_v1_hbb_448_mix_v1_geom_train_only.json"}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-"deepspeed_configs/zero_stage1_config.json"}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-sdpa}
GRAD_CHECKPOINT=${GRAD_CHECKPOINT:-True}

PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRADIENT_ACC=${GRADIENT_ACC:-2}
MAX_STEPS=${MAX_STEPS:-100000}
SAVE_STEPS=${SAVE_STEPS:-500}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-3}
LR=${LR:-1.5e-5}
WARMUP_STEPS=${WARMUP_STEPS:-500}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-4096}
MAX_NUM_TOKENS_PER_SAMPLE=${MAX_NUM_TOKENS_PER_SAMPLE:-4096}
MAX_NUM_TOKENS=${MAX_NUM_TOKENS:-4096}
PACKING_BUFFER_SIZE=${PACKING_BUFFER_SIZE:-32}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-4}

USE_LLM_LORA=${USE_LLM_LORA:-64}
USE_BACKBONE_LORA=${USE_BACKBONE_LORA:-0}
FREEZE_LLM=${FREEZE_LLM:-True}
FREEZE_BACKBONE=${FREEZE_BACKBONE:-True}
FREEZE_MLP=${FREEZE_MLP:-False}

# 基座 + Local Detail Adapter（基座 config 中 LDA 为关，训练时显式开启）
USE_LDA=${USE_LDA:-true}
LDA_BOTTLENECK_DIM=${LDA_BOTTLENECK_DIM:-128}

EXTRA_ARGS=()
if [[ -n "${VISION_MERGE_KERNEL_SIZE:-}" ]]; then
  EXTRA_ARGS+=(--vision_merge_kernel_size "$VISION_MERGE_KERNEL_SIZE")
fi
EXTRA_ARGS+=(--use_lda "$USE_LDA")
EXTRA_ARGS+=(--lda_bottleneck_dim "$LDA_BOTTLENECK_DIM")

mkdir -p "$OUTPUT_DIR"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"

script_name=$(basename "${BASH_SOURCE[0]}")

LAUNCHER=pytorch python -m torch.distributed.run \
  --nnodes="$NNODES" \
  --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" \
  --nproc_per_node="$GPUS" \
  --master_port="$PORT" \
  eaglevl/train/locany_finetune_magi_stream.py \
  --model_name_or_path "$MODEL_PATH" \
  --max_steps "$MAX_STEPS" \
  --output_dir "$OUTPUT_DIR" \
  --meta_path "$META_PATH" \
  --overwrite_output_dir False \
  --block_size 6 \
  --attn_implementation "$ATTN_IMPLEMENTATION" \
  --causal_attn False \
  --freeze_llm "$FREEZE_LLM" \
  --freeze_mlp "$FREEZE_MLP" \
  --freeze_backbone "$FREEZE_BACKBONE" \
  --use_llm_lora "$USE_LLM_LORA" \
  --use_backbone_lora "$USE_BACKBONE_LORA" \
  --vision_select_layer -1 \
  --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
  --bf16 True \
  --num_train_epochs 1 \
  --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACC" \
  --save_strategy "steps" \
  --save_steps "$SAVE_STEPS" \
  --save_total_limit "$SAVE_TOTAL_LIMIT" \
  --learning_rate "$LR" \
  --weight_decay 0.01 \
  --warmup_steps "$WARMUP_STEPS" \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --video_total_pixels 8192 \
  --sample_log_interval 1 \
  --packing_buffer_size "$PACKING_BUFFER_SIZE" \
  --max_seq_length "$MAX_SEQ_LENGTH" \
  --max_num_tokens_per_sample "$MAX_NUM_TOKENS_PER_SAMPLE" \
  --max_num_tokens "$MAX_NUM_TOKENS" \
  --do_train True \
  --grad_checkpoint "$GRAD_CHECKPOINT" \
  --group_by_length False \
  --deepspeed "$DEEPSPEED_CONFIG" \
  --report_to "tensorboard" \
  --run_name "$script_name" \
  --use_onelogger True \
  --mlp_connector_layers 2 \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
