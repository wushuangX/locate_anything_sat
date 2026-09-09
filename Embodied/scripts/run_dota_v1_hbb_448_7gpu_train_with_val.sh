#!/usr/bin/env bash
set -eo pipefail

cd /data/locate_anything_sat/Embodied
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
export GPUS=7
export HF_TOKEN=${HF_TOKEN:-dummy-local-path}
export MODEL_PATH=/data/LocateAnything-3B
export META_PATH=/data/locate_anything_sat/Embodied/data/dota_v1_hbb_448_mix_v1/recipes/dota_v1_hbb_448_mix_v1_train_only.json
export OUTPUT_DIR=/data/locate_anything_sat/Embodied/work_dirs/dota_v1_hbb_448_lora_7gpu_valmon_run1
unset VISION_MERGE_KERNEL_SIZE
export ATTN_IMPLEMENTATION=sdpa
export GRAD_CHECKPOINT=True
export MAX_SEQ_LENGTH=4096
export MAX_NUM_TOKENS=4096
export MAX_NUM_TOKENS_PER_SAMPLE=4096
export PACKING_BUFFER_SIZE=32
export DEEPSPEED_CONFIG=deepspeed_configs/zero_stage1_config.json
export USE_LLM_LORA=64
export USE_BACKBONE_LORA=0
export FREEZE_LLM=True
export FREEZE_BACKBONE=True
export FREEZE_MLP=False
export MAX_STEPS=5000
export SAVE_STEPS=250
export WARMUP_STEPS=100
export LR=2e-5
export PER_DEVICE_BATCH_SIZE=1
export GRADIENT_ACC=1
export DATALOADER_NUM_WORKERS=4
export NCCL_DEBUG=INFO
export MASTER_PORT=29507

mkdir -p "$OUTPUT_DIR"

nohup bash shell/locate-anything-lora-visual-prompt.sh > "$OUTPUT_DIR/launcher.log" 2>&1 &
echo $! > "$OUTPUT_DIR/train_launcher.pid"

nohup env \
  OUTPUT_DIR="$OUTPUT_DIR" \
  DATA_ROOT=data/dota_v1_hbb_448_mix_v1 \
  ANNOTATION=annotations/DOTA-v1.0_test_t1_hbb_448.jsonl \
  BASE_CODE_DIR=/data/LocateAnything-3B \
  MERGE_KERNEL=2,2 \
  NUM_SAMPLES=50 \
  INTERVAL_SECONDS=120 \
  EVAL_CUDA_VISIBLE_DEVICES=7 \
  EVAL_TIMEOUT_SECONDS=600 \
  EVAL_MAX_NEW_TOKENS=512 \
  bash scripts/monitor_dota_val_checkpoints.sh > "$OUTPUT_DIR/val_monitor_launcher.log" 2>&1 &
echo $! > "$OUTPUT_DIR/val_monitor.pid"

echo "training launcher pid: $(cat "$OUTPUT_DIR/train_launcher.pid")"
echo "val monitor pid: $(cat "$OUTPUT_DIR/val_monitor.pid")"
