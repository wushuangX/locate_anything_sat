#!/usr/bin/env bash
# 2x RTX 4090 launcher for DOTA-v1.0 HBB 448 LoRA fine-tuning (39.104.160.156).
#
# Differences vs run_dota_v1_hbb_448_7gpu_train_with_val.sh:
#   - GPUS=2 (CUDA 0,1); the 7-GPU script's CUDA 0-6 + sidecar GPU 7 do not exist here.
#   - No sidecar val monitor: run scripts/eval_dota_val_checkpoints_batch.sh after
#     training (or against saved checkpoints mid-run) instead.
#   - GRADIENT_ACC=2 + LR=1.5e-5: effective batch is ~16.4k tokens/step vs ~28.7k on
#     7 GPUs; LoRA is robust to the smaller batch. Raise MAX_STEPS toward ~8750 if
#     you want to match the 7-GPU run's total token budget (~143M).
#   - SAVE_TOTAL_LIMIT=20: keep every 250-step checkpoint (~9 GB each, ~187 GB
#     total for 5000 steps) so the post-training eval sweep has dense coverage;
#     the 7-GPU default is 3.
set -eo pipefail

cd /data/locate_anything_sat/Embodied
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1
export GPUS=2
export HF_TOKEN=${HF_TOKEN:-dummy-local-path}
export MODEL_PATH=/data/LocateAnything-3B
export META_PATH=/data/locate_anything_sat/Embodied/data/dota_v1_hbb_448/recipes/dota_v1_hbb_448_train_only.json
export OUTPUT_DIR=/data/locate_anything_sat/Embodied/work_dirs/dota_v1_hbb_448_lora_2gpu_run1
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
export SAVE_TOTAL_LIMIT=20
export WARMUP_STEPS=100
export LR=1.5e-5
export PER_DEVICE_BATCH_SIZE=1
export GRADIENT_ACC=2
export DATALOADER_NUM_WORKERS=4
export NCCL_DEBUG=INFO

mkdir -p "$OUTPUT_DIR"

nohup bash shell/locate-anything-lora-visual-prompt.sh > "$OUTPUT_DIR/launcher.log" 2>&1 &
echo $! > "$OUTPUT_DIR/train_launcher.pid"

echo "training launcher pid: $(cat "$OUTPUT_DIR/train_launcher.pid")"
echo "output dir: $OUTPUT_DIR"
echo "after training, evaluate checkpoints with:"
echo "  OUTPUT_DIR=$OUTPUT_DIR bash scripts/eval_dota_val_checkpoints_batch.sh"
