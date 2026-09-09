#!/usr/bin/env bash
set -eo pipefail

cd /data/locate_anything_sat/Embodied

export GPUS=1
export MODEL_PATH=/data/LocateAnything-3B
export META_PATH=/data/locate_anything_sat/Embodied/data/dota_v1_hbb_448_mix_v1/recipes/dota_v1_hbb_448_mix_v1_train_only.json
export OUTPUT_DIR=/data/locate_anything_sat/Embodied/work_dirs/dota_v1_hbb_448_lora_1gpu_smoke_default_merge
export MAX_STEPS=5
export LR=2e-5
export DEEPSPEED_CONFIG=deepspeed_configs/zero_stage1_config.json
export ATTENTION_IMPLEMENTATION=sdpa
export GRAD_CHECKPOINT=True
export MAX_SEQ_LENGTH=4096
export MAX_NUM_TOKENS=4096
export MAX_NUM_TOKENS_PER_SAMPLE=4096
export PACKING_BUFFER_SIZE=8
export USE_LLM_LORA=64
export USE_BACKBONE_LORA=0
export FREEZE_LLM=True
export FREEZE_BACKBONE=True
export FREEZE_MLP=False
export SAVE_STEPS=1000
export WARMUP_STEPS=2
export PER_DEVICE_BATCH_SIZE=1
export GRADIENT_ACC=1
export LR_SCHEDULER_TYPE=cosine
export LOGGING_STEPS=1
export WEIGHT_DECAY=0.01
export MLP_CONNECTOR_LAYERS=2
export DATALOADER_NUM_WORKERS=2

rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

export CUDA_VISIBLE_DEVICES=0
export NCCL_DEBUG=INFO

EXTRA_ARGS=(
  --attn_implementation "${ATTENTION_IMPLEMENTATION}"
  --max_num_tokens_per_sample "${MAX_NUM_TOKENS_PER_SAMPLE}"
)

LAUNCHER=pytorch python -m torch.distributed.run \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr="127.0.0.1" \
  --nproc_per_node=1 \
  --master_port=29500 \
  eaglevl/train/locany_finetune_magi_stream.py \
  --model_name_or_path "${MODEL_PATH}" \
  --meta_path "${META_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_steps "${MAX_STEPS}" \
  --learning_rate "${LR}" \
  --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
  --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}" \
  --per_device_eval_batch_size "${PER_DEVICE_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACC}" \
  --bf16 True \
  --logging_steps "${LOGGING_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --warmup_steps "${WARMUP_STEPS}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --num_train_epochs 1 \
  --grad_checkpoint "${GRAD_CHECKPOINT}" \
  --mlp_connector_layers "${MLP_CONNECTOR_LAYERS}" \
  --freeze_llm "${FREEZE_LLM}" \
  --freeze_backbone "${FREEZE_BACKBONE}" \
  --freeze_mlp "${FREEZE_MLP}" \
  --use_llm_lora "${USE_LLM_LORA}" \
  --use_backbone_lora "${USE_BACKBONE_LORA}" \
  --vision_select_layer -1 \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
  --block_size 6 \
  --causal_attn False \
  --packing_buffer_size "${PACKING_BUFFER_SIZE}" \
  --max_seq_length "${MAX_SEQ_LENGTH}" \
  --max_num_tokens "${MAX_NUM_TOKENS}" \
  --do_train True \
  --group_by_length False \
  --deepspeed "${DEEPSPEED_CONFIG}" \
  --report_to tensorboard \
  --run_name "dota_v1_hbb_448_lora_1gpu_smoke_default_merge" \
  --remove_unused_columns False \
  --overwrite_output_dir False \
  --save_strategy steps \
  --save_total_limit 3 \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "${OUTPUT_DIR}/training_smoke.log"
