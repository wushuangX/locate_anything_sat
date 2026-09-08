#!/usr/bin/env bash
# Post-training batch evaluation of DOTA val checkpoints (no sidecar monitor).
# Complements run_dota_v1_hbb_448_2gpu_train.sh: iterates over saved
# checkpoint-*/ (including milestones/), copies remote-code files from the base
# model dir when a checkpoint lacks them, pins merge_kernel_size, and runs
# scripts/eval_baseline_dota.py once per checkpoint.
#
# Usage:
#   OUTPUT_DIR=/data/locate_anything_sat/Embodied/work_dirs/dota_v1_hbb_448_lora_2gpu_run1 \
#     bash scripts/eval_dota_val_checkpoints_batch.sh
#
# Useful overrides: NUM_SAMPLES, EVAL_CUDA_VISIBLE_DEVICES, EVAL_TIMEOUT_SECONDS,
# EVAL_MAX_NEW_TOKENS, DATA_ROOT, ANNOTATION, BASE_CODE_DIR, MERGE_KERNEL.
set -eo pipefail

OUTPUT_DIR=${OUTPUT_DIR:?Set OUTPUT_DIR to the training output dir}
DATA_ROOT=${DATA_ROOT:-data/dota_v1_hbb_448}
ANNOTATION=${ANNOTATION:-annotations/DOTA-v1.0_val_hbb_448.jsonl}
NUM_SAMPLES=${NUM_SAMPLES:-50}
BASE_CODE_DIR=${BASE_CODE_DIR:-/data/LocateAnything-3B}
MERGE_KERNEL=${MERGE_KERNEL:-2,2}
EVAL_DIR=${EVAL_DIR:-${DATA_ROOT}/_eval_results}
LOG_DIR=${LOG_DIR:-${OUTPUT_DIR}/val_eval_logs}
EVAL_CUDA_VISIBLE_DEVICES=${EVAL_CUDA_VISIBLE_DEVICES:-0}
EVAL_TIMEOUT_SECONDS=${EVAL_TIMEOUT_SECONDS:-1200}
EVAL_MAX_NEW_TOKENS=${EVAL_MAX_NEW_TOKENS:-512}

mkdir -p "$EVAL_DIR" "$LOG_DIR"

copy_remote_code() {
  local ckpt="$1"
  for f in \
    configuration_locateanything.py configuration_qwen2.py \
    modeling_locateanything.py modeling_qwen2.py modeling_vit.py \
    generate_utils.py image_processing_locateanything.py processing_locateanything.py \
    mask_magi_utils.py mask_sdpa_utils.py \
    added_tokens.json chat_template.json chat_template.jinja generation_config.json \
    merges.txt processor_config.json special_tokens_map.json tokenizer_config.json vocab.json; do
    if [[ -f "${BASE_CODE_DIR}/${f}" && ! -f "${ckpt}/${f}" ]]; then
      cp "${BASE_CODE_DIR}/${f}" "${ckpt}/${f}"
    fi
  done
  MERGE_KERNEL="$MERGE_KERNEL" python - "$ckpt" <<'PY'
import json, os, sys
from pathlib import Path
ckpt = Path(sys.argv[1])
kernel = [int(part.strip()) for part in os.environ["MERGE_KERNEL"].split(",")]
if len(kernel) != 2 or kernel[0] <= 0 or kernel[1] <= 0:
    raise ValueError("MERGE_KERNEL must be H,W with two positive integers")
for name in ["preprocessor_config.json", "processor_config.json"]:
    p = ckpt / name
    if p.exists():
        d = json.loads(p.read_text())
        d["merge_kernel_size"] = kernel
        p.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n")
PY
}

shopt -s nullglob
ckpts=()
for d in "$OUTPUT_DIR" "$OUTPUT_DIR/milestones"; do
  for c in "$d"/checkpoint-*/; do
    [[ -f "${c}model.safetensors.index.json" ]] && ckpts+=("${c%/}")
  done
done

if [[ ${#ckpts[@]} -eq 0 ]]; then
  echo "No complete checkpoints (missing model.safetensors.index.json) under $OUTPUT_DIR" >&2
  exit 1
fi

# Order by step number; dedupe milestone vs rolling copies of the same step.
ordered_file="$(mktemp)"
printf '%s\n' "${ckpts[@]}" \
  | awk -F'/checkpoint-' 'NF>1 {print $NF "\t" $0}' \
  | sort -t"$(printf '\t')" -k1,1n -u \
  | cut -f2- > "$ordered_file"
ckpts=()
while IFS= read -r line; do
  [[ -n "$line" ]] && ckpts+=("$line")
done < "$ordered_file"
rm -f "$ordered_file"

echo "evaluating ${#ckpts[@]} checkpoints: ${ckpts[*]}"

for ckpt in "${ckpts[@]}"; do
  parent="$(basename "$(dirname "$ckpt")")"
  if [[ "$parent" == "milestones" ]]; then
    name="milestones_$(basename "$ckpt")"
  else
    name="$(basename "$ckpt")"
  fi
  marker="${LOG_DIR}/${name}.done"
  if [[ -f "$marker" ]]; then
    echo "[$(date '+%F %T')] skip ${ckpt} (already evaluated)"
    continue
  fi
  echo "[$(date '+%F %T')] evaluating ${ckpt}"
  copy_remote_code "$ckpt"
  out="${EVAL_DIR}/$(basename "$OUTPUT_DIR")_${name}_${NUM_SAMPLES}.json"
  if timeout "$EVAL_TIMEOUT_SECONDS" env CUDA_VISIBLE_DEVICES=${EVAL_CUDA_VISIBLE_DEVICES} PYTHONUNBUFFERED=1 \
    python scripts/eval_baseline_dota.py \
      --model-path "$ckpt" \
      --data-root "$DATA_ROOT" \
      --annotation "$ANNOTATION" \
      --num-samples "$NUM_SAMPLES" \
      --max-new-tokens "$EVAL_MAX_NEW_TOKENS" \
      --output-file "$out" \
      > "${LOG_DIR}/${name}.log" 2>&1; then
    touch "$marker"
    echo "[$(date '+%F %T')] ok -> ${out}"
  else
    status=$?
    echo "[$(date '+%F %T')] eval failed or timed out with status ${status}; skipping" | tee -a "${LOG_DIR}/batch_eval.log"
    touch "${LOG_DIR}/${name}.failed" "$marker"
  fi
  tail -n 5 "${LOG_DIR}/${name}.log"
done

echo "done. results in ${EVAL_DIR}, logs in ${LOG_DIR}"
