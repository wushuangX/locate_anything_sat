#!/usr/bin/env bash
set -eo pipefail

OUTPUT_DIR=${OUTPUT_DIR:?Set OUTPUT_DIR}
DATA_ROOT=${DATA_ROOT:-data/dota_v1_hbb_448_mix_v1}
ANNOTATION=${ANNOTATION:-annotations/DOTA-v1.0_test_t1_hbb_448.jsonl}
NUM_SAMPLES=${NUM_SAMPLES:-50}
INTERVAL_SECONDS=${INTERVAL_SECONDS:-120}
BASE_CODE_DIR=${BASE_CODE_DIR:-/data/LocateAnything-3B}
EVAL_DIR=${EVAL_DIR:-${DATA_ROOT}/_eval_results}
LOG_DIR=${LOG_DIR:-${OUTPUT_DIR}/val_eval_logs}
MERGE_KERNEL=${MERGE_KERNEL:-2,2}
EVAL_TIMEOUT_SECONDS=${EVAL_TIMEOUT_SECONDS:-1200}
EVAL_MAX_NEW_TOKENS=${EVAL_MAX_NEW_TOKENS:-256}

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

latest_step() {
  find "$OUTPUT_DIR" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null \
    | sed 's/.*checkpoint-//' \
    | sort -n \
    | tail -1
}

while true; do
  step=$(latest_step || true)
  if [[ -n "${step:-}" ]]; then
    ckpt="${OUTPUT_DIR}/checkpoint-${step}"
    marker="${LOG_DIR}/checkpoint-${step}.done"
    if [[ ! -f "$marker" && -f "${ckpt}/model.safetensors.index.json" ]]; then
      echo "[$(date '+%F %T')] evaluating checkpoint-${step}" | tee -a "${LOG_DIR}/monitor.log"
      copy_remote_code "$ckpt"
      out="${EVAL_DIR}/$(basename "$OUTPUT_DIR")_checkpoint-${step}_${NUM_SAMPLES}.json"
      if timeout "$EVAL_TIMEOUT_SECONDS" env CUDA_VISIBLE_DEVICES=${EVAL_CUDA_VISIBLE_DEVICES:-7} PYTHONUNBUFFERED=1 python scripts/eval_baseline_dota.py \
        --model-path "$ckpt" \
        --data-root "$DATA_ROOT" \
        --annotation "$ANNOTATION" \
        --num-samples "$NUM_SAMPLES" \
        --max-new-tokens "$EVAL_MAX_NEW_TOKENS" \
        --output-file "$out" \
        > "${LOG_DIR}/checkpoint-${step}.log" 2>&1; then
        touch "$marker"
      else
        status=$?
        echo "[$(date '+%F %T')] checkpoint-${step} eval failed or timed out with status ${status}; skipping" | tee -a "${LOG_DIR}/monitor.log"
        touch "${LOG_DIR}/checkpoint-${step}.failed" "$marker"
      fi
      tail -40 "${LOG_DIR}/checkpoint-${step}.log" | tee -a "${LOG_DIR}/monitor.log"
    fi
  fi
  if [[ -f "${OUTPUT_DIR}/done.txt" ]]; then
    echo "[$(date '+%F %T')] training done; monitor exiting" | tee -a "${LOG_DIR}/monitor.log"
    exit 0
  fi
  sleep "$INTERVAL_SECONDS"
done
