#!/usr/bin/env bash
# Gate the broader RL-single pool before paired, independent single-GPU GRPO runs.
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate
export HF_HOME="${HF_HOME:-/root/autodl-tmp/cache/}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"

run_root="${RUN_ROOT:-work_dirs/rl_v2_pair_seed43}"
gate_dir="$run_root/gate_small2"
model_path=work_dirs/dota_geom_lora_lda_2gpu_4k_100k_run1
ann=data/dota_v1_hbb_448_rl_v2/annotations/DOTA-v1.0_train_rl_single_hbb_448.jsonl
data_root=data/dota_v1_hbb_448_rl_v2
mkdir -p "$run_root"
if [[ -e "$run_root/control_small10" || -e "$run_root/expanded_small2" ]]; then
  echo "[pair] Training output already exists under $run_root; refusing to mix runs" >&2
  exit 1
fi
if [[ -e "$gate_dir" ]]; then
  if [[ "${RESUME_GATE:-0}" != 1 || ! -d "$gate_dir" || -f "$gate_dir/summary_gate.json" ]]; then
    echo "[pair] Existing gate is not an incomplete resume target; refusing to overwrite it" >&2
    exit 1
  fi
  echo "[pair] $(date -Is) resuming gate from completed (sample_id,k) rows in $gate_dir"
else
  if [[ "${RESUME_GATE:-0}" == 1 ]]; then
    echo "[pair] Missing gate output under $gate_dir; cannot resume" >&2
    exit 1
  fi
  mkdir "$gate_dir"
  echo "[pair] $(date -Is) gate: 512 samples x G16, smallGT>=2, two 4090 shards"
fi
for gpu in 0 1; do
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/rl/freeze_check.py \
    --stage gate --model-path "$model_path" --ann "$ann" --data-root "$data_root" \
    --out-dir "$gate_dir" --seed 43 --min-small-gt 2 \
    --samples 512 --n 16 --shard "$gpu" --num-shards 2 \
    >> "$run_root/gate_gpu${gpu}.log" 2>&1 &
  gate_pids[$gpu]=$!
done

gate_failed=0
for gpu in 0 1; do
  if ! wait "${gate_pids[$gpu]}"; then
    echo "[pair] gate shard $gpu failed; see $run_root/gate_gpu${gpu}.log" >&2
    gate_failed=1
  fi
done
if (( gate_failed )); then
  exit 1
fi
python scripts/rl/freeze_check.py --stage summarize --ann "$ann" \
  --out-dir "$gate_dir" --seed 43 --min-small-gt 2 --samples 512 --n 16 \
  > "$run_root/gate_summary.log" 2>&1 || {
    echo "[pair] gate failed; no policy updates started" >&2
    exit 1
  }
python - "$gate_dir/summary_gate.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
assert summary["all_pass"] and summary["min_small_gt"] == 2 and summary["seed"] == 43
assert summary["diagnostics"]["n_groups"] == 512
assert summary["diagnostics"]["n_total"] == 8192
PY

echo "[pair] $(date -Is) gate passed; starting GPU0 control >=10 and GPU1 expanded >=2"
for gpu in 0 1; do
  if (( gpu == 0 )); then
    threshold=10
    name=control_small10
  else
    threshold=2
    name=expanded_small2
  fi
  CUDA_VISIBLE_DEVICES="$gpu" MODEL_PATH="$model_path" ANN="$ann" DATA_ROOT="$data_root" \
    OUTPUT_DIR="$run_root/$name" SEED=43 MIN_SMALL_GT="$threshold" \
    N=16 LR=5e-7 KL_BETA=0.02 MAX_STEPS=200 SAVE_STEPS=50 \
    bash shell/rl-grpo-hbb-dense.sh > "$run_root/${name}.log" 2>&1 &
  train_pids[$gpu]=$!
  echo "[pair] GPU$gpu $name pid=${train_pids[$gpu]} log=$run_root/${name}.log"
done

train_failed=0
for gpu in 0 1; do
  if ! wait "${train_pids[$gpu]}"; then
    echo "[pair] GPU$gpu training failed" >&2
    train_failed=1
  fi
done
echo "[pair] $(date -Is) trainings exited; failed=$train_failed"
exit "$train_failed"
