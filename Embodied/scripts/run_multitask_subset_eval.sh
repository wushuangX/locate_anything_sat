#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export DATA_ROOT=${DATA_ROOT:-/data/dota_multitask_eval_subset}
export MODEL_PATH=${MODEL_PATH:-/data/checkpoints/checkpoint-16500}
export SPLIT=${SPLIT:-val} MODE=${MODE:-fast}
export OUT=${OUT:-/data/eval_results/${SPLIT}_${MODE}}
export PER_TASK=${PER_TASK:-0}
mkdir -p "$OUT"
if [[ -f "$OUT/predictions.jsonl" ]]; then
  echo "Existing predictions; choose a new OUT directory to preserve results." >&2
  exit 1
fi
.venv/bin/python - <<'PY'
import json,os
from pathlib import Path
assert os.environ['SPLIT'] in ('train','val')
assert os.environ['MODE'] in ('fast','hybrid','slow')
config={'data_root':os.environ['DATA_ROOT'],'split':os.environ['SPLIT'],
        'model_path':os.environ['MODEL_PATH'],'output_dir':os.environ['OUT'],
        'generation_mode':os.environ['MODE'],'num_per_task':int(os.environ['PER_TASK']),
        'tasks':[f'T{i}' for i in range(1,12)],'ignore_ioa':.5,'max_new_tokens':4096}
Path(os.environ['OUT'],'config.json').write_text(json.dumps(config,indent=2)+'\n')
PY
echo $$ > "$OUT/RUNNING"
trap 'rm -f "$OUT/RUNNING"' EXIT
.venv/bin/python scripts/eval_dota_multitask.py --config "$OUT/config.json"
touch "$OUT/DONE"
