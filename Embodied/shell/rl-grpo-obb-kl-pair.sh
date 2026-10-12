#!/usr/bin/env bash
# Two independent single-card jobs; identical protocol, beta .02 versus .05.
set -euo pipefail
cd "$(dirname "$0")/.."
RL_PYTHON=${RL_PYTHON:-/data/locate_anything_sat/Embodied/.venv/bin/python}
TRL_SITE=${TRL_SITE:-/data/rl_framework_probe_20261012/site}
RUN_ROOT=${RUN_ROOT:-/root/rl_runs/obb_kl_pair_20261013}
DEADLINE=${DEADLINE:-2026-10-13T10:00:00+08:00}
export PYTHONPATH="$TRL_SITE:$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec "$RL_PYTHON" scripts/rl/pair_multitask_grpo.py --root "$RUN_ROOT" --deadline "$DEADLINE"
