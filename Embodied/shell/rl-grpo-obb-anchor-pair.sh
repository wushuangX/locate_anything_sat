#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${TRL_SITE:-/data/rl_framework_probe_20261012/site}:.${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
RL_OUTPUT_ROOT="${RL_OUTPUT_ROOT:-/root/rl_runs/obb_grpo_anchor_$(date +%Y%m%d_%H%M%S)}"
exec "${RL_PYTHON:-/data/locate_anything_sat/Embodied/.venv/bin/python}" scripts/rl/pair_multitask_grpo_v2.py --root "$RL_OUTPUT_ROOT" --hours "${RL_HOURS:-12}"
