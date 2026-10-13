# TRL OBB GRPO / KL pair

This is a separate RL path. It preserves the original SFT launcher and PBD runtime.
Use the checkpoint's self-contained LocateAnything runtime; do not replace it with
an upstream Qwen model. TRL 0.24.0 supplies the grouped advantage, clipped policy
objective, old-policy cache and reference KL.

## Server environment and launch

Work root is `Embodied/`. Use the existing pip venv and isolated TRL package target:

```bash
export RL_PYTHON=/data/locate_anything_sat/Embodied/.venv/bin/python
export TRL_SITE=/data/rl_framework_probe_20261012/site
export RUN_ROOT=/root/rl_runs/obb_kl_pair_20261013
export DEADLINE=2026-10-13T10:00:00+08:00
bash shell/rl-grpo-obb-kl-pair.sh
```

Run from a clean, independent Git checkout. Code/configuration must be committed,
pushed and fetched; never copy them with SCP. The original SFT environment,
checkpoint and image dataset are read-only. Only outputs go to `RUN_ROOT`, on the
larger filesystem. Two independent jobs use physical GPU0 and GPU1, not DDP.

The supervisor runs paired multi-image smoke, fresh 2-step + resume equivalence,
20 effective-update timing, a fixed-budget baseline, and then the main experiment.
Any failed preflight prevents main training. The two configuration files differ
only in beta (.02 / .05), and both use SFT checkpoint-16500 and seed42.

## Data and reward

`multitask_grpo_data.py` indexes the complete train records and TaskSpecs by file
position. It validates identities, codec and image availability, and resolves
single-target canonical floating GT when a group is loaded. Dataset files and
pixels are not regenerated or copied.

Each 100-query block contains T6=30, T8=30, T9=20, T1/T4/T5/T7=5 each. Per-task
shuffled pools preserve the source positive/negative population. One generation
group has exactly one sample identity, even when different images share prompt
text. Only the active group's frozen image context is held on the GPU.

Positive rewards require EOS, one legal frame, expected geometry and correct
class. HBB/OBB reward is canonical geometric IoU. Point reward is 1 inside/on GT;
outside it is exp(-distance to GT polygon / target diagonal). Wrong/duplicate/
unterminated outputs score -1. Negative queries reward only a terminated legal
none response. There is no separate angle, box-count, format or reasoning reward.

The fixed initial settings are G8, serial microbatch1, accumulation8, two updates
per generated group, LR5e-7 constant, clip .2, bf16, grad norm cap1, temperature1,
top_p1, no top_k/repetition penalty and 128 completion tokens. Only LLM LoRA is
trained. MoonViT, LDA and MLP are frozen. Slow sampling and causal token probability
calculation use the same model; Fast is used for deployment validation. Coordinate
special tokens must be decoded from completion IDs without stripping them.

## Recoverable, small checkpoints

A checkpoint is not a standalone LocateAnything model. It contains the trained
policy adapter and `locany_dependency.json`, which identifies the immutable SFT
checkpoint supplying frozen visual/MLP/LDA/base weights and runtime.

Optimizer, scheduler, Python/NumPy/Torch/CUDA RNG, frozen SFT reference adapter,
TRL microstep, buffered trajectories/old probabilities and sample identity use
typed JSON plus safetensors (`safe_rl_state.py`). No pickle loader or Transformers
security-check bypass is used. A checksum and COMPLETE marker detect incomplete
states. Data hashes, sampling schedule, beta, seed and SFT config must agree on
resume. The same adapter state can be installed into the original worker using
PEFT `set_peft_model_state_dict`; do not give the adapter folder directly to the
unmodified LocateAnythingWorker constructor.

Example restart after a complete checkpoint:

```bash
export PYTHONPATH="$TRL_SITE:$(pwd)"
CUDA_VISIBLE_DEVICES=0 "$RL_PYTHON" scripts/rl/train_multitask_grpo.py \
  --config configs/rl/obb_grpo_kl002.json \
  --output "$RUN_ROOT/runs/kl002" --phase train --max-steps <common-budget> \
  --resume "$RUN_ROOT/runs/kl002/checkpoint-<step>"
```

Each job keeps latest3, first effective checkpoint and best monitored checkpoint,
with no duplicate copies when a checkpoint belongs to several categories. The
supervisor preserves one resume-proven smoke checkpoint per job and removes its
own redundant smoke checkpoints only after equivalence checks pass.

## Evaluation and budget

The quick set uses 50 balanced/backfilled queries per task (550 total), frozen by
existing manifest identity. Final evaluation uses the existing 3330-query subset.
Both use fixed budgets: single128, multi1024. This differs from the earlier
GT-answer-length-dependent budget, so both jobs recalculate their initial SFT
baseline before training. Do not compare changed-protocol numbers as a paired
improvement over the historical report.

Primary = equal mean positive success at IoU.5 for T6/T8/T9; report mean IoU too.
Point/HBB auxiliary success and negative-none accuracy are separate guards.
Dense T2/T3/T10/T11 metrics are diagnostics. Every approximately one hour the
current model is evaluated without loading a second model replica.

Nonfinite loss/gradients, probability inconsistency or reference mutation stop
immediately. Two consecutive monitor evaluations more than5 percentage points
below baseline on primary/Point/HBB, or more than10 on negative-none, stop that
job. Reward growth alone does not trigger continuation.

The common total-update budget is calculated after both benchmarks from the
slower measured rate, reserving quick/full evaluation and checkpoint time until
the specified deadline. It is rounded down to50, with a minimum200 updates. Both
jobs continue their benchmark states to this same global update budget. If the
minimum requires longer, complete the stage and final evaluation rather than
claiming the deadline was met. Hourly guards may stop one job earlier.

Inspect `status.json`, `budget.json`, `resume_checks.json`, per-job `progress.json`,
`metrics.jsonl`, `groups.jsonl`, `retention.json` and eval metrics. Per-job logs record
actual entropy, KL, clipping, token lengths, reward variance and effective updates.
The supervisor writes `comparison.json` after both final evaluations. This is a
single-seed experiment, not a significance test or proof of new spatial reasoning.

## Overnight time window update

The user authorized additional equal steps for both arms when observed training is faster than the initial benchmark. The supervisor updates a shared budget from the slower wall-clock throughput and the time remaining until 10:00 Beijing, reserving quick/full final evaluation time. Extensions do not depend on reward and cannot override the numerical or regression stops. The budget remains common to both arms; an arm failure prevents further extensions. All new checkpoints, safe state and logs reside under `/root/rl_runs`, on the larger filesystem.

The probability adapter reproduces native Slow prefill and cached autoregressive decoding, with differentiable KV tensors rather than detached prefix caches. This avoids the observed BF16 numerical difference between cache sampling and full-sequence teacher forcing. Exact resume and both production benchmarks must pass again after this change.

## Evaluation precision correction (2026-10-13)

Accelerator wraps the language model forward with bf16 autocast when training starts. The initial baseline was evaluated before this wrapping, while the original completed run's final evaluations retained it. FP32 LoRA layers consequently followed different arithmetic paths; those original baseline-versus-policy numbers are not a controlled weight-only comparison.

`evaluate_suite` now temporarily uses the LM's original forward and restores the training wrapper afterward. Its metadata records `native_worker_no_amp`, matching a fresh deployment worker. The correction was checked through the real evaluator on 22 positive/negative queries spanning all 11 tasks: all outputs matched the native worker and the training wrapper was restored. Completed adapters were independently re-evaluated on all 3,330 queries. Original results remain available for traceability.

`analyze_multitask_grpo.py` provides paired source-image bootstrap, reward-group statistics, standalone adapter reload checks, controlled decoder/precision replay and counterfactual point-condition probes. Its equivalent-OBB representation calculation is a separate diagnostic; it preserves the original rectangle geometry and does not overwrite official predictions or change the current strict reward protocol.
