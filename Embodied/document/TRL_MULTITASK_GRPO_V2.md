# Multi-query GRPO + PBD supervision (v2)

This experiment starts both independent RTX4090 jobs from SFT checkpoint-16500.
GPU0 uses anchor_weight=0; GPU1 uses anchor_weight=0.1. Both use beta=0.02,
G=8, four queries per generation batch, microbatch=1, accumulation=32 and two
iterations. All other RL settings match the original validated TRL0.24 profile.

The v1 entry and configs are retained. V2 runs use train_multitask_grpo_v2.py;
v1 checkpoints cannot be silently resumed as v2 experiments.

## Supervision and probabilities

Each raw sampled trajectory carries a query row and trajectory ID through TRL
shuffle, split, old-probability caching and checkpoint restore. Images are never
resolved by prompt text. Only the current four-query visual contexts are cached.

PBD supervision uses the original SFT MTP label builder (identical AST), block7,
original position resets and the checkpoint Qwen2 training SDPA mask. Its loss
includes both original AR labels and masked blocks. The frozen visual features
are inserted into cloned embeddings. Hidden states feed chunked CE with
activation checkpointing and explicit bf16 training autocast, avoiding a full
sequence-by-vocabulary logits tensor. Sampling probability checks stay on GPU
to avoid CPU softmax thread overhead and cover every token in selected rollouts.

There are four independent SFT examples per optimizer update. RL backward is
already scaled by TRL for 32 accumulated completions. Each SFT backward is
scaled by lambda/4, with Accelerator accumulation=1 checked explicitly. There
is one optimizer/scheduler and one gradient clip per combined update.

Anchor stream: T1/T4/T5/T7 each30 rows, T6/T8/T9 each20, T2/T3/T10/T11 each5
in a 200-row block. Polarity is balanced, with the odd row alternating across
blocks; missing polarity is backfilled and pool counts recorded. No images are
copied. RL retains the previous 100-query task quota and natural polarity.

## Protocol and evaluation

geometry_protocol.parse_answer has an explicit obb_validation argument. Strict
remains the legacy default. geometric_equivalent accepts only exact rectangle
equivalences: width/height swap with90-degree rotation and180-degree angle wrap.
Raw quantized values and trajectory IDs are never rewritten. Training GT stays
canonical. Wrong type/class, malformed/duplicate output and missing EOS fail.

V2 reward, Worker parsing and evaluation use geometric_equivalent, while eval
also reports strict metrics from the same raw answers. Native Worker precision
is used by temporarily unwrapping the Accelerator forward during evaluation.
Frozen quick550/full3330 query lists and Slow84 diagnostics remain fixed.

## Recovery and operation

Safe JSON+safetensors state stores adapters, optimizer, scheduler, RNG, TRL
microstep/buffer, query IDs, context checksums, generation count, anchor stream
cursor and experiment signature. Frozen 3B weights are referenced, not copied.
Incomplete or incompatible checkpoints fail closed. Deployment reload uses the
original SFT model with the saved policy adapter.

Production output must be a new directory under /root/rl_runs, not /data.
Code deployment uses Git only. Preflight includes protocol/identity/gradient
checks, exact continuous4 vs1+resume4, native evaluation equivalence and20
production updates per arm. Only after all gates pass may formal training start.
Budget is shared, rounded to100 RL queries/50 updates, with a12-hour training
and monitoring cap per arm; setup and final full evaluation are separate.
Only time/throughput can extend both arms together. Reward cannot extend runs.

Launch from the clean isolated checkout with:

```bash
RL_OUTPUT_ROOT=/root/rl_runs/<new_run> RL_HOURS=12 bash shell/rl-grpo-obb-anchor-pair.sh
```

The supervisor performs smoke4, smoke1+resume4, exact state comparison, GPU
identity/gradient/native-eval checks, fresh baseline evaluations, benchmark20,
then resumes those20 updates into the formal stage. The initial common target
is rounded to50 updates, includes the20 benchmark updates, and must cover at
least400 RL queries. Formal training/monitoring is capped at12 hours from the
formal-stage launch. A high Trainer loop ceiling permits common time-based
extension; budget.json, not the Trainer loop ceiling, controls stopping.
Final full evaluation is outside that cap. A protected paired checkpoint
supports comparisons at a common update count if either arm stops early.

Every50 updates retain latest3, first effective and best qualifying checkpoint.
Every approximately2 hours run Fast550 + Slow84. Original nonfinite, reference,
probability and consecutive metric-regression guards remain in force.

## Throughput investigation

`--phase profile --eval-checkpoint <adapter-checkpoint>` compares eight native
Slow trajectories per query with the existing generate API's projected visual
feature input. It checks raw tokens, CUDA RNG and sampling/training probabilities
on OBB, conditional OBB, Point and negative queries. It also profiles the actual
sequential differentiable cached probability/backward path. A final-state-only
output-head variant is measured as a probe; it is not enabled in training.
GPU utilization is secondary to completed query throughput. Performance changes
must pass these checks and apply to both arms before deployment.

V2 Slow rollouts now reuse only the identity-bound current-batch projected visual
features through the existing generate interface. No PBD/model changes, output
head restriction, precision change, sampling change or probability approximation
is enabled. This deterministic cache does not change the mathematical experiment
signature. Deployment still records its Git SHA and sampling_context_cache flag.
An explicit `--start-stage formal` restart additionally requires both performance
reports and exact original-continuous4 vs new-code-resume1-to4 state equality.
It reuses the completed production20 benchmark as a conservative throughput bound;
the shared budget can subsequently extend from observed actual formal throughput.
End-of-run throughput is corrected using actual updates, because HF's nominal
Trainer loop ceiling overstates its default throughput after callback early stop.
