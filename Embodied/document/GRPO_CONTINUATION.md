# Pure GRPO200 to400 on GPU0

The previous pair stopped after the anchor RL probability forward ran out of
memory on a batch with33-token trajectories. The complete200 policy/reference,
optimizer, scheduler, RNG and generation state are the continuation origin.
Unsaved201-246 pure updates are recomputed; they are not recoverable weights.
GPU1 is reserved for the user's other jobs and is never launched by this entry.

The mathematical profile remains four queries, eight trajectories, two iterations,
beta0.02, LR5e-7, group scaling and the same rewards/data/reference. Answers remain
capped at128 tokens. For differentiable trajectories of24 tokens or longer,
non-reentrant activation checkpointing recomputes each cached AR step. All prefix
KV gradients remain attached. Recompute restores eval mode, adapter identity and
visual injection, even after compute_loss restored LM training mode. No model
core, PBD mask, output-head projection or sampling approximation is changed.

GPU gates compare probabilities, entropy and all policy gradients against the
original12/16-token path;33/128-token forced continuations check original no-grad
probabilities, finite gradients and memory headroom. These stress trajectories
are diagnostics only, not extra training data. The standalone entry fails closed
if these gates fail. expandable_segments reduces allocator fragmentation; it does
not replace activation recomputation.

Resume compatibility uses the same mathematical signature. Checkpoint stores the
new Git SHA and runtime memory configuration. A fixed400-step target allows200
additional updates (400 additional RL queries). Twelve-hour cap is a new maximum
for training/monitoring; final complete evaluation is additional. No automatic
budget extension. Save every25 updates; keep latest3 and protected candidates.
Fast550 and Slow84 monitor at100-step boundaries or about two hours; final
Fast3330 and counterfactual diagnostics retain the same protocol.

Run from a clean Git-deployed independent checkout with isolated TRL PYTHONPATH:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/rl/continue_multitask_grpo.py \
  --root /root/rl_runs/obb_grpo_continue_20261014
```

A data-only `runs/grpo/PAUSE_REQUEST.json` requests checkpoint/save/exit at the
next optimizer boundary. During pause, final GPU evaluation is skipped; model,
optimizer, RNG and pending old-probability cache are saved. Resume requires removing
the request and supplying that complete checkpoint; no SIGSTOP memory hold is used.
Original pair outputs and GPU1 training are not modified.
