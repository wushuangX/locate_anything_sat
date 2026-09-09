# Experiment: DOTA-v1.0 HBB 448 LoRA (2×4090 run1)

Recorded 2026-09-09. Numbers below are from this run only; they are not COCO mAP.

## 1. Setup

| Item | Value |
|---|---|
| Host | AutoDL `39.104.160.156:20079` (`autodl-container-59914c9002`) |
| GPU | 2× RTX 4090 24GB |
| Base | `/data/LocateAnything-3B` (`nvidia/LocateAnything-3B`, bf16) |
| Code | `2e74a0e` (MoonViT sdpa fallback) + 2048 packing override |
| Data | DOTA-v1.0 → 448×448 HBB tiles, `dota_v1_hbb_448_train_only.json` |
| Train JSONL | 18201 samples (1411 images, 98990 source objects) |
| Val JSONL | 5424 samples (458 images); this eval used **50 tiles**, seed 42 |
| Output | `work_dirs/dota_v1_hbb_448_lora_2gpu_run1/` |

### Training hyperparameters (what actually ran)

| | |
|---|---|
| Task | DOTA HBB detection dialogue, class names unchanged, `</c>` over classes in the tile |
| Adaptation | freeze LLM + MoonViT, train MLP, LLM LoRA r=64 (`lora_alpha=128`) |
| Merge | default 2×2 (`VISION_MERGE_KERNEL_SIZE` unset) |
| Attention | LLM `sdpa`; vision fa2 if `flash_attn` present else sdpa |
| Sequence | `max_seq_length=max_num_tokens=max_num_tokens_per_sample=2048` |
| Packing buffer | 1 |
| Optimizer | DeepSpeed ZeRO-1 AdamW, bf16, grad checkpoint |
| LR | 1.5e-5 cosine, warmup 100 |
| Batch | per-device 1, `GRADIENT_ACC=2` |
| Steps | 5000, save every 250, `SAVE_TOTAL_LIMIT=20` |
| Log | tensorboard / `training_log.txt` (not W&B) |

2048 packing is a **memory workaround**. Without flash-attn, packed MoonViT sdpa OOM at 4096 on 24GB. Upstream H100 Magi LoRA default packing is 16384, not 32K.

## 2. Training

| | |
|---|---|
| Wall clock | 3h 13m 56s (~2.30 s/it) |
| Final step | 5000 / 5000 |
| Traceback | 0 |
| Last ckpt | `checkpoint-5000` |
| `done.txt` | written |

Loss (logged): start ~2.8 → ~step 243 **1.21** → step 5000 **1.18**.

## 3. Val protocol

`scripts/eval_baseline_dota.py` on 50 val tiles:

- IoU 0.5, greedy class-wise match
- **Per-class `detect`** (15 DOTA-v1.0 names one at a time), matching training prompts
- `max_new_tokens=512`, `generation_mode=hybrid`, seed 42
- Not COCO mAP; no small/medium/large buckets
- Querying all 15 classes on every tile inflates FP when GT has none of that class

Reports:

- LoRA: `data/dota_v1_hbb_448/_eval_results/2gpu_run1_checkpoint-5000_50.json`
- Base: `data/dota_v1_hbb_448/_eval_results/base_LocateAnything-3B_50.json`

## 4. Results (50 tiles, IoU=0.5)

Overall:

| Model | Pred | none | TP | FP | FN | P | R | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base LocateAnything-3B | 426 | 324 | 18 | 408 | 268 | 0.042 | 0.063 | 0.051 |
| LoRA checkpoint-5000 | 1692 | 0 | 180 | 1512 | 106 | 0.106 | 0.629 | 0.182 |

Per-class recall (classes with GT>0):

| Class | GT | Base R | LoRA R | Base F1 | LoRA F1 |
|---|---:|---:|---:|---:|---:|
| small-vehicle | 87 | 0.034 | 0.598 | 0.047 | 0.237 |
| ship | 82 | 0.073 | 0.646 | 0.122 | 0.308 |
| harbor | 38 | 0.026 | 0.553 | 0.031 | 0.333 |
| storage-tank | 33 | 0.061 | 0.606 | 0.049 | 0.198 |
| plane | 24 | 0.125 | 0.792 | 0.146 | 0.264 |
| tennis-court | 7 | 0.000 | 0.429 | 0.000 | 0.066 |
| large-vehicle | 5 | 0.000 | 0.800 | 0.000 | 0.079 |
| basketball-court | 3 | 0.333 | 1.000 | 0.059 | 0.076 |
| ground-track-field | 2 | 0.500 | 1.000 | 0.038 | 0.071 |
| roundabout | 2 | 0.500 | 0.500 | 0.074 | 0.025 |
| soccer-ball-field | 2 | 0.000 | 0.500 | 0.000 | 0.038 |
| bridge | 1 | 0.000 | 1.000 | 0.000 | 0.032 |

GT=0 classes (baseball-diamond, helicopter, swimming-pool): base mostly `none`; LoRA still emits tens of boxes per class → protocol FP.

## 5. Reading

- Domain shift is real: frozen 3B on DOTA tiles barely detects (R=0.063, 324× `none`).
- LoRA + unfrozen MLP raises recall to **0.63** (~10×) on ship / vehicle / plane / tank.
- Precision stays low. LoRA is worse on FP because it almost never answers `none` under exhaustive 15-class prompts. This is not mAP.
- n=50 tiles; no area buckets; no SAHI full-image eval.

## 6. Follow-ups (not done here)

1. Same 50 tiles are enough for a first delta; full val JSONL (5424) still unrun.
2. Restrict eval classes to those present in the tile GT (cuts empty-class FP).
3. Report AP_small / AP_tiny.
4. After flash-attn 2.7.3 (compiled on this host during the run), a new run at packing 4096 is possible; this run stayed at 2048.
