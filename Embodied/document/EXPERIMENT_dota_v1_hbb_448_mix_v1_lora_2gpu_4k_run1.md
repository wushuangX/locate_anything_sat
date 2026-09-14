# Experiment: DOTA-v1.0 HBB mix_v1 LoRA (2×4090 4k run1)

Recorded from artifacts 2026-09-14. Training finished **2026-09-09 21:42** (server local). Numbers are from this run only; they are not COCO mAP.

Dataset **v1**. Index: [EXPERIMENTS.md](EXPERIMENTS.md).

## 1. Setup

| Item | Value |
|---|---|
| Host | AutoDL `39.104.160.156:20079`，2× RTX 4090 24GB |
| Base | `/data/LocateAnything-3B` (`nvidia/LocateAnything-3B`, bf16) |
| Code | After `992ec6e`（task-mix 转换器）/ `8da1e0f`（launcher 指向 mix_v1）；**无 LDA**（`config.json` 中 `use_lda` 缺省） |
| Data | **v1** `data/dota_v1_hbb_448_mix_v1/`：复用 v0 tiles，task-mix T1–T5，train 源图 9:1 内部验证 |
| Recipe | `data/dota_v1_hbb_448_mix_v1/recipes/dota_v1_hbb_448_mix_v1_train_only.json` |
| Train JSONL | `DOTA-v1.0_train_mix_hbb_448.jsonl` **97242** 行（1270 源图）。内部 val 9951 行不进 `--meta_path` |
| Tile 像素 | 与 v0 相同：448×448、overlap 0、`tiles` → `data/dota_v1_hbb_448/tiles` |
| Output | `work_dirs/dota_v1_hbb_448_mix_v1_lora_2gpu_4k_run1/` |

相对 **#1 / v0** 的数据变化：每非空 tile 从「一条全类检测」变为加权混合（T1 全类 0.50、T2 单类 0.15、T3 类子集 0.10、T4 纯负样本 `<box>none</box>` 0.15、T5 指代 0.10），`--samples-per-tile 6`。训练行数 18201 → 97242（约 5.3×），像素不变。

### Training hyperparameters (what actually ran)

| | |
|---|---|
| Task | mix_v1 对话（T1–T5）；类别名仍是 DOTA 原名 |
| Adaptation | freeze LLM + MoonViT，train MLP，LLM LoRA r=64（`lora_alpha=128`） |
| LDA | 关 |
| Merge | default 2×2 |
| Attention | LLM `sdpa`；此日后 MoonViT 可用 flash-attn |
| Sequence | `max_seq_length=max_num_tokens=max_num_tokens_per_sample=4096`（相对 #1 的 2048） |
| Packing buffer | 32 |
| Optimizer | DeepSpeed ZeRO-1 AdamW，bf16，grad checkpoint |
| LR | 1.5e-5 cosine，warmup **100** |
| Batch | per-device 1，`GRADIENT_ACC=2` |
| Steps | 5000，save every 250，`SAVE_TOTAL_LIMIT=20` |
| Log | tensorboard / `training_log.txt`（not W&B） |

## 2. Training

| | |
|---|---|
| Wall clock | 4h 15m 56s（`train_runtime` 15355.9 s） |
| Final step | 5000 / 5000 |
| Last ckpt | `checkpoint-5000` |
| `done.txt` | written 2026-09-09 21:42 |

Loss（`trainer_state.json`）：first 3.563 → mean 1.091 → last 0.872；末 100 步 **0.974**，末 1000 步 0.985。相对 #1（mean 1.181 / 末 100 1.100），同为 5k 步但对话更杂、packing 更长，终值更低。

## 3. Val protocol

`scripts/eval_baseline_dota.py`，50 张 tile，seed 42：

- IoU 0.5，类内贪心匹配
- 每类单独 `detect`（15 个 DOTA-v1.0 名），与训练 T1/T2 句式一致
- `max_new_tokens=512`，`generation_mode=hybrid`
- 标注：`data/dota_v1_hbb_448_mix_v1/annotations/DOTA-v1.0_test_t1_hbb_448.jsonl`（官方 val 的纯 T1 渲染，5424 行；本次只抽 50）

报告：`data/dota_v1_hbb_448_mix_v1/_eval_results/mix4k_run1_checkpoint-5000_50.json`

该 JSON 的 `per_class` 键混有解析噪声（如 `small</box><ref>small-vehicle`），下表只用 **overall**。

## 4. Results (50 tiles, IoU=0.5)

| Model | Pred | none | TP | FP | FN | P | R | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base LocateAnything-3B（#1 同协议，v0 val JSONL） | 426 | 324 | 18 | 408 | 268 | 0.042 | 0.063 | 0.051 |
| #1 LoRA ckpt-5000（v0，全类-only 数据） | 1692 | 0 | 180 | 1512 | 106 | 0.106 | 0.629 | 0.182 |
| **#2 mix LoRA ckpt-5000** | 690 | 526 | 148 | 542 | 138 | **0.214** | 0.517 | **0.303** |

相对 #1：召回下降（0.629 → 0.517），精度约翻倍（0.106 → 0.214），F1 0.182 → 0.303。`none` 从 0 回到 526 / 50×15 次查询——T4 负样本开始起作用，模型不再对空类几乎必出框。

## 5. Reading

- Task-mix 的直接效应是 **少喷框、会说 none**，不是单纯把 #1 的召回再拉高。
- 50-tile 子集与 #1 是否同一批文件名未核验；方向（P↑ R↓ F1↑）可信，点估计不要当严格 A/B。
- 无面积分桶、无全量 val、无 LDA。

## 6. Follow-ups (not done here)

1. 同协议加 LDA、加长步数 → 见 #3。
2. 全量 `test_t1` AP → #3 才做。
3. 几何增强 → #4。
