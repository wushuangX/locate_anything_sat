# Experiment: DOTA-v1.0 HBB mix_v1 LoRA+LDA (2×4090 4k 15k run1)

Recorded from artifacts 2026-09-14. Training finished **2026-09-10 12:24** (server local). 50-tile F1 不是 COCO mAP；同 run 另有一套 score=1.0 的 AP 表。

Dataset **v1**（与 #2 同一 JSONL）。Index: [EXPERIMENTS.md](EXPERIMENTS.md).

## 1. Setup

| Item | Value |
|---|---|
| Host | AutoDL `39.104.160.156:20079`，2× RTX 4090 24GB |
| Base | `/data/LocateAnything-3B` |
| Code | `1465745`（config-gated residual LDA）之后。流产目录 `work_dirs/dota_v1_hbb_448_mix_v1_lora_lda_2gpu_4k_run1/` 仅有 log、无权重，正式产物在本目录 |
| Data | **v1** 与 #2 相同：`DOTA-v1.0_train_mix_hbb_448.jsonl` 97242 行 |
| Recipe | `recipes/dota_v1_hbb_448_mix_v1_train_only.json` |
| Output | `work_dirs/dota_v1_hbb_448_mix_v1_lora_lda_2gpu_4k_15k_run1/` |
| 最终权重 | `checkpoint-15000/`（run 根目录亦有 HF 保存） |

相对 **#2** 的唯一结构变化：开启 MoonViT **Local Detail Adapter**（patch merge 前残差，`F' = F + γ·ΔF`，γ 初始化 0；`lda_bottleneck_dim=128`）。`freeze_backbone=True` 时 LDA 仍解冻。数据不变。

### Training hyperparameters (what actually ran)

| | |
|---|---|
| Task | 同 #2（mix_v1 T1–T5） |
| Adaptation | freeze LLM + MoonViT，train MLP + **LDA**，LLM LoRA r=64 |
| LDA | `--use_lda true`，bottleneck 128（写入 `config.json` `vision_config`） |
| Merge | default 2×2 |
| Attention | LLM `sdpa` |
| Sequence | 4096 / packing buffer 32 |
| Optimizer | DeepSpeed ZeRO-1 AdamW，bf16，grad checkpoint |
| LR | 1.5e-5 cosine，warmup **300**（#2 为 100） |
| Batch | per-device 1，grad acc 2 |
| Steps | **15000**，save every 250，`SAVE_TOTAL_LIMIT=20` |
| Log | tensorboard / `training_log.txt`（not W&B） |

## 2. Training

| | |
|---|---|
| Wall clock | 13h 17m 4s（`train_runtime` 47824.1 s） |
| Final step | 15000 / 15000 |
| Last ckpt | `checkpoint-15000` |
| `done.txt` | written 2026-09-10 12:24 |

Loss：first 3.564 → mean 0.834 → last 0.658；末 100 步 **0.599**，末 1000 步 0.612。#2 在 5k 处末 100 已是 0.974，本 run 再跑到 15k 明显继续下降。

## 3. Val protocol

三套数字，不要混用：

1. **50-tile F1**（与 #1/#2 相同微平均协议）：`lda15k_run1_checkpoint-15000_50.json`
2. **50-tile AP**（框 score 全 1.0，排序=发射序）：`lda15k_checkpoint-15000_50_map.json`
3. **全量官方 val `test_t1` AP**（5424 图，35866 GT）：`lda15k_checkpoint-15000_test_t1_map.json`

另：8 条 prompt 在同一 50 tile、seed 42、15 类穷举上的扫描 → `lda15k_prompt_sweep_50.json`。

## 4. Results

### 4.1 50-tile F1（IoU=0.5）

| Model | Pred | none | TP | FP | FN | P | R | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| #2 mix LoRA 5k | 690 | 526 | 148 | 542 | 138 | 0.214 | 0.517 | 0.303 |
| **#3 LoRA+LDA 15k** | 611 | 543 | 156 | 455 | 130 | **0.255** | **0.545** | **0.348** |

P/R/F1 相对 #2 同向小幅提升；`none` 仍高（543），没有退回 #1 那种「从不 none」。

### 4.2 50-tile AP（score=1.0）

| 桶 | AP | AP50 | AP75 | n_classes_with_gt |
|---|---:|---:|---:|---:|
| all | 0.200 | 0.327 | 0.241 | 12 |
| small | 0.068 | 0.157 | 0.052 | 5 |
| medium | 0.224 | 0.341 | 0.303 | 9 |
| large | 0.400 | 0.583 | 0.377 | 7 |

### 4.3 全量 `test_t1` AP（5424 图，score=1.0）

| 桶 | AP | AP50 | AP75 | n_classes_with_gt |
|---|---:|---:|---:|---:|
| all | 0.136 | 0.250 | 0.123 | 15 |
| small | 0.026 | 0.063 | 0.016 | 15 |
| medium | 0.134 | 0.258 | 0.122 | 15 |
| large | 0.238 | 0.383 | 0.236 | 14 |

Per-class AP50（全量）：

| Class | npos | ndet | P | R | AP50 |
|---|---:|---:|---:|---:|---:|
| tennis-court | 1171 | 1354 | 0.683 | 0.790 | 0.611 |
| plane | 2649 | 2848 | 0.726 | 0.781 | 0.603 |
| swimming-pool | 506 | 619 | 0.436 | 0.534 | 0.348 |
| storage-tank | 3094 | 3009 | 0.534 | 0.520 | 0.309 |
| ground-track-field | 139 | 266 | 0.372 | 0.712 | 0.292 |
| baseball-diamond | 250 | 343 | 0.402 | 0.552 | 0.281 |
| large-vehicle | 6530 | 9527 | 0.392 | 0.572 | 0.282 |
| soccer-ball-field | 156 | 311 | 0.325 | 0.647 | 0.224 |
| ship | 10923 | 12480 | 0.417 | 0.477 | 0.215 |
| basketball-court | 149 | 245 | 0.216 | 0.356 | 0.176 |
| harbor | 2518 | 9748 | 0.141 | 0.545 | 0.171 |
| small-vehicle | 7013 | 12161 | 0.253 | 0.439 | 0.168 |
| helicopter | 77 | 347 | 0.121 | 0.545 | 0.067 |
| roundabout | 195 | 5344 | 0.017 | 0.472 | 0.010 |
| bridge | 496 | 6686 | 0.010 | 0.133 | 0.002 |

`bridge` / `roundabout` / `harbor` 仍是高召回低精度（场景尺度类 + T4 跳过这三类作负样本）。小目标桶 AP50=0.063 是后续几何/切片要打的数。

### 4.4 Prompt sweep（同一 50 tile，ckpt-15000）

训练原句（A）最好。语法改一个字母、换成 T5/`Detect`/裸类名都会掉。

| id | 句式 | P | R | F1 | pred | none |
|---|---|---:|---:|---:|---:|---:|
| **A** | `Locate all the instances that matches the following description: {class}.` | **0.255** | 0.545 | **0.348** | 611 | 543 |
| B | `... that match ...` | 0.240 | 0.483 | 0.321 | 575 | 541 |
| G | `Find every object of category {class} in this aerial image.` | 0.220 | 0.566 | 0.316 | 738 | 440 |
| D | `Locate all the {class}.` | 0.196 | 0.385 | 0.260 | 560 | 513 |
| C | T5 `Locate a single instance...` | 0.177 | 0.434 | 0.252 | 699 | 307 |
| H | 类名在 prompt 里重复 | 0.075 | 0.559 | 0.133 | 2128 | 145 |
| E | `Detect all {class}.` | 0.083 | 0.297 | 0.130 | 1018 | 198 |
| F | 仅 `{class}` | 0.067 | 0.455 | 0.116 | 1948 | 7 |

## 5. Reading

- 同数据、加 LDA、步数 5k→15k：50-tile F1 0.303 → 0.348。LDA 与加长步数 **未做消融拆开**（没有「15k 无 LDA」对照）。
- 全量 AP50=0.25、small=0.063：域适配有了，小目标仍弱。score 全 1.0 的 AP 会低估排序质量。
- Prompt 必须对齐训练 T1 句式（含原文语法错误 `matches`）。评测/Streamlit 默认应锁 A。
- Streamlit 默认 checkpoint 曾指向本 run 的 `checkpoint-15000`。

## 6. Follow-ups (not done here)

1. 几何五向 + color_jitter、10 万步 → #4。
2. LDA γ=0 推理消融（量化 LDA 净贡献）— **未做**。
3. 与 #4 同 50-tile / 全量 `test_t1` 对比 — #4 尚未评测。
