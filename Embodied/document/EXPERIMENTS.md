# DOTA HBB LoRA 实验索引

本目录记录 **LocateAnything-3B → DOTA-v1.0 HBB 448** 的已完成微调。数字均来自对应 `work_dirs/*/train_results.json`、`trainer_state.json` 与 `_eval_results/*.json`，**不是**上游 `RESULTS.md` 里的论文表。

机器（四次共用）：AutoDL `root@39.104.160.156:20079`，2× RTX 4090 24GB，工作根 `/data/locate_anything_sat/Embodied`（与 `/root/autodl-tmp/locate_anything_sat` 同一数据盘）。日志 tensorboard / `training_log.txt`，**无 W&B**。

## 数据集版本

像素 tile **只切过一次**。后续版本复用 `data/dota_v1_hbb_448/tiles`（`--reuse-tiles-from`），改的是 JSONL 任务构成与几何视图。

| 版本 | 目录 | 对话构成 | 训练 JSONL | 评测 JSONL | 使用的 run |
|---|---|---|---|---|---|
| **v0** | `data/dota_v1_hbb_448/` | 每非空 tile 一条全类检测（超 30 框则拆样本） | `DOTA-v1.0_train_hbb_448.jsonl` **18201** 行（1411 源图，98990 源目标） | 官方 val 全类 `DOTA-v1.0_val_hbb_448.jsonl` **5424** 行 | #1 |
| **v1** | `data/dota_v1_hbb_448_mix_v1/` | 默认 task-mix：每 tile 抽 6 条，T1 0.50 / T2 0.15 / T3 0.10 / T4 0.15 / T5 0.10；train 源图 9:1 内部验证 | `DOTA-v1.0_train_mix_hbb_448.jsonl` **97242** 行（1270 源图）；内部 val **9951** 行（141 源图，训练不用） | 官方 val **双渲染**：`test_t1` **5424**（纯 T1，给 `eval_baseline_dota.py`）+ `test_mix` **31959** | #2、#3 |
| **v1-geom** | 同 v1 目录，另四份几何 JSONL | v1 的 0° mix + 离线 `rot90/180/270/hflip`（T5 指代表达丢弃）+ recipe `color_jitter` | 0° 97242 + 四向各 **87900**（共 **448842** 行；像素仍是同一套 tiles） | 评测仍只用 0° `test_t1` | #4 |

转换器与 recipe 字段见 [DATA_PREPARATION.md](DATA_PREPARATION.md)（DOTA HBB / 几何 JSONL 两节）。v1 metadata：`data/dota_v1_hbb_448_mix_v1/metadata/dota_v1_hbb_448_mix_v1_stats.json`。

## 四次正式训练

| # | 文档 | 输出目录 | 数据 | 适配 | 步数 | 墙钟 | 全程 mean loss | 末 100 步 loss |
|---|---|---|---|---|---:|---:|---:|---:|
| 1 | [EXPERIMENT_dota_v1_hbb_448_2gpu_run1.md](EXPERIMENT_dota_v1_hbb_448_2gpu_run1.md) | `work_dirs/dota_v1_hbb_448_lora_2gpu_run1/` | v0 | LoRA r=64 + 解冻 MLP | 5000 | 3.2 h | 1.181 | 1.100 |
| 2 | [EXPERIMENT_dota_v1_hbb_448_mix_v1_lora_2gpu_4k_run1.md](EXPERIMENT_dota_v1_hbb_448_mix_v1_lora_2gpu_4k_run1.md) | `work_dirs/dota_v1_hbb_448_mix_v1_lora_2gpu_4k_run1/` | v1 | 同 #1，packing 4096 | 5000 | 4.3 h | 1.091 | 0.974 |
| 3 | [EXPERIMENT_dota_v1_hbb_448_mix_v1_lora_lda_2gpu_4k_15k_run1.md](EXPERIMENT_dota_v1_hbb_448_mix_v1_lora_lda_2gpu_4k_15k_run1.md) | `work_dirs/dota_v1_hbb_448_mix_v1_lora_lda_2gpu_4k_15k_run1/` | v1 | #2 + LDA（γ 初值 0，bottleneck 128） | 15000 | 13.3 h | 0.834 | 0.599 |
| 4 | [EXPERIMENT_dota_geom_lora_lda_2gpu_4k_100k_run1.md](EXPERIMENT_dota_geom_lora_lda_2gpu_4k_100k_run1.md) | `work_dirs/dota_geom_lora_lda_2gpu_4k_100k_run1/` | v1-geom | 同 #3，几何五 key + color_jitter | 100000 | 88.6 h | 0.691 | 0.436 |

共同超参（四次）：基座 `nvidia/LocateAnything-3B`（本地 `/data/LocateAnything-3B` 或 `/root/autodl-tmp/LocateAnything-3B`），freeze LLM+MoonViT、train MLP、LLM LoRA r=64 α=128、DeepSpeed ZeRO-1、bf16、cosine LR **1.5e-5**、grad acc 2、per-device batch 1、`attn_implementation=sdpa`、默认 2×2 merge、**无** backbone LoRA。#1 packing **2048** / buffer 1；#2–#4 packing **4096** / buffer 32。

### 50-tile 协议对照（IoU=0.5，15 类穷举 `detect`，seed 42）

同一套 `eval_baseline_dota.py` 微平均 P/R/F1。#1 扫的是 v0 val JSONL；#2/#3 扫的是 v1 `test_t1`（tiles 复用自 v0）。**未核验 50 张文件名是否逐张相同**，只比协议、不把 ΔF1 当严格 A/B。

| 模型 | Pred | none | TP | FP | FN | P | R | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base LocateAnything-3B | 426 | 324 | 18 | 408 | 268 | 0.042 | 0.063 | 0.051 |
| #1 ckpt-5000 | 1692 | 0 | 180 | 1512 | 106 | 0.106 | 0.629 | 0.182 |
| #2 ckpt-5000 | 690 | 526 | 148 | 542 | 138 | 0.214 | 0.517 | 0.303 |
| #3 ckpt-15000 | 611 | 543 | 156 | 455 | 130 | 0.255 | 0.545 | 0.348 |
| #4 最终权重 | — | — | — | — | — | — | — | **未评** |

#3 另有 COCO 风格 AP（框 score 全 1.0，排序=发射序）：50 tile AP50=0.327；全量 5424 张 `test_t1` AP=0.136 / AP50=0.250（small AP50=0.063）。

## 不算正式实验

| 目录 | 步数 | 说明 |
|---|---:|---|
| `dota_v1_hbb_448_2gpu_smoke10` | 10 | 通路 smoke |
| `dota_v1_hbb_448_mix_v1_lora_2gpu_smoke4k` | 10 | 4k packing smoke |
| `dota_v1_hbb_448_mix_v1_lora_lda_2gpu_4k_smoke` | 10 | LDA smoke |
| `dota_v1_hbb_448_mix_v1_lora_lda_2gpu_4k_run1` | 0 | 只有 launcher log，无权重（#3 的流产目录） |

`scripts/run_dota_v1_hbb_448_7gpu_train_with_val.sh` 与 `shell/train-dota-obb-lora.sh` **从未跑满**。

## RL（GRPO）实验

基于 #4 最终权重的 on-policy GRPO（单卡、无 KL）。详见 [EXPERIMENT_rl_grpo_hbb.md](EXPERIMENT_rl_grpo_hbb.md)。

| run | 目录 | 数据 | G | 步数 | 结论 |
|---|---|---|---:|---:|---|
| A | `work_dirs/rl_grpo_hbb_100k/` | 全池 15542 tile | 8 | 500 | 无信号：奖励饱和，skip 16.8%，密集团指标与 #4 持平 |
| B | `work_dirs/rl_grpo_hbb_dense_g16_2k5/` | smallGT≥10 的 712 tile | 16 | 1000（原计划 2500，watcher 终止） | **负迁移**：train reward↑ 但密集团 small AP50 0.275→0.197（配对 3/3 负）；奖励/协议错位，ckpt 作废 |

## 新 run 怎么记

复制本目录 `EXPERIMENT_*.md` 模板：Setup（host / 数据版本 / recipe / 超参）→ Training（墙钟、loss）→ Val protocol（文件路径）→ Results（表）→ Reading → Follow-ups。并在上表加一行。不要把 loss 或 50-tile F1 写成 COCO mAP。
