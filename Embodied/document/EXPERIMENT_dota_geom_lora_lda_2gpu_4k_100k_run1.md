# Experiment: DOTA-v1.0 HBB mix_v1 几何增强 LoRA+LDA (2×4090 4k 100k)

Recorded from artifacts 2026-09-14. Training finished **2026-09-14 14:33** (server local). **本 run 收官时尚未做 val / AP。** 下表只有训练曲线，没有检测指标。

Dataset **v1-geom**。Index: [EXPERIMENTS.md](EXPERIMENTS.md).

## 1. Setup

| Item | Value |
|---|---|
| Host | AutoDL `39.104.160.156:20079`，2× RTX 4090 24GB |
| Base | `/root/autodl-tmp/LocateAnything-3B`（与 `/data/LocateAnything-3B` 同一份基座） |
| Code | `3916ad7`（`shell/train-dota-geom-lora.sh`）+ `7eaf3bc`（几何 JSONL / `rotate`/`hflip`/`color_jitter`） |
| Data | **v1-geom**：v1 的 0° mix + 四份离线几何 JSONL，loader 按 recipe 做 PIL 变换 + color_jitter |
| Recipe | `data/dota_v1_hbb_448_mix_v1/recipes/dota_v1_hbb_448_mix_v1_geom_train_only.json`（五 key） |
| Train JSONL | 0° **97242**；`rot90`/`rot180`/`rot270`/`hflip` 各 **87900**（T5 指代已丢）；合计 **448842** 行。像素仍是 v0 tiles |
| Output | `work_dirs/dota_geom_lora_lda_2gpu_4k_100k_run1/` |
| 最终权重 | run 根目录 `model-00001-of-00002.safetensors` + `model-00002-of-00002.safetensors`；滚动 ckpt `SAVE_TOTAL_LIMIT=3`；`milestones/` 另留整步 checkpoint |

相对 **#3** 的数据变化：不新增地理场景，只覆盖同一 448 tile 的 D4 取向 + 光度抖动。几何文件 **不复制像素**，只改 `[0,1000]` 的 `<box>` token；T5（`the leftmost ...` 等）自动丢弃，否则位置词与旋转冲突。五 key 等权采样时，固定 `max_steps` 下每个取向大约分到 1/5 的 step。

### Training hyperparameters (what actually ran)

| | |
|---|---|
| Task | mix_v1 T1–T4（几何文件无 T5）+ 0° key 仍含 T5 |
| Adaptation | 同 #3：freeze LLM+MoonViT，train MLP+LDA，LLM LoRA r=64 |
| LDA | `--use_lda true`，bottleneck 128 |
| Merge | default 2×2 |
| Attention | LLM `sdpa` |
| Sequence | 4096 / packing buffer 32 |
| Optimizer | DeepSpeed ZeRO-1 AdamW，bf16，grad checkpoint |
| LR | 1.5e-5 cosine，warmup **500** |
| Batch | per-device 1，grad acc 2 |
| Steps | **100000**，save every 500，滚动保留 3 个；milestone 另存 |
| Launcher | `bash shell/train-dota-geom-lora.sh` |
| Log | tensorboard / `training_log.txt`（not W&B；脚本里的 `WANDB_*` 未实际上报） |

## 2. Training

| | |
|---|---|
| Wall clock | 88.6 h（`train_runtime` 319057.9 s ≈ 3.7 天） |
| Final step | 100000 / 100000 |
| `done.txt` | written 2026-09-14 14:33 |
| `train_loss`（全程均值） | 0.691 |
| 末 100 步 / 末 1000 步 | **0.4355** / **0.4339** |
| first / last logged | 3.360 / 0.511 |
| HF `epoch` 字段 | 1.0（`max_steps` 训练，不是数据 epoch） |

Loss 形态（监控记录，非逐点表）：前 500 步陡降；之后到约 20k 近似线性慢降；70k 以后斜率约为中段的 1/3，末段落在 0.43–0.44。全程无发散。

磁盘：收官后 `/root/autodl-tmp` 约 93%。`milestones/` 曾含非整 10k 的 2k 间隔 ckpt；清理脚本 `work_dirs/prune_milestones.sh`（只留 step%10000==0）。

## 3. Val protocol

**未跑。** 计划协议（与 #3 可比）应是：

- 0° `annotations/DOTA-v1.0_test_t1_hbb_448.jsonl`（不要用 mix 文件，不要用几何 JSONL）
- `scripts/eval_baseline_dota.py` 50-tile seed 42 F1
- 全量 5424 `test_t1` AP（`eval_dota_map` 一类脚本，score=1.0 限制照旧）
- 候选权重：run 根目录最终模型，以及 `milestones/checkpoint-{10000,20000,…,100000}`

## 4. Results

无检测数字。训练侧对照：

| Run | 数据 | 步数 | mean loss | 末 100 loss |
|---|---|---:|---:|---:|
| #3 LoRA+LDA | v1 0° mix | 15000 | 0.834 | 0.599 |
| **#4 LoRA+LDA geom** | v1-geom | 100000 | 0.691 | **0.436** |

不能从 loss 推断 AP。#3 全量 AP50 才 0.25，本 run 是否过拟合 / 是否 30k 已够，只能靠 milestone 扫评。

## 5. Reading

- 几何增强的价值是 **取向覆盖**（尤其以后 OBB），不是「5 倍新场景」。
- 10 万步是否过长是开放问题：滚动 ckpt 只留 3 个，判断依赖 `milestones/`。
- LDA γ 终值未从权重读出；γ=0 消融未做。
- 评测/Streamlit 必须继续用 0° `test_t1`，不要把 `rotate` recipe 用到 val。

## 6. Follow-ups (not done here)

1. Milestone 扫评：整 10k ckpt × 0° `test_t1` AP/F1 曲线，选最佳步。
2. 与 #3 `checkpoint-15000` 同 50-tile / 全量 val 对比（几何 + 6.6× 步数的净增益）。
3. LDA γ=0 推理消融。
4. 磁盘：按需跑 `prune_milestones.sh`。
