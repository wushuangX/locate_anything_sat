# RL Phase B：GRPO on HBB 100k（Run A 全池 / Run B 密集团）

Recorded 2026-09-22. 两次 GRPO 均以 **#4 HBB geom 100k** 为初始策略；奖励 `scripts/rl/reward.py` v2.1（Phase A 冻结门已放行：`f1_anchor` ρ=0.704）。Index: [EXPERIMENTS.md](EXPERIMENTS.md)。

## Setup（两次共用）

- 机器：AutoDL 2×4090，实际**单卡 GPU0**，无 DeepSpeed / 无 KL / 无 packing
- 循环：`scripts/rl/grpo_loop.py`（NTP 采样 → `r_main` 组内标准化优势 → 仅 LLM LoRA 的 policy gradient，`lr=1e-6`，grad clip 1.0）
- 采样：`rollout.RL_GENERATION_KW`（slow / top_p=1 / top_k=0 / rep=1，temperature=1.0）
- logprob：`language_model(input_ids, visual_features)`，eval 模式 + clone 注入（100k Qwen2 的 `self.training` 分支不可用）
- ckpt：`save_pretrained` 全量 ~7.5G/份，滚动留 3

| | Run A `rl_grpo_hbb_100k` | Run B `rl_grpo_hbb_dense_g16_2k5` |
|---|---|---|
| 训练 tile | 全池 dedupe **15542** | **smallGT≥10 过滤 → 712** |
| G | 8 | 16 |
| 步数 | 500（完） | 计划 2500，**watcher 于 1000 终止** |
| 墙钟 | ~2.3 h | ~22.5 h（1000 步） |
| skip(zero_adv) | 84/500（16.8%，其中 75 次 mean_r=1.0 空/none 组） | 74/1000（7.4%） |
| train `mean_r` 走势 | 平（0.80→0.86 波动，Spearman(step)=0.05） | 开局 0.006/parse_ok 0.06 → 末段 mean_r 0.1–0.8、parse_ok 常 1.0 |

## Val 协议

1. **随机 50-tile**（`eval_baseline_dota.py` + `eval_dota_map.py`）：v1 `test_t1` 随机 50，15 类穷举，hybrid，`max_new_tokens=512`，greedy。50 张中仅 5 张 smallGT≥10 → small 桶方差极大。
2. **密集团协议**（新脚本 `scripts/eval_dota_small_dense.py`）：`test_t1` 中 **smallGT≥10 的 517 张**池，**每 seed 抽 50**（seed 42/43/44），**只问该 tile GT 内的类**，`max_new_tokens=2048`，greedy。报告 mean±pstdev（3 子集）。

## Results（密集团协议，3 seed）

| | 100k 基线 | RunA grpo500 | RunB d250 | RunB d500 | RunB d750 | RunB d1000 |
|---|---:|---:|---:|---:|---:|---:|
| F1 | 0.399±0.038 | 0.401±0.044 | 0.335±0.047 | 0.340±0.037 | 0.333±0.026 | 0.352±0.013 |
| AP50 | 0.402±0.074 | 0.427±0.061 | 0.360±0.052 | 0.378±0.055 | 0.355±0.050 | 0.375±0.052 |
| **small AP50** | **0.275±0.062** | 0.276±0.051 | 0.207±0.024 | 0.192±0.040 | 0.182±0.029 | 0.197±0.026 |
| SV AP50 | 0.160±0.017 | 0.182±0.012 | 0.131±0.052 | 0.146±0.022 | 0.147±0.045 | 0.168±0.027 |

随机 50-tile（协议 1）：100k vs RunA grpo500 — F1 0.419 vs 0.422、AP50 0.346 vs 0.351、small AP50 0.349 vs 0.197（伪信号，密集团仅 5 张所致，协议 2 中消失）。

**配对检验**（同 tile 子集跨模型差）：

- d1000 − d750：small **+0.015±0.014**（3/3 子集为正）、AP50 +0.020±0.011（3/3 正）→ 轻微回升，幅度在子集噪声边缘
- d1000 − 基线：small **−0.078±0.046**（3/3 负）、F1 −0.047±0.029（3/3 负）→ 净伤害确凿

## Reading

1. **Run A 无信号**：100k 对 `r_main` 已饱和，16.8% 组零方差，加长同一配方无意义。
2. **Run B 有信号但方向为负**：train `mean_r`↑ 的同时密集团 small AP50 单调下滑（0.275→0.182@750，1000 微弹）——奖励与评测协议错位（reward hacking 类）。可疑机制：训练 prompt 取 train_mix 首行（多为 T1 多类 `</c>` 句式）而评测为单类 detect；奖励 `n_raw>20→r=0` 与密集团 GT≤30 冲突；奖励无面积加权。
3. **评测确定性**：greedy（temperature=0）解码同图同 ckpt 重跑不变；方差来自 **tile 子集抽样**（seed），已 3 子集平均。收紧区间应加 seed 或跑全 517，而非同图重复解码；温度>0 多次采样平均是另一协议（评策略期望，非 greedy 行为）。

## Full-517 终判（2026-09-23 补，同协议单 seed=42，517 张全池）

| | 100k 基线 | d250 | d500 | d750 | d1000 | d1250（resume250） | d1500（resume500） |
|---|---:|---:|---:|---:|---:|---:|---:|
| F1 | **0.404** | 0.373 | 0.362 | 0.347 | 0.352 | 0.359 | 0.356 |
| AP50 | 0.241 | 0.197 | 0.191 | 0.200 | 0.200 | 0.238 | **0.314** |
| small AP50 | **0.098** | 0.092 | 0.081 | 0.076 | 0.082 | 0.077 | 0.076 |
| SV AP50 | **0.154** | 0.146 | 0.139 | 0.132 | 0.139 | 0.139 | 0.137 |
| AP | 0.101 | 0.094 | 0.098 | 0.093 | 0.103 | 0.107 | **0.177** |

resume 后总指标持续回升，d1500（全局 1500）AP50/AP 已**越过基线**（0.314/0.177 vs 0.241/0.101），但 F1 仍低于基线、**small AP50 全程横带 0.076–0.082 始终够不到基线 0.098**——涨的是发射序排序质量，不是 RL 目标的密集团小目标。
**终判：现有奖励/协议下，dense GRPO 对其目标项（small）为净伤害；总 AP 靠排序改善在 1500 步后反超。**（score=1.0 AP 协议下全量绝对值系统性低于 50 张子集，协议效应，只做同协议内比较。）


续训 `work_dirs/rl_grpo_hbb_dense_g16_2k5_resume`（从 ckpt-1000 起，本地 1500 步 ≈ 全局 2500，watcher 归档 `snapshots/checkpoint-resume*`）：495 步时 skip 率 0.8%、parse_ok 常 1.0、train reward 仍高——奖励继续被吃分，与终判趋势一致。

## Follow-ups（未做）

- 修奖励/协议后再重启 RL：单类句式对齐 eval、按 GT 上限放行框数或去掉 20 框硬门、面积加权 `r_main`
- 密集团全量 517 评测（~2.5 h/ckpt）作为最终判决协议
- Run B ckpt 留存 `work_dirs/rl_grpo_hbb_dense_g16_2k5/snapshots/checkpoint-{250,500,750,1000}` 作反例
