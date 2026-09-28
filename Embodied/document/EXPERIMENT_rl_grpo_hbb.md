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

resume 后总指标持续回升，d1500（全局 1500）AP50/AP **越过基线**（0.314/0.177 vs 0.241/0.101），但 F1 仍低于基线、**small AP50 全程横带 0.076–0.082 始终够不到基线 0.098**。
**终判：现有奖励/协议下，dense GRPO 对其目标项（small）为净伤害。**（score=1.0 AP 协议下全量绝对值系统性低于 50 张子集，协议效应，只做同协议内比较。）

### 事实更正（2026-09-23，v2 复盘确认）

1. **full-517 协议失真**：所谓 full-517 实为 **517 个 annotation chunk（test_t1 按 smallGT≥10 过滤出的行），仅覆盖 371 个唯一图像**——同图多 chunk 重复计入 GT/预测，且同一图像的 GT 被行切分稀释。v2 起改用 canonical unique-image 协议（`merge_t1_chunks` 合并去重后 385 张 dense 图像）。
2. **d1500 的 AP50 反超是单类宏平均假象**：由 `ground-track-field`（val 中 GT 极少，单 GT 类贡献 1/15 宏平均权重）的个别匹配拉动；**排除该类后 d1500 AP50 = 0.2572，仍低于基线 0.2607**，不存在真实总指标反超。
3. **"发射序排序质量提升"不再作为结论**：AP 排序改善与单类宏平均假象纠缠，且 F1/small 均未改善，不足以支撑该表述。


续训 `work_dirs/rl_grpo_hbb_dense_g16_2k5_resume`（从 ckpt-1000 起，本地 1500 步 ≈ 全局 2500，watcher 归档 `snapshots/checkpoint-resume*`）：495 步时 skip 率 0.8%、parse_ok 常 1.0、train reward 仍高——奖励继续被吃分，与终判趋势一致。

### Resume run 终止记录（2026-09-23）

- 终止时间：2026-09-23 18:42 UTC（SIGTERM 至进程组 83493；launcher 83495 干净退出，子进程零残留）。**应用户指示提前停止**，未等到 checkpoint-1000 边界（v2 watchdog 已取消，未发信号即手动终止）。
- 最终状态：止步 **step 857**；最后一个完整 checkpoint 为 `resume-750`（已快照 `snapshots/checkpoint-resume750`）。step 750→857 未保存，按用户决策弃用。
- 止步时训练指标：parse_ok 0.5–1.0 振荡、train reward 仍高位——与"奖励被吃分"终判一致，无新信息。
- v2 闭环（RL-single 数据 / reward v3 / 组内 gate / reference-KL GRPO / canonical eval）已落地，见下文 v2 各节。

## v2 闭环结果（2026-09-24）

**数据**：`data/dota_v1_hbb_448_rl_v2`（converter `--emit-rl-single-class`，复用 mix_v1 tiles）：train_rl_single **19154 行** + internal_val 2196 行，(image,prompt) 唯一、单类、每行完整 GT；max GT=169（≤256 上限），最长 answer 1188 token（≤2048）。recipe `_rl_single.json` 只含 train partition。

**奖励 v3**（`reward.py`）：一对一匈牙利匹配（label-aware IoU 最大化）+ small recall 加权，输出上限 20→256，无 NMS（重复框=FP）。selftest 全过：30/30=1.0、20/30 全 small≈0.733 严格小于完整输出、重复框/错类/截断严格降分。

**Freeze gate v2**（512 samples × G16，双卡 8192 rollouts，geom100k 冻结策略）——**四门全过**：

| 门 | 值 | 阈值 |
|---|---|---|
| pairwise concordance（组内宏平均） | **0.918** | ≥0.70 |
| median group Spearman | **0.937** | ≥0.60 |
| signal fraction（std≥0.02） | **0.988**（506/512 组） | ≥0.70 |
| timing ratio | **0.0002** | ≤0.25 |

诊断：全局 reward↔anchor Spearman **0.9897**（v2.1 门禁 ρ 仅 0.704）；parse_fail 14.1%（超密 tile 撞 2048 token 截断，reward −1 正确垫底）；mean n_pred 16.4；mean hard small recall 0.463。v3 奖励在组内排序上与评测锚点几乎完全一致。

**Smoke/resume**：fresh 首组 policy/reference token logprob diff **0.00e+00**（修复 fp32/bf16 dtype 陷阱后）；KL 有限非负；checkpoint-2 → resume 至 step 4：optimizer/order/cursor 连续无重放；checkpoint 可直接 `LocateAnythingWorker.detect()`，state 中无 `.reference.` key。

**Baseline canonical hybrid**（geom100k，385 unique dense 图，15 类穷举，max_new_tokens 2048）：all micro P/R/F1 = 0.553/0.379/**0.450**；small R=**0.362**、small F1=0.446；mean preds 25.6/图；label mismatch 仅 74。tied-score AP50 0.255 / smallAP50 0.132（仅诊断）。go/no-go 以 fixed_iou_05 为准。

## Pilot 250-step 终判（2026-09-25，`work_dirs/rl_grpo_hbb_dense_v2_pilot`）

设置：LR 5e-7、G16、KL β=0.02、loss 分母 512、min-reward-std 0.02、save 50；数据 RL-single（smallGT≥10 的 1089 样本池）；train n_skip=3/250。每个 checkpoint 由 watcher 自动跑 canonical hybrid eval 对比 baseline（regression→自动停训，未触发）。

| canonical hybrid（385 unique 图） | baseline | step50 | step100 | step150 | step200 | step250 |
|---|---:|---:|---:|---:|---:|---:|
| all micro F1 | 0.4495 | 0.4460 | 0.4517 | 0.4560 | **0.4577** | 0.4484 |
| small recall | 0.3622 | 0.3574 | 0.3637 | 0.3668 | **0.3734** | 0.3673 |
| small F1 | 0.4456 | 0.4429 | 0.4468 | 0.4458 | **0.4495** | 0.4428 |
| mean preds/图 | 25.6 | 25.4 | 25.7 | 26.5 | 27.0 | 26.8 |

**Slow 诊断**（同 385 图、NTP 解码；只诊断 decoder mismatch，不参与放行）：baseline all F1 0.4902 / small R 0.3986 → step200 all F1 **0.5025** / small R **0.4226**。slow 下改善更大（small R +0.0240 vs hybrid +0.0112），方向一致——PBD hybrid 解码不是瓶颈。

**Go/no-go（既定规则，step250 final-check）**：FAIL——small R 0.3673 ✓（+0.0051）、all F1 0.4484 ✓、mean preds ✓，但 **small F1 0.4428 < baseline 0.4456 ✗**。→ **不扩展 1000 步**。step200 用同一 final 规则为 **PASS**（4/4）。

**结论**：修复后的闭环（完整单类 GT + 一对一 small-recall 奖励 + reference-KL）首次实现 small recall 与 overall F1 同向改善（step200：hybrid small R +0.0112、all F1 +0.0082；slow small R +0.0240），且无少发框/少预警等 reward hacking 迹象（mean preds 上升、parse fail 正常）。150→200 单调改善、250 末回落属训练后期退化；**最佳产出 checkpoint = step200**。扩步决策需回到 reward gate 重新验证（不通过增加步数补救）。

## 稳定性探路（2026-09-25，`work_dirs/rl_grpo_hbb_dense_v2_probe200`）

动机：探索 pilot step200 后是否还有收益。从该 checkpoint 权重 warm-start，**LR 5e-7→2.5e-7、KL β 0.02→0.05**，再走 150 步；优化器、数据顺序、RNG 和 reference 重新初始化，不是 `--resume-from-checkpoint` 的连续训练轨迹。

| canonical hybrid | all F1 | small R | small F1 |
|---|---:|---:|---:|
| pilot step200（探路起点） | 0.4577 | 0.3734 | 0.4495 |
| pilot step250（原配置末，FAIL） | 0.4484 | 0.3673 | 0.4428 |
| **probe50（=全局 250，低 LR+强 KL）** | **0.4615** | **0.3818** | **0.4568** |
| probe100（=全局 300） | 0.4592 | 0.3772 | 0.4524 |
| probe150（=全局 350） | 0.4578 | 0.3796 | 0.4548 |

**判读（2026-09-28 更正）**：
1. probe50 在 dense-385 上是最高观测点，但与 pilot step250 相比同时改变了 LR、KL、优化器、样本顺序和 reference；“全局 250”仅表示权重经历 200+50 次更新，不能确认 pilot 末回落的原因或收益在 250 步收敛。
2. probe 首组 `max|pol-ref| token logp diff=4.53e-01`，pilot 为 0；旧代码先把 fp32 policy 权重拷进可能为 bf16 的 reference，再对齐 dtype，已发生的舍入无法恢复。probe 权重的评测数字成立，但“强 KL 相对精确 policy 副本”的因果解释不成立。后续训练已改为先对齐 dtype、再复制，并对 fresh reference 不等价硬失败。
3. pilot step200 是较可信的默认候选（同样只有单 seed）；probe50 保留为密集集探索性峰值。稀疏 300 图上 pilot step200 更好；两者差异需独立训练 seed 与未用于挑选 checkpoint 的验证来确认。

## 稀疏子集泛化复验（2026-09-25，300 图 smallGT∈[0,9]，15 类穷举）

| | all P / R / F1 | small P / R / F1 | mean preds |
|---|---|---|---|
| baseline | 0.420 / 0.669 / 0.516 | 0.178 / 0.462 / 0.257 | 7.5 |
| pilot step200 | 0.421 / 0.708 / 0.528 | 0.185 / 0.530 / 0.275 | 7.9 |

**方向性信号**：单次 pilot 在稀疏图 small recall +0.068、all F1 +0.012；样本来自固定 300 图且只有一个训练 seed，不足以断言稀疏泛化的稳定提升。

| 稀疏 300 图 | all P/R/F1 | small P/R/F1 | mean preds |
|---|---|---|---|
| pilot step200 | 0.421 / **0.708** / **0.528** | 0.185 / **0.530** / **0.275** | 7.9 |
| probe50 | 0.413 / 0.688 / 0.516 | 0.183 / 0.508 / 0.269 | 7.9 |

probe50 稀疏侧仍全面 ≥ baseline（small R +0.046、small F1 +0.012、all F1 持平），但低于 step200 的稀疏峰值——**两个产出 checkpoint 的取舍**：probe50 是 dense 小目标峰值（项目主目标），step200 在稀疏图上更均衡。

## 剩余错误结构诊断（2026-09-25，dense-385，从 preds.json 离线分析）

对每个 FN（未匹配 GT）判定其性质：**漏检** = 无任何同标签 IoU≥0.1 的预测；**定位不准** = 存在 IoU∈[0.1,0.5) 的同标签预测。

| | FN 总数 | 纯漏检 | 定位不准 | smallFN 占全部 FN |
|---|---:|---:|---:|---:|
| baseline | 8943 | 7294（82%） | 1649（18%） | 80% |
| step200 | 8728 | 7126（82%） | 1602（18%） | 80% |
| probe50 | 8636 | **7000（81%）** | 1636（19%） | 80% |

**判读边界**：按“无同类 IoU≥0.1 预测”定义，剩余 FN 的 81–82% 属于漏检类；这是启发式分类（可能混有错类/重复匹配），不能单凭该比例断言坐标量化不是瓶颈、切片一定有效，或 RL 收益已经收敛。

## 下一步建议（2026-09-25，基于上述诊断）

1. **slice_infer 切片推理待验证**（AGENTS §7.1 TODO）：448→224 子切片推理 + 坐标还原 + NMS 合并；现有 81% “纯漏检”是按同标签 IoU<0.1 定义的误差分类，不足以预言切片效果。验证协议：dense-385 上 baseline/step200/probe50 × {448, 448+224 切片}。
2. 扩池训练对照：RL-single 的 smallGT≥5 实际仅 **1905 行/10 类**，≥2 为 **3851 行/15 类**（2026-09-28 实测）；先在 ≥2 池重跑 freeze gate，再与 ≥10 池从同一 geom100k 权重做双卡 seed43 对照。
3. 产出固化：`probe200/checkpoint-50`（dense 峰值）与 `pilot/checkpoint-200`（稀疏均衡）做硬链接快照防滚动清理。
4. 暂缓：全量 5102 图 mAP（~40h GPU，dense+sparse 双证据已够方向性结论）、跨数据集（AI-TOD/VisDrone）迁移。

## v2 双卡扩池对照（2026-09-28，进行中；尚无评测结论）

- 前提修复：`grpo_loop.py` 在复制 reference 权重**之前**对齐 adapter dtype，逐张量核对精确相等；fresh run 首组 log-prob 差超过 `1e-3` 直接中止。旧 pilot checkpoint-200 上 GPU 前向差 **0.00e+00**；geom100k 扩池两步 smoke 首组差 **0.00e+00**，loss 有限并成功存盘。checkpoint-2 恢复至 step3，optimizer/order/cursor 正确延续；恢复后的 policy/reference 已不同，首组差 **0.265** 不应触发 fresh 检查。
- 双卡 gate 于 2026-09-28 21:15（UTC+8）启动：geom100k 冻结策略，RL-single smallGT≥2 **3851 行 / 15 类**，seed43、512 个样本 × G16、GPU0/1 各一半；输出 `work_dirs/rl_v2_pair_seed43/gate_small2/`，日志 `gate_gpu{0,1}.log`。只有两个 shard 完成、`summary_gate.json` 四门通过且确有 512 组 / 8192 轨迹才允许训练。23:42 因用户准备重启服务器，已停止两个 gate 子进程、编排器退出并 `sync`；当时**尚未通过 gate、训练未启动**。
- 暂停点：shard0 **2563/4096**、shard1 **2387/4096**；全部 JSONL 行均完整可解析、`(sample_id,k)` 唯一，且与 seed43 抽样 / 分片匹配。重启后在 `Embodied/` 工作根执行 `RESUME_GATE=1 nohup bash shell/rl-grpo-hbb-v2-pair.sh >> work_dirs/rl_v2_pair_seed43/orchestrator.log 2>&1 < /dev/null &`；launcher 保留旧日志，`freeze_check.py` 按 `(sample_id,k)` 跳过已有结果。不要不设 `RESUME_GATE` 重跑编排器；如 GPU 尚不可新建进程，保留文件待 GPU 恢复再继续。
- 通过后由 `shell/rl-grpo-hbb-v2-pair.sh` 自动并发启动 GPU0 `control_small10`（1089 行 / 8 类）与 GPU1 `expanded_small2`（3851 行 / 15 类）：同从 geom100k 权重、seed43、G16、LR 5e-7、KL 0.02、200 步、每 50 步存档。唯一预设实验因素是训练池过滤阈值；训练输入顺序随池变化，不应解释为逐 minibatch 配对。总控日志：`work_dirs/rl_v2_pair_seed43/orchestrator.log`。
- 待评估：每个 checkpoint 在同一 canonical dense-385、固定 sparse-300 图像上用 hybrid/15 类协议比 micro F1、小目标召回/F1、预测框数/误报、类别错误；和 geom100k 及既有 pilot seed42 对照。未复验前不称扩池效果或多 seed 稳定性。

## Follow-ups（未做）

- 双卡对照完成后核对 gate、训练轨迹和 canonical dense/sparse 验收；指标尚未产出。
- Run B ckpt 留存 `work_dirs/rl_grpo_hbb_dense_g16_2k5/snapshots/checkpoint-{250,500,750,1000}` 作反例
