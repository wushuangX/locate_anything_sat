# 小目标检测增强 — 2026 主流方法设计与资源受限落地方案

> 本文件面向当前项目目标：**在没有足够 GPU 和大规模数据训练网络的条件下，用微调/PEFT 技术让 LocateAnything-3B 适配遥感图像小目标检测；若能力允许，再探索 OBB（Oriented Bounding Box，有向框）扩展。**
>
> 文档内容包含：源码瓶颈诊断 → 2026 主流方法对齐 → 资源受限 PEFT 路线 → 小目标切片/粗到精方案 → 可选 OBB 扩展 → 实验路线图。每个策略点尽量标出可参考的方法或论文，方便后续实现和学习。

______________________________________________________________________

## 0. 结论先行

当前项目最现实、最符合 2026 主流的路线不是 full SFT，也不是重训检测器，而是：

```text
LocateAnything-3B
  + tiled / sliced inference（SAHI-style）
  + LoRA / PEFT 小样本域适配
  + 小目标上采样与视觉提示 / zoom crop refine
  + small-object 分桶评估
  + OBB 先做 AABB envelope baseline，真 OBB 作为后续协议扩展
```

推荐优先级：

| 优先级 | 方向 | 训练成本 | 是否建议现在做 |
|---|---|---:|---|
| P0 | Base LocateAnything 直接 RS 小目标评估 | 0 | 必做 baseline |
| P1 | SAHI / tiled inference | 0 | 必做，最划算 |
| P2 | LoRA / PEFT 遥感小样本适配 | 低 | 主线 |
| P3 | visual prompt / GT crop refine | 低~中 | 可做增强 |
| P4 | CoT / structured reasoning | 低~中 | 辅助，不做主贡献 |
| P5 | 真 OBB 输出协议扩展 | 中~高 | 后续，不建议第一阶段做 |

______________________________________________________________________

## 1. 与 2026 主流技术的对齐判断

### 1.1 小目标主流：切片 / 粗到精仍是强基线

遥感小目标检测最稳定的工程路线仍然是：**大图切片 → 小图检测 → 坐标还原 → NMS/去重**。这类方法不要求改模型，能直接提高有效目标尺度。

**参考方法 / 论文：**

- **SAHI — Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection**，Akyon et al., 2022，arXiv:2202.06934。该论文提出通用 slicing aided inference 和 slicing aided fine-tuning，在 VisDrone/xView 上提升小目标 AP；不改检测器即可使用。
- **ASAHI / Adaptive Slicing-Aided Hyper Inference**：自适应切片方向，减少固定切片冗余，适合高分辨率遥感图。
- **Sliding Window / Tiled Inference**：遥感传统强基线，DOTA/xView/VisDrone 工程中常用。

对 LocateAnything 来说，SAHI-style 切片尤其合理：模型坐标固定为 `[0,1000]`，tile 变小后同一个 20px 目标占用更多坐标单位，有效定位精度提升。

### 1.2 资源受限主流：PEFT / LoRA，而不是 full training

你没有足够 GPU 和数据，这和 2025–2026 RS visual grounding 方向高度一致：主流不是从头训练网络，而是用 **Parameter-Efficient Fine-Tuning (PEFT)** 适配基础模型。

**参考方法 / 论文：**

- **LoRA — Low-Rank Adaptation of Large Language Models**，Hu et al., 2021/2022。冻结原模型，只训练低秩矩阵。
- **Efficient Adaptation For Remote Sensing Visual Grounding**，Moughnieh et al., 2025，arXiv:2503.23083。针对 RS visual grounding，比较 LoRA、BitFit、Adapters，适配 Grounding DINO / OFA，强调用低成本 PEFT 达到接近或超过 SOTA 的效果。
- **BitFit — Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models**，Ben Zaken et al., 2021。只调 bias，极低成本，但表达能力有限。
- **Adapters / Houlsby Adapters**，Houlsby et al., 2019。插入小型瓶颈模块，冻结主体模型。

对本项目，最推荐：

```text
freeze_llm=True
freeze_backbone=True
freeze_mlp=False
use_llm_lora=64
use_backbone_lora=0 或 低 rank
```

即：**冻结大部分权重，只训练 MLP + LLM LoRA；若遥感域差特别大，再少量打开 vision LoRA。**

### 1.3 RS-VLM 主流：统一 grounding，而不是只做单一检测器

2026 前后的遥感 VLM 趋势是：HBB、OBB、mask、文本 grounding 尽量统一到一个 VLM/grounding 框架里。

**参考方法 / 论文：**

- **GeoGround — A Unified Large Vision-Language Model for Remote Sensing Visual Grounding**，Zhou et al., 2024，arXiv:2411.11904。统一支持 HBB、OBB、mask RS visual grounding，引入 Text-Mask、Prompt-Assisted Learning、Geometry-Guided Learning。
- **SATGround — A Spatially-Aware Approach for Visual Grounding in Remote Sensing**，Toker et al., 2025，arXiv:2512.08881。通过 specialized control tokens / grounding module 增强 VLM 在卫星图中的 spatial localization。
- **Grounding DINO**，Liu et al., 2023。开放词汇检测/grounding 强基线，RS 适配工作常以其为底座。
- **OFA — One For All**，Wang et al., 2022。统一多模态预训练模型，被 PEFT RS grounding 工作用作适配对象。

LocateAnything 本身就是 grounding VLM，路径上是对齐主流的；但当前文档需要把“PEFT + 统一 grounding + OBB 可扩展”写清楚。

______________________________________________________________________

## 2. LocateAnything 小目标瓶颈诊断（源码核验）

### 2.1 视觉 token 分辨率：784 像素 / token

MoonViT 的空间压缩只发生在首尾两步，中间无下采样（`eaglevl/model/moon_vit/modeling_vit.py` L606–610，transformer 在 patch 全分辨率上跑，靠 `grid_hws` 保持二维结构）：

| 阶段 | 操作 | 压缩 | 来源 |
|---|---|---|---|
| patchify | Conv2d kernel=stride=**14** | 每维 ÷14 | `modeling_vit.py` L264–265 |
| merge | 2×2 patch 合并成 1 token（学得线性层） | 每维再 ÷2 | `patch_merger` L546–548 |
| **合计** | **1 视觉 token = 28×28 = 784 像素** | **784× 空间压缩** | — |

图像级缩放（`rescale`，`image_processing_locateanything.py` L52–55）只在 patch 数超过 `in_token_limit=25600`（`preprocessor_config.json` L8；合并后 6400 token，约对应 2240×2240 图）时才触发 BICUBIC 缩图；位置编码硬上限 511 patch/维（L68）。

> 精确表述：主因不是“把图缩小”（MoonViT 是原生分辨率，中等图不缩），而是 **patchify(14×) + merge(2×)** 导致亚 patch 目标塌缩进单 token。这是不可逆的信息瓶颈。

### 2.2 代入遥感小目标真实尺寸

设目标边长 $s$ 像素，token 感受野 $28\times28=784$ 像素：

| 目标 | patch 占用 | token 占用 | 目标在单 token 中的信号占比 |
|---|---|---|---|
| **AI-TOD 均值 12.8px** | $12.8/14\approx0.91$ patch（亚 patch） | ~1 token（与背景混合） | $\approx 12.8^2/784\approx 21\%$ |
| 20px | $20/14\approx1.4$ patch | ~1 token | $\approx 20^2/784\approx 51\%$ |

**参考数据集 / 论文：**

- **AI-TOD — Tiny Object Detection in Aerial Images**。极小目标遥感数据集，平均目标尺寸约 12.8px，是验证 tiny object detection 的核心数据集之一。
- **NWD — Normalized Gaussian Wasserstein Distance for Tiny Object Detection**，Wang et al., 2021/2022。针对 tiny object IoU 对微小位移过敏的问题，用 NWD 作为更稳定的度量/损失思想；即使 LocateAnything 不直接用 NWD，也可作为小目标评估解释参考。
- **VisDrone**，Zhu et al., 2018/2021。无人机密集小目标检测常用数据集。
- **xView**，Lam et al., 2018。大规模遥感目标检测数据集，小目标/密集目标丰富。

### 2.3 第二个叠加瓶颈：坐标量化 `[0,1000]`

坐标硬约束为归一化整数 `[0,1000]`（`<0>`~`<1000>` 共 1001 token，见 `arguments.py` 与 AGENTS 1.2）。

- 4000px 图中，1 坐标单位 = 4px；
- 20px 框只占 $20/4000\times1000=5$ 个坐标单位；
- 视觉 token 给不出亚像素 + 坐标网格只到 4px 粒度 → 小框定位误差被放大。

**对应方法：**

- **SAHI / sliced inference**：减小输入 tile 尺寸，等价提高坐标实际像素精度。
- **Coarse-to-fine crop refinement**：先粗定位，再在 crop 上重新预测 `[0,1000]` 坐标。
- **Super-resolution / zoom-in**：理论可用，但训练成本和伪影风险更高，不建议第一阶段做。

______________________________________________________________________

## 3. 资源受限主线：PEFT 微调方案

### 3.1 不建议 full SFT

Full SFT 需要大量 GPU、长上下文显存、足够 RS 数据和防遗忘混合数据。你的约束不适合 full SFT。

**不建议第一阶段做：**

- 全量解冻 LLM；
- 全量解冻 MoonViT；
- 改 PBD 结构；
- 从头训练遥感检测模型。

### 3.2 推荐 LoRA 配置

当前项目已有 LoRA 脚本：`Embodied/shell/locate-anything-lora-visual-prompt.sh`。

推荐从以下配置开始：

```bash
FREEZE_LLM=True
FREEZE_BACKBONE=True
FREEZE_MLP=False
USE_LLM_LORA=64
USE_BACKBONE_LORA=0
MAX_SEQ_LENGTH=4096 或 8192   # 非 Hopper/Blackwell 优先低上下文
MAX_NUM_TOKENS=4096 或 8192
MAX_STEPS=500~2000 先 smoke
```

若遥感域差距太大，再试：

```bash
USE_BACKBONE_LORA=16 或 32
```

但 vision LoRA 会提高显存和过拟合风险，优先级低于 LLM LoRA + MLP。

**参考方法 / 论文：**

- **LoRA**：Hu et al., 2021/2022。
- **Efficient Adaptation For Remote Sensing Visual Grounding**：Moughnieh et al., 2025，RS visual grounding 的 PEFT 实证参考。
- **Adapters**：Houlsby et al., 2019。
- **BitFit**：Ben Zaken et al., 2021。

### 3.3 数据策略：少数据也要做“小目标重采样”

小数据条件下，不要平均采样。建议 recipe 层面或转换脚本中做：

| 策略 | 方法 | 参考思想 |
|---|---|---|
| 小目标样本上采样 | 按 bbox 面积中位数/平均面积加权 | AI-TOD / tiny object detection 常规做法 |
| 密集图上采样 | 按目标数量加权 | VisDrone/xView dense detection |
| 原始能力防遗忘 | 混入少量 LocateAnything 原数据 | Continual SFT / rehearsal |
| prompt 多样化 | car/vehicle/small vehicle 等同义 prompt | visual grounding prompt augmentation |
| tile/crop 样本 | 大图切片生成训练样本 | SAHI slicing fine-tuning |

建议 recipe：

```json
{
  "rs_tiny_aabb": {
    "annotation": "data/anns/rs_tiny_train.jsonl",
    "root": "data/images/",
    "repeat_time": 2.0,
    "data_augment": true,
    "visual_prompt": false
  },
  "rs_visual_prompt": {
    "annotation": "data/anns/rs_tiny_single_class_train.jsonl",
    "root": "data/images/",
    "repeat_time": 1.0,
    "data_augment": true,
    "visual_prompt": true
  },
  "locany_rehearsal": {
    "annotation": "data/anns/locany_subset.jsonl",
    "root": "data/images/",
    "repeat_time": 0.1,
    "data_augment": false
  }
}
```

______________________________________________________________________

## 4. 主方法：Tile + PEFT + Refine

### 4.1 Stage 0：Base direct baseline

先不训练，直接评估：

```text
LocateAnything-3B + 原图 direct detect
```

目的：知道 base model 在 RS 上到底差在哪里，是漏检、类别混淆，还是框偏移。

### 4.2 Stage 1：SAHI / tiled inference baseline

```text
大图 → overlap tiles → LocateAnything detect each tile → 坐标还原 → NMS / IoM 去重
```

推荐参数起点：

| 参数 | 起点 |
|---|---:|
| tile size | 768 / 1024 / 1280 |
| overlap | 0.2 或 0.25 |
| merge | NMS 或 IoM |
| score | 若模型输出无 score，可用文本解析 + 后处理规则 |

**参考方法 / 论文：**

- **SAHI**，Akyon et al., 2022。
- **ASAHI**，Adaptive Slicing-Aided Hyper Inference。
- **Slicing Fine-tuning (SF)**，SAHI 论文中的训练增强分支。

> 对你的项目：Stage 1 是最高性价比方案，即使不微调，也很可能显著改善小目标。

### 4.3 Stage 2：LoRA / PEFT on tiled AABB data

把遥感大图切片后转成 LocateAnything JSONL，用 LoRA 训练：

```text
大图 GT → tile crop → 保留 tile 内目标 → 坐标重归一化到 [0,1000] → JSONL
```

优点：

- token 更短；
- 小目标变大；
- 坐标量化更细；
- 非 Hopper/Blackwell 也更容易跑。

### 4.4 Stage 3：Visual prompt / GT crop refine

现有代码已有 `visual_prompt=true` 路径：`apply_visual_prompt_to_sample` 会用 GT 框裁 crop，并把 prompt 中类别替换为 `<image-N>`。

这个阶段训练的是：

```text
源图 + 目标 crop visual prompt → 更精确定位
```

注意：这不是完整 learned coarse proposal policy。它训练的是 **zoom/refine 能力**；推理时 coarse proposal 可以来自：

- tiled inference；
- base LocateAnything 粗检；
- 外部 detector；
- 人工/规则 ROI。

**参考方法 / 论文：**

- **Visual Prompting / In-context Visual Prompting**：以图像示例作为 query，提高跨域/细粒度识别。
- **GeoGround** 中的 prompt-assisted learning 思想：不同输出信号由 prompt 控制。
- **Grounding DINO**：文本 prompt 引导检测，是可对照的 grounding 基线。

______________________________________________________________________

## 5. CoT / Structured Reasoning：辅助，不做主线

原文档已经指出：CoT 与 PBD 的“并行快”目标冲突。这个判断保留。

更建议使用 2026 主流表述：

```text
structured spatial reasoning / localization control tokens
```

而不是自由文本 CoT。

**参考方法 / 论文：**

- **SATGround — A Spatially-Aware Approach for Visual Grounding in Remote Sensing**，Toker et al., 2025。通过 structured localization mechanism / task tokens 让 VLM 更好处理卫星图定位。
- **GeoGround**，Zhou et al., 2024。通过 Prompt-Assisted Learning 和 Geometry-Guided Learning 统一 HBB/OBB/mask。

对 LocateAnything：

- 第一阶段不要改 PBD；
- 可在 prompt 中加入轻量空间提示，例如“small vehicles on roads”“ships in harbor”；
- 若要 CoT，只在 `slow/NTP` 或 `hybrid fallback` 中测试，不作为主贡献。

______________________________________________________________________

## 6. OBB 扩展：先 AABB baseline，再考虑真 OBB

当前 LocateAnything 输出协议天然是 AABB/HBB：

```text
<ref>label</ref><box><x1><y1><x2><y2></box>
```

真 OBB 不是简单换数据格式，会影响：

- 输出 token 协议；
- PBD block size；
- generate pattern parser；
- 后处理；
- rotated IoU；
- rotated NMS；
- DOTA evaluation。

### 6.1 OBB 任务为什么重要

遥感图像中飞机、舰船、车辆、桥梁等具有任意方向。HBB 会包含大量背景，密集场景中更容易互相覆盖。

**参考方法 / 论文：**

- **DOTA — A Large-scale Dataset for Object Detection in Aerial Images**，Xia et al., CVPR 2018，arXiv:1711.10398。提供 HBB 和 OBB 两类评估任务，OBB 用四角点标注。
- **Oriented Object Detection in Optical Remote Sensing Images Using Deep Learning: A Survey**，Wang et al., 2023，arXiv:2302.10473。总结 OBB 的 feature misalignment、spatial misalignment、angle periodicity、OBB regression 等问题。
- **Rotated Faster R-CNN / RoI Transformer / Oriented R-CNN / ReDet / S2A-Net / R3Det / KFIoU / GWD / KLD**：遥感 OBB 检测常见代表方法，可作为传统检测器对照。
- **MMRotate**：OpenMMLab rotated object detection toolbox，适合学习 DOTA 格式、rotated NMS、rotated IoU。

### 6.2 第一阶段推荐：OBB → AABB envelope

把 DOTA / OBB 四点框转成外接水平框：

```text
(x1,y1),(x2,y2),(x3,y3),(x4,y4)
  → xmin, ymin, xmax, ymax
  → LocateAnything AABB 格式
```

优点：

- 不改模型；
- 不改 PBD；
- 可直接使用现有微调与评估；
- 能先验证 RS 域适配和小目标能力。

缺点：

- 丢方向；
- 密集旋转目标框会包含更多背景；
- DOTA OBB 指标不能直接公平比较。

### 6.3 第二阶段可选：Quad 输出协议

仓库已有 `<quad>` / `</quad>` token（`constants.py`），但当前 PBD/generate path 没有完整 quad 解码支持。

可设计：

```text
<ref>ship</ref><quad><x1><y1><x2><y2><x3><y3><x4><y4></quad>
```

需要改：

| 模块 | 改动 |
|---|---|
| 数据转换 | DOTA 四点 → `<quad>` 8 坐标 |
| MTP target | block_size 至少 10：`<quad>` + 8 coords + `</quad>` |
| `generate_utils.py` | 新增 quad pattern parser |
| 后处理 | quad → polygon / OBB |
| 评估 | rotated IoU / DOTA eval |
| NMS | rotated NMS / polygon NMS |

风险：这已经是模型输出协议扩展，不再是轻量 LoRA 适配。

### 6.4 第三阶段可选：5 参数 OBB

格式示例：

```text
<obox><cx><cy><w><h><angle></obox>
```

问题：

- 需要新增 `<obox>` token；
- angle 量化范围要设计；
- 角度周期性导致训练不稳定；
- 宽高交换等价问题复杂。

不建议第一阶段做。

### 6.5 与 GeoGround 的关系

GeoGround 代表更主流的统一方向：HBB、OBB、mask 都用 VLM 统一表达。但其方法复杂度显著高于本项目当前资源约束。

本项目建议：

```text
阶段 1：AABB envelope baseline
阶段 2：AABB + tile + LoRA 做强
阶段 3：若 AABB 已稳定，再做 quad / OBB protocol extension
```

______________________________________________________________________

## 7. 评估设计

### 7.1 必须报告 small-object 分桶

仅报告整体 mAP 会掩盖小目标变化。建议至少：

| 指标 | 说明 |
|---|---|
| AP_small | COCO small，area < 32² |
| AP_tiny | 可自定义，side < 16px 或 area < 16² |
| AP_medium / AP_large | 看是否牺牲大目标 |
| Recall_small | 小目标漏检率 |
| tile latency | 切片推理成本 |
| boxes per image | 后处理压力 |

### 7.2 Ablation 表设计

| 实验 | 目的 |
|---|---|
| Base direct | 原模型能力 |
| Base + SAHI | 纯切片收益 |
| LoRA direct | 纯域适配收益 |
| LoRA + SAHI | 主结果 |
| LoRA + visual prompt refine | refinement 收益 |
| AABB envelope on OBB data | OBB 第一阶段 baseline |

### 7.3 推荐数据集

| 数据集 | 用途 | 参考 |
|---|---|---|
| AI-TOD | tiny object 主评估 | Tiny Object Detection in Aerial Images |
| VisDrone | UAV 密集小目标 | VisDrone benchmark |
| xView | 大规模遥感检测 | xView dataset |
| DOTA | OBB/AABB 扩展 | DOTA CVPR 2018 |
| DIOR / DIOR-RSVG | 遥感视觉 grounding | DIOR / RSVG 系列 |
| OPT-RSVG | RS visual grounding PEFT 参考 | Efficient Adaptation for RSVG |

______________________________________________________________________

## 8. 实施路线图

### Phase A：零训练验证

1. Base LocateAnything direct detect；
2. 实现 SAHI-style tiled inference；
3. 坐标还原 + NMS/IoM；
4. 报 small/tiny 分桶指标。

参考：SAHI。

### Phase B：小样本 LoRA

1. AABB 数据转换到 LocateAnything JSONL；
2. 切片训练样本生成；
3. LoRA 微调：LLM LoRA + MLP；
4. small-object weighted sampling；
5. 防遗忘混入少量原数据。

参考：LoRA、Efficient Adaptation for Remote Sensing Visual Grounding。

### Phase C：Refine / visual prompt

1. `visual_prompt=true` 数据；
2. GT crop refine 训练；
3. 推理时用 tile/base model coarse proposal 生成 crop；
4. 对比是否提升 AP_small / AP_tiny。

参考：visual prompting、GeoGround prompt-assisted learning。

### Phase D：OBB baseline

1. DOTA OBB → AABB envelope；
2. 用 AABB 流程训练/评估；
3. 若效果可接受，再设计 quad 输出协议。

参考：DOTA、OBB survey、MMRotate。

______________________________________________________________________

## 9. 当前文档相对原版的关键修正

原版文档强在瓶颈分析，但偏“方法想法”。本版补充了：

1. **把 PEFT/LoRA 提升为主线**，符合你的 GPU/数据约束；
2. **把 SAHI/tiled inference 明确为 P1 必做 baseline**，不是只作为泛泛 C1；
3. **把 visual prompt refine 与 learned coarse-to-fine 区分开**：当前训练的是 refine 能力，不是完整 learned proposal policy；
4. **加入 OBB 分阶段扩展路线**：先 AABB envelope，再 quad/5-param；
5. **为每个关键策略添加方法/论文名**，方便后续学习和实现。

______________________________________________________________________

## 10. 参考方法与论文索引

### 小目标 / 切片

- **SAHI: Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection**，Akyon et al., 2022，arXiv:2202.06934，https://arxiv.org/abs/2202.06934。
- **Adaptive Slicing-Aided Hyper Inference (ASAHI)**，adaptive tiling / adaptive slicing for high-resolution RS images。
- **AI-TOD: Tiny Object Detection in Aerial Images**，tiny aerial object benchmark；官方/社区资料可从 Papers with Code 与 AI-TOD GitHub 入口追踪。
- **NWD: A Normalized Gaussian Wasserstein Distance for Tiny Object Detection**，tiny object localization metric/loss 思路。
- **VisDrone** benchmark，UAV 密集小目标检测/跟踪数据集。
- **xView** dataset，大规模遥感目标检测数据集。

### PEFT / 资源受限微调
- **LoRA: Low-Rank Adaptation of Large Language Models**，Hu et al., 2021/2022，https://arxiv.org/abs/2106.09685。
- **Efficient Adaptation For Remote Sensing Visual Grounding**，Moughnieh et al., 2025，arXiv:2503.23083，https://arxiv.org/abs/2503.23083。
- **BitFit: Simple Parameter-efficient Fine-tuning**，Ben Zaken et al., 2021，https://arxiv.org/abs/2106.10199。
- **Parameter-Efficient Transfer Learning for NLP / Adapters**，Houlsby et al., 2019，https://arxiv.org/abs/1902.00751。

### RS-VLM / grounding
- **GeoGround: A Unified Large Vision-Language Model for Remote Sensing Visual Grounding**，Zhou et al., 2024，arXiv:2411.11904，https://arxiv.org/abs/2411.11904。
- **SATGround: A Spatially-Aware Approach for Visual Grounding in Remote Sensing**，Toker et al., 2025，arXiv:2512.08881，https://arxiv.org/abs/2512.08881。
- **Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection**，Liu et al., 2023，https://arxiv.org/abs/2303.05499。
- **OFA: Unifying Architectures, Tasks, and Modalities Through a Simple Sequence-to-Sequence Learning Framework**，Wang et al., 2022，https://arxiv.org/abs/2202.03052。

### OBB / 旋转框
- **DOTA: A Large-scale Dataset for Object Detection in Aerial Images**，Xia et al., CVPR 2018，arXiv:1711.10398，https://arxiv.org/abs/1711.10398。
- **Oriented Object Detection in Optical Remote Sensing Images Using Deep Learning: A Survey**，Wang et al., 2023，arXiv:2302.10473，https://arxiv.org/abs/2302.10473。
- **RoI Transformer**，Ding et al., CVPR 2019，rotation-aware RoI learning。
- **Rotated Faster R-CNN**，classic two-stage rotated detector baseline。
- **Oriented R-CNN**，oriented proposal + oriented box regression baseline。
- **ReDet**，rotation-equivariant detector for aerial object detection。
- **S2A-Net**，single-shot alignment network for oriented object detection。
- **R3Det**，refined single-stage rotated detector。
- **KFIoU / GWD / KLD**，OBB regression losses for Gaussian/rotated box modeling。
- **MMRotate** toolbox，OpenMMLab rotated object detection toolbox，https://github.com/open-mmlab/mmrotate。

______________________________________________________________________

## 11. 关键代码定位速查

| 主题 | 位置 |
|---|---|
| MoonViT 前向（patchify→encoder→merge，无内部下采样） | `eaglevl/model/moon_vit/modeling_vit.py:595-611` |
| patch embed（Conv2d kernel=stride=14） | `modeling_vit.py:264-266` |
| patch merger（2×2 合并） | `modeling_vit.py:533-559` |
| 图像 rescale + token 上限 + pos emb 上限 | `eaglevl/utils/locany/image_processing_locateanything.py:46-71` |
| `in_token_limit=25600` | `preprocessor_config.json:8` |
| MTP/PBD 训练目标构造（block_size=6 切块） | `eaglevl/train/locany_finetune_magi_stream.py:254-479` |
| visual prompt GT 裁剪（Path A 入口） | `eaglevl/train/tools.py:406-459` |
| visual_prompt 触发（recipe `visual_prompt=true`） | `locany_finetune_magi_stream.py:219, 469-480` |
| 推理模式 fast/hybrid/slow | `eaglevl/utils/locany/generate_utils.py:473-485` |
| 序列长度=视觉 token + 文本 token（打包上限） | `locany_finetune_magi_stream.py:706-989`（`max_num_tokens`） |
