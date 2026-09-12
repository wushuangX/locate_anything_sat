# AGENTS.md — LocateAnything 遥感微调开发指南

> 本文件指导 AI Agent 在本项目中规范地进行数据准备、微调训练、评估与代码改动。
> 基座模型为 NVIDIA **LocateAnything-3B**（MoonViT + Qwen2.5，Parallel Box Decoding）。
> **项目目标：将 LocateAnything 微调到遥感（Remote Sensing, RS）图像，并着重增强小目标检测性能。**

______________________________________________________________________

## 1. 项目概览

### 1.1 目标与定位

- **域适配**：LocateAnything 预训练于自然图像（检测/GUI/OCR/指代等 138M 样本）。本项目通过**持续监督微调（Continual SFT）**将 grounding 能力迁移到遥感场景（俯视、高分辨率、密集小目标、RS 专属类别：车辆/舰船/飞机/储油罐/桥梁等）。
- **小目标增强（核心）**：遥感图像目标常为极高密度、极小尺寸（几像素至几十像素）。本项目围绕**坐标量化精度、视觉 token 分辨率、切片推理、视觉提示微调、多尺度增强、防遗忘数据混合**六条主线系统性增强小目标检测。

### 1.2 关键设计约束（锁定）

- **基座模型与架构不可替换**：MoonViT 视觉编码器 + Qwen2.5 LLM + MLP 投影器 + Parallel Box Decoding（PBD）。微调只调权重/数据/推理流程，不改核心 `modeling_locateanything.py` 的 PBD 解码结构（除非有明确理由并记录）。
- **坐标系统为归一化整数 `[0, 1000]`**：这是 LocateAnything 的硬约束（`<0>`~`<1000>` 共 1001 个坐标 token）。小目标增强策略必须在此约束内工作——**切片是提升有效坐标精度的首选手段**（详见第 7 节）。
- **训练目标平台：NVIDIA GPU**（Hopper/Blackwell 用 MagiAttention 支持长上下文 32K+；A100/4090/3090 用 SDPA，仅支持 ~4K 序列）。本 MacBook（Apple Silicon）**仅作代码开发与数据转换，不训练**。
- **本目录 `Embodied/` 是自包含工作根**：所有路径、`pip install -e .`、训练/评估命令都以 `Embodied/` 为基准。同级 `Eagle/`、`Eagle2_5/` 是 EAGLE 单仓库的兄弟模型发布，**本项目零代码依赖它们**（见根 README 与评估说明），可忽略。

### 1.3 目录结构

```
locate_anything_sat/                 # ← 仓库根（AGENTS.md 位于此，便于 agent 发现）
├── AGENTS.md                        # ★ 本文件（agent 开发指南）
├── docs/                            # 手工评测清单等
│   └── dota_v1_val_manual_tiles.md  # DOTA val 15 类 × 3 tile
├── Embodied/                        # ← 工作根（训练/评估/pip install 在此执行）
│   ├── eaglevl/                     # 核心 Python 包（pip install -e . 安装名为 locate_anything）
│   │   ├── model/
│   │   │   ├── locany/              # LocateAnything 模型实现
│   │   │   │   ├── modeling_locateanything.py   # LocateAnythingForConditionalGeneration（PBD 主模型，慎改）
│   │   │   │   ├── modeling_qwen2.py            # Qwen2.5 LLM（76KB，含 PBD 相关改动）
│   │   │   │   ├── configuration_locateanything.py
│   │   │   │   ├── mask_sdpa_utils.py / mask_magi_utils.py   # 注意力掩码（PBD 关键）
│   │   │   │   └── tokenization_qwen2*.py
│   │   │   └── moon_vit/
│   │   │       └── modeling_vit.py              # MoonViT 视觉编码器（原生分辨率视觉 token）
│   │   ├── patch/                   # monkey patches（fused ops / packing attn / dataloader / fp8）
│   │   ├── sp_utils/                # 流式打包（streaming packing）+ 序列并行（ring/ulysses）
│   │   ├── train/                   # 训练逻辑
│   │   │   ├── locany_finetune_magi_stream.py   # ★ 主训练脚本（full SFT + LoRA 均走此入口）
│   │   │   ├── dataset.py                       # 数据集与流式打包 collator（38KB，数据混入点）
│   │   │   ├── arguments.py                     # ModelArguments / DataTrainingArguments（新增超参在此）
│   │   │   ├── constants.py                     # 特殊 token（IMG_CONTEXT_TOKEN / BOX_* / <0>~<1000>）
│   │   │   ├── augmentation.py                  # resize 增强（data_augment 开关）
│   │   │   ├── tools.py                         # checkpoint 回调 / sample 处理
│   │   │   └── fastseek/                        # FastSeek 相关
│   │   ├── utils/locany/            # LocateAnythingProcessor / ImageProcessor
│   │   └── conversation.py          # 对话模板
│   ├── evaluation/                  # 评估（COCO/LVIS/Grounding/Point/ScreenSpot-Pro）
│   │   ├── inference_grounding_ddp.py           # grounding/point 推理（切片推理改造点）
│   │   ├── inference_detection_ddp.py           # 检测推理
│   │   ├── fastevaluate/                        # COCO/LVIS mAP 指标（需从 Rex-Omni 下载）
│   │   └── scripts/                             # eval_coco.sh / eval_lvis.sh / eval_grounding.sh / eval_sspro.sh
│   ├── shell/                       # ★ 训练启动脚本
│   │   ├── locate-anything-lora-visual-prompt.sh   # LoRA + 视觉提示微调（本项目主用）
│   │   └── locate-anything-streaming.sh            # full SFT 流式打包
│   ├── deepspeed_configs/           # zero_stage1 / zero_stage2
│   ├── document/                    # TRAINING.md / DATA_PREPARATION.md / STREAMING_PACKING.md / RESULTS.md
│   ├── assets/                      # 论文图
│   ├── locateanything_worker.py     # ★ 推理 worker API（detect / ground_multi / point / detect_text）
│   ├── scripts/prompt_viz_app.py    # Streamlit：改 prompt + GT/预测可视化（§3.5.1）
│   └── pyproject.toml               # 依赖（pinned：transformers==4.57.1, deepspeed==0.15.4, peft==0.12.0 …）
├── Eagle/                           # EAGLE 兄弟发布（零依赖，可忽略）
└── Eagle2_5/                        # 同上
```

______________________________________________________________________

## 2. 环境管理

### 2.1 包管理器

- LocateAnything 用 **pip**（非 uv/poetry/conda）。根 `pyproject.toml` 已 pin 关键版本，**不要随意升降**：
  - `transformers==4.57.1`、`tokenizers==0.22.0`、`deepspeed==0.15.4`、`accelerate==1.5.2`、`peft==0.12.0`、`liger_kernel==0.3.1`、`timm>=1.0.11`。
- 安装（在 `Embodied/` 下）：
  ```bash
  pip install -e .
  ```

### 2.2 注意力后端（决定可用 GPU 与序列长度）

| 后端 | `--attn_implementation` | GPU 架构 | 最大序列 | 用途 |
|---|---|---|---|---|
| **Magi Attention** | `magi` | **Hopper**(H100/H800/H20)/**Blackwell** | **32K+** | 长上下文、PBD 块并行注意力。**推荐** |
| **SDPA** | `sdpa` | 任意 GPU（含 A100/4090/3090） | **~4K** | 非 Hopper/Blackwell 唯一选项；仅支持短序列微调 |

> 遥感高分辨率切片训练往往需要长上下文（16K–32K）。若算力为 A100/4090，必须用切片降低单序列 token 量（见第 7 节），否则受 ~4K 限制。

MagiAttention 安装（仅 Hopper/Blackwell）：
```bash
git clone https://github.com/SandAI-org/MagiAttention.git && cd MagiAttention
git checkout v1.0.5 && git submodule update --init --recursive
pip install -r requirements.txt && pip install --no-build-isolation .
```

### 2.3 远程训练服务器

训练在 NVIDIA GPU 上进行（本机仅开发）。**<TODO: 填写本项目实际使用的远程服务器（SSH/端口、GPU 型号显存、仓库数据盘路径、UV_CACHE_DIR/HF_HOME 等环境变量）>**。参考 CoastExtraction 的双服务器模板格式登记。

通用准备：
```bash
# 1. SSH 连接后设环境变量（每次非交互式 shell 需手动 export）
export PATH="$PATH:/root/.local/bin"
export HF_HOME=<数据盘上的 HF 缓存>      # 模型权重较大，必须放数据盘
export HF_TOKEN=<your_token>             # 下载 nvidia/LocateAnything-3B 需要

# 2. 同步仓库到数据盘
cd <数据盘> && git clone <仓库地址> && cd locate_anything_sat/Embodied
pip install -e .

# 3. 拉取基座权重（首次）
huggingface-cli download nvidia/LocateAnything-3B --local-dir <数据盘>/LocateAnything-3B
```

______________________________________________________________________

## 3. 运行训练与评估

### 3.1 微调入口

所有微调（full SFT 或 LoRA）都走 **`eaglevl/train/locany_finetune_magi_stream.py`**，通过 `--freeze_*` / `--use_llm_lora` / `--use_backbone_lora` 区分模式。本项目默认 **LoRA 微调**（省显存、防遗忘、可插拔）。

关键模式开关：

| 目标 | `--freeze_llm` | `--freeze_backbone` | `--freeze_mlp` | `--use_llm_lora` | `--use_backbone_lora` | DeepSpeed |
|---|---|---|---|---|---|---|
| **LoRA（本项目首选）** | True | True | False | 64（秩） | 0/64 | zero_stage1 |
| Full SFT | False | False | False | 0 | 0 | zero_stage2 |

### 3.2 LoRA 微调（遥感 + 视觉提示，主用）

直接用现成脚本，通过环境变量覆盖（脚本内 `${VAR:-default}` 全部可覆盖）：

```bash
cd Embodied
export GPUS=8
export MODEL_PATH=<数据盘>/LocateAnything-3B          # 或 nvidia/LocateAnything-3B
export META_PATH="./locany_recipe/rs_small_obj.json"  # 遥感数据 recipe（见第 4 节）
export OUTPUT_DIR="work_dirs/rs_lora_small_obj"
export MAX_STEPS=5000
export LR=2e-5
export DEEPSPEED_CONFIG="deepspeed_configs/zero_stage1_config.json"
# 非视觉提示任务时 USE_BACKBONE_LORA=0；视觉提示微调建议开 backbone LoRA
export USE_LLM_LORA=64
export USE_BACKBONE_LORA=0

bash shell/locate-anything-lora-visual-prompt.sh
```

脚本默认参数（来自 `shell/locate-anything-lora-visual-prompt.sh`，均可覆盖）：
`MAX_SEQ_LENGTH=16384`、`MAX_NUM_TOKENS_PER_SAMPLE=16384`、`MAX_NUM_TOKENS=16384`、`PACKING_BUFFER_SIZE=32`、`SAVE_STEPS=100`、`WARMUP_STEPS=500`、`bf16=True`、`grad_checkpoint=True`、`block_size=6`、`attn_implementation=magi`、`report_to=tensorboard`。

> **非 Hopper/Blackwell**：把脚本里 `--attn_implementation magi` 改为 `sdpa`，并把 `MAX_SEQ_LENGTH`/`MAX_NUM_TOKENS` 降到 ~4096。

**几何增强长时程训练**（2×4090 / sdpa / 4K token / 基座+LDA / 默认 10 万步 / 保留最新 3 个 ckpt）：
```bash
bash shell/train-dota-geom-lora.sh   # META_PATH 默认 dota_v1_hbb_448_mix_v1_geom_train_only.json（五 key：0°/90°/180°/270°/hflip + color_jitter）
```

### 3.3 Full SFT（需要更强域适配时）

```bash
torchrun --nproc_per_node=8 \
  eaglevl/train/locany_finetune_magi_stream.py \
  --model_name_or_path <数据盘>/LocateAnything-3B \
  --meta_path "./locany_recipe/rs_small_obj.json" \
  --output_dir work_dirs/rs_full_sft \
  --freeze_llm False --freeze_backbone False --freeze_mlp False \
  --use_llm_lora 0 --use_backbone_lora 0 \
  --block_size 6 --causal_attn False --attn_implementation magi \
  --max_steps 25000 --learning_rate 2e-5 --bf16 True \
  --max_seq_length 16384 --max_num_tokens 25600 \
  --grad_checkpoint True \
  --deepspeed "deepspeed_configs/zero_stage2_config.json" \
  --report_to tensorboard
```

完整参数表见 `document/TRAINING.md`。

### 3.4 后台训练与监控

```bash
nohup bash shell/locate-anything-lora-visual-prompt.sh > work_dirs/train.log 2>&1 &
echo $! > work_dirs/train.pid
tail -f work_dirs/train.log
nvidia-smi
# 终止：kill $(cat work_dirs/train.pid)
```

### 3.5 推理（Worker API）

LocateAnything 统一通过 `locateanything_worker.LocateAnythingWorker` 推理（基于 transformers，非训练路径）：

```python
from locateanything_worker import LocateAnythingWorker
worker = LocateAnythingWorker("<数据盘>/LocateAnything-3B")  # 或微调后 checkpoint 路径
print(worker.detect(img, ["vehicle", "ship", "aircraft"])["answer"])
print(worker.point(img, "the small target")["answer"])
```

输出格式：框 `<ref>label</ref><box><x1><y1><x2><y2></box>`（坐标为 `[0,1000]` 整数，除以 1000 得相对坐标）；点 `<box><x><y></box>`；无目标 `<box>none</box>`。

### 3.5.1 Streamlit 提示词 / GT 可视化

交互改 detect 提示词、对照 GT（绿）与预测（红）。脚本：`Embodied/scripts/prompt_viz_app.py`（依赖已在 `pyproject.toml` 的 `streamlit`）。**在 NVIDIA 训练机上跑**，Mac 不能加载 3B。

```bash
cd Embodied
source .venv/bin/activate   # 或项目实际 venv
CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$(pwd)" \
  streamlit run scripts/prompt_viz_app.py \
  --server.port 6006 --server.address 0.0.0.0 --server.headless true
```

侧栏填写 checkpoint 与 data root（DOTA mix 默认识别 `…/data/dota_v1_hbb_448_mix_v1`）。页面选 `tiles` split + 文件名过滤，不必手打路径。模板含 `{class}` 时按勾选的 DOTA 类各调用一次 `worker.predict`（与 `eval_baseline_dota.py` 相同句式）；全勾 15 类才接近 mAP 穷举协议。点 Detect 才占 GPU。

手工 45 张 val tile（每类 3 张）：[`docs/dota_v1_val_manual_tiles.md`](docs/dota_v1_val_manual_tiles.md)。


### 3.6 评估

评估脚本在 `evaluation/scripts/`，依赖 `fastevaluate`（需从 Rex-Omni 下载，见 `evaluation/README.md`）：

```bash
# 1. 安装 fastevaluate（首次）
svn export https://github.com/IDEA-Research/Rex-Omni/trunk/evaluation/fastevaluate evaluation/fastevaluate
cd evaluation/fastevaluate && pip install -e . && pip install shapely && cd ..

# 2. 通用 grounding/point 评估（dataset 可选 VisDrone/Dense200/RefCOCOg_* 等）
bash evaluation/scripts/eval_grounding.sh \
    --dataset VisDrone --eval_type box_eval \
    --model_path work_dirs/rs_lora_small_obj/<ckpt> \
    --image_root <EvalData 根> \
    --output_base <EvalData>/_eval_results/box_eval/
```

**遥感小目标自定义评估**：将 RS 测试集（如 AI-TOD/VisDrone test）转为 LocateAnything JSONL 标注格式（见第 4 节），复用 `eval_grounding.sh`，并额外按目标尺寸（small/medium/large，COCO 面积分桶）分桶报告 mAP——**小目标分桶 mAP 是本项目核心指标**。

______________________________________________________________________

## 4. 数据准备（遥感适配的核心工作量）

### 4.1 两级数据配置

1. **JSONL 标注**（每行一个 ShareGPT 样本）
2. **Recipe JSON**（`--meta_path`，映射数据集名 → 标注/图像根/采样权重）

JSONL 样例（遥感检测，多类别用 `</c>` 分隔）：
```jsonl
{"conversations": [{"from": "human", "value": "Locate all the instances that matches the following description: car</c>ship</c>plane."}, {"from": "gpt", "value": "<ref>car</ref><box><120><200><145><225></box><ref>ship</ref><box><500><600><640><680></box>"}], "image": "rs/train/0001.tif"}
```

> **注意坐标**：归一化整数 `[0,1000]`。像素框 → token 框：`round(x / W * 1000)`。小目标（如 20px 框）在 4000px 图中仅占 ~5 个 token 宽——**这是为何必须切片**（见第 7 节）。

Recipe JSON 样例（遥感 + 防遗忘混合）：
```json
{
  "rs_ai_tod": {
    "annotation": "data/anns/ai_tod_train.jsonl",
    "root": "data/images/",
    "repeat_time": 2.0,
    "data_augment": true
  },
  "rs_visdrone": {
    "annotation": "data/anns/visdrone_train.jsonl",
    "root": "data/images/",
    "repeat_time": 1.5,
    "data_augment": true
  },
  "locany_original_sample": {
    "annotation": "data/anns/locany_subset.jsonl",
    "root": "data/images/",
    "repeat_time": 0.3,
    "data_augment": false
  }
}
```


Recipe 字段：`annotation`（str/list，多文件合并）/ `root`（图像根）/ `repeat_time`（≥1 重复，<1 下采样）/ `data_augment`（resize 增强，多尺度，推荐对小目标开）/ `rotate`+`hflip`（配合离线几何 JSONL，加载时旋转翻转图像）/ `color_jitter`（亮度/对比度/饱和度抖动，strength 0.2）/ `visual_prompt`（视觉提示微调，与几何增强互斥）。


> **随机性与可复现性约定（2026-09 决策）**：数据集生成（任务混合采样、负样本采样、train/val 划分）**不固定随机种子**，接受由此引入的随机波动，不为 seed 复现做任何工程投入。实验对比与追溯以**数据集版本目录 + metadata 统计**（实际任务构成、样本数、划分规模）为准，不追求逐 bit 复现数据文件。

### 4.2 遥感数据集转换

主流遥感检测数据集（COCO/DOTA 格式）需转成 LocateAnything JSONL。DOTA HBB 第一阶段使用 `Embodied/scripts/convert_dota_hbb_to_locany.py`（默认 `--task-mix`：输出 T1–T5 静态混合任务 JSONL；DOTA train 按源图 9:1 划分内部验证；官方 val 双测试渲染；`--reuse-tiles-from` 复用已切好的 tiles 不重切像素）：
```bash
cd Embodied
python scripts/convert_dota_hbb_to_locany.py \
  --dota-root /data/data/738c885b302947929603f33110544338/道路检测数据集/DOTA-v1.0 \
  --output-root /data/locate_anything_sat/Embodied/data/dota_v1_hbb_448_mix_v1 \
  --reuse-tiles-from /data/locate_anything_sat/Embodied/data/dota_v1_hbb_448 \
  --version v1.0 --splits train val --tile-size 448 \
  --max-boxes-per-sample 30 --recipe-name dota_v1_hbb_448_mix_v1
```
转换要点：
- DOTA OBB 行格式为 `x1 y1 ... x4 y4 class difficult`；第一阶段取外接水平框 HBB/AABB。
- 框格式统一为 `(x1,y1,x2,y2)` 水平框 → `<x1><y1><x2><y2>` token，坐标归一化到 `[0,1000]`。
- 训练 tile 默认 `448×448`、`overlap=0`、`min_visibility=0.5`、`max_boxes_per_sample=30`；448 是 `14px patch × 2×2 merge = 28` 的整数倍，避免训练时额外 padding/resize；超 cap 时复用同一 tile 拆成多条 JSONL 样本。
- 默认保留 `difficult=1` 样本用于训练；评估阶段再排除 difficult。
- 类别名默认保留 DOTA 原名（如 `small-vehicle`），保持 prompt 与 `<ref>` 一致。
- 多目标 prompt 用 `</c>` 拼接当前样本中的类别。
- **任务混合（默认开）**：每个非空 tile 抽 `--samples-per-tile 6` 条，权重 T1 全类检测 0.50 / T2 单类 0.15 / T3 类子集 0.10 / T4 纯负样本 0.15 / T5 指代表达 0.10。T4 从 split 全类集合取“保证不在该 tile 内”的类，答案 `<box>none</box>`；T5 短语如 `the leftmost small-vehicle`，prompt 与 `LocateAnythingWorker.ground_single` 逐字符一致。`--no-task-mix` 恢复旧行为（每 tile 每 chunk 一条全类样本）。
- **train 9:1 内部验证**：按源图（非 tile）随机划分，输出 `*_train_mix_*.jsonl` 与 `*_internal_val_mix_*.jsonl`。
- **官方 val 双渲染**：`*_test_t1_*.jsonl`（纯全类 T1，供 `eval_baseline_dota.py` 与 val 监控使用，**勿指向 mix 文件**）与 `*_test_mix_*.jsonl`（与 train 相同的混合发射）。
- **`--reuse-tiles-from DIR`**：复用既有输出的 `tiles/`（缺失 tile 记入 `tiles_missing` 并跳过），新输出目录的 `tiles` 为指向 `DIR/tiles` 的符号链接；旧 `data/dota_v1_hbb_448/` 保持原样不动。
- **几何增强 JSONL（`scripts/augment_locany_jsonl.py`）**：对既有 0° mix JSONL 离线生成 `*_rot90/_rot180/_rot270/_hflip.jsonl`（token 空间改写 `<box>` 坐标，像素不动），训练 recipe 用五个 key（`rotate`/`hflip` 字段）在加载时对图像做对应 PIL 变换；T5 指代样本（leftmost/…、single-instance prompt）自动丢弃；配合 `color_jitter: true` 做亮度/对比度/饱和度抖动。评估/Streamlit 仍只用 0° 文件。

**小目标专用参考数据集**：AI-TOD（专为微小目标，8 类，目标均值 12.8px）、VisDrone（无人机视角密集小目标）、xView（超高分辨率）、DOTA（遥感旋转目标）、DIOR。优先用 AI-TOD / VisDrone 验证小目标能力。

______________________________________________________________________

## 5. 代码开发规范

### 5.1 分层与改动边界

- **模型层**（`eaglevl/model/{locany,moon_vit}/`）：PBD 核心。**默认不改**；确需改动（如小目标坐标表达）须在 `document/` 记录设计依据并评估对 PBD 解码稳定性的影响。
- **训练层**（`eaglevl/train/`）：数据混入、采样加权、增强在 `dataset.py` / `augmentation.py`；新增训练超参在 `arguments.py` 的 `ModelArguments`/`DataTrainingArguments`。
- **推理层**（`locateanything_worker.py`、`evaluation/inference_*`）：切片推理（SAHI 式）的主要改造点（见第 7 节）。
- **入口训练脚本** `locany_finetune_magi_stream.py`：**禁止破坏性改动**，新增逻辑优先通过 arguments + 现有回调扩展。

### 5.2 新增超参的流程

1. `eaglevl/train/arguments.py` 的 `ModelArguments`/`DataTrainingArguments` 加字段（带 default + help）
2. 在 `locany_finetune_magi_stream.py` 消费该字段
3. 同步更新本文件第 8 节参数表与 `document/TRAINING.md`
4. shell 脚本（`shell/`）暴露对应环境变量

### 5.3 包名约定

代码内 import 统一 `from eaglevl.xxx import ...`（包名 `eaglevl`，安装名 `locate_anything`）。**禁止** `from model.xxx` 这类裸导入。

______________________________________________________________________

## 6. 日志规范

- 训练默认 `--report_to tensorboard`（shell 脚本）；tensorboard 日志在 `output_dir` 下。
- 若改用 WandB：`--report_to wandb`，shell 脚本已设 `WANDB_PROJECT`/`WANDB_RUN_ID` 环境变量可覆盖。Run 命名建议 `rs_<数据集>_<lr>_<模式>_<版本>`，如 `rs_aitod_2e5_lora_v1`。
- 关键监控量：train/loss、学习率、packing efficiency、显存。**遥感实验额外记录**：小目标分桶 mAP、训练/评估数据集构成（recipe repeat_time）。

______________________________________________________________________

## 7. 小目标检测增强策略（项目核心）

> 遥感小目标的根本矛盾：**目标像素少** + **坐标量化到 `[0,1000]` 整数**，导致小框可用坐标“比特”极少，定位精度受限。增强围绕「让小目标在输入与坐标空间都变“大”」展开。

### 7.1 切片推理（Tiled / Sliced Inference）— 首选

- **原理**：把大图（如 4000×4000）切成重叠小块（如 1000×1000），每块独立检测，再按偏移拼回 + NMS 去重。同一 20px 目标：在 4000px 全图中占 5/1000 坐标单位；在 1000px 切片中占 20/1000 单位——**有效坐标精度提升 ~4×**。
- **落地**：改造 `locateanything_worker.py` / `evaluation/inference_detection_ddp.py` 增加 slice 流程；复用 `worker.detect_batch(...)` 批量处理各切片。
- **参数**：切片尺寸、重叠率（stride，如 0.2）、合并 IoM（intersection-over-min）阈值。
- **<TODO: 实现 `slice_infer.py` 工具，支持单图切片→批量推理→坐标还原→NMS 合并>**。

### 7.2 高分辨率微调

- 微调阶段用较高输入分辨率，使 MoonViT 对小目标产生更多视觉 token，提升 grounding 学习信号。
- 受限于序列长度预算（`--max_num_tokens`）；非 Hopper/Blackwell（SDPA ~4K）需配合切片训练或降 batch。

### 7.3 视觉提示微调（Visual Prompt FT）

- LocateAnything 原生支持：recipe 中 `visual_prompt=true` 的单类别检测样本，自动把正样本裁剪为视觉提示（裁剪图作为额外 image placeholder，源图仍是目标图）。
- **对小目标意义**：用目标裁剪块作为 query，帮助模型学习 RS 小目标的纹理/形状先验，缓解文本类别歧义。
- 启动脚本即 `shell/locate-anything-lora-visual-prompt.sh`。

### 7.4 多尺度数据增强

- recipe 中 `data_augment=true` 触发 resize 增强（`eaglevl/train/augmentation.py`），提升尺度鲁棒性，**对小目标尤其有效**（模拟不同航高/GSD）。448 tile 训练时保持 false。
- **已实现几何 + 颜色增强**：`augmentation.py` 提供 `transform_box`/`apply_image_geometry`（rot90/rot180/rot270/hflip，token 空间与 PIL 变换逐像素对齐已验证）与 `apply_color_jitter`（亮度/对比度/饱和度）。流程：`scripts/augment_locany_jsonl.py` 离线生成四个几何 JSONL → recipe 五 key（`rotate`/`hflip`）加载时变换图像 → `color_jitter: true` 颜色抖动（按 idx 播种）。T5 指代样本不参与几何增强（位置词失效，rewriter 自动丢弃）。

### 7.5 小目标加权采样 [INFERENCE]

- 默认数据流按 repeat_time 等权采样。建议对“小目标密集”样本上采样（如按图中目标面积中位数排序，oversample 小目标图）。**需在 `dataset.py` 增加自定义采样逻辑**。

### 7.6 防遗忘数据混合

- 纯 RS 微调易灾难性遗忘通用 grounding。Recipe 中保留少量原 LocateAnything-Data（`repeat_time<1`，如 0.3）做正则，平衡域适配与通用能力。

### 7.7 坐标精度边界 [INFERENCE]

- `[0,1000]` 量化下，最小可分辨框宽 ≈ 图像边长 / 1000。4000px 图 → 4px/单位。**经验阈值**：目标边长 < ~20px（全图）时定位误差显著，应触发切片。评估时按 COCO 面积分桶（small<32²/medium<96²/large）单独报告 mAP，量化小目标增益。

______________________________________________________________________

## 8. 训练/数据参数表（改动须同步本表）

| 来源 | 参数 | 默认 | 说明 |
|---|---|---|---|
| `locany_finetune_magi_stream.py` | `--model_name_or_path` | — | 基座/微调 checkpoint 路径 |
| 同上 | `--meta_path` | — | Recipe JSON 路径 |
| 同上 | `--block_size` | 6 | PBD 块长（MTP 并行解码） |
| 同上 | `--causal_attn` | False | MTP 训练须 False |
| 同上 | `--attn_implementation` | magi | magi(Hopper/Blackwell,32K+) / sdpa(任意,~4K) |
| 同上 | `--freeze_llm`/`--freeze_backbone`/`--freeze_mlp` | False | 冻结开关；LoRA 模式 freeze LLM+backbone |
| 同上 | `--use_llm_lora` / `--use_backbone_lora` | 0 | LoRA 秩（0=关闭）；本项目 LLM LoRA=64 |
| 同上 | `--max_seq_length` | 2048 | 最大序列长度（遥感长序列设 16384） |
| 同上 | `--max_num_tokens` | 36864 | 每打包 batch token 预算 |
| 同上 | `--max_num_tokens_per_sample` | 32768 | 超此长度样本被丢弃 |
| 同上 | `--packing_buffer_size` | 32 | 流式打包缓冲（效率 vs 显存） |
| 同上 | `--learning_rate` | 2e-5 | 峰值学习率 |
| 同上 | `--grad_checkpoint` | False | 梯度检查点省显存（遥感长序列开） |
| 同上 | `--vision_select_layer` | -1 | ViT 取特征层（-1=最后） |
| 同上 | `--mlp_connector_layers` | 2 | MLP 投影器层数 |
| 同上 | `--vision_merge_kernel_size` | None | 覆盖 MoonViT patch merge；默认 None 保持原始 `2×2`，DOTA/RS 微调优先保持默认以复用原始视觉-MLP 分布；仅实验性消融才设 `1,1` |
| 同上 | `--use_lda` | None | 残差 Local Detail Adapter（MoonViT encoder 输出与 patch merge 之间，`F' = F + γ·ΔF`，γ 初始 0 故首前向为恒等）；`'true'`/`'false'`，未设置沿用 checkpoint/config（基座为关）；序列化进 checkpoint `config.json` 的 `vision_config`；开启时即使 `freeze_backbone` 也会解冻 LDA |
| 同上 | `--lda_bottleneck_dim` | 128 | LDA 瓶颈通道数（仅 `--use_lda true` 生效） |
| Recipe JSON | `repeat_time` | 1.0 | 采样权重（≥1 重复，<1 下采样） |
| Recipe JSON | `data_augment` | false | resize 多尺度增强（小目标推荐开） |
| Recipe JSON | `rotate` / `hflip` | 0 / false | 加载时对图像做 90° 旋转/水平翻转（配 `augment_locany_jsonl.py` 离线 JSONL；与 `visual_prompt` 互斥） |
| Recipe JSON | `color_jitter` | false | 亮度/对比度/饱和度抖动（PIL ImageEnhance，strength 固定 0.2，按样本 idx 播种可复现） |
| Recipe JSON | `visual_prompt` | false | 视觉提示微调（裁剪作 query） |
| Recipe JSON | `geometry` | `hbb` | `obb` selects 7-token MTP and `<obb></obb>` tokens; CLI `--block_size` stays the HBB window |
| `deepspeed_configs/` | config | — | zero_stage1（通信省）/ zero_stage2（显存省，推荐 full SFT） |

______________________________________________________________________

## 9. 测试与验证

- 本项目无独立单元测试套件（属微调项目）。验证以**端到端 smoke + 评估**为主：
  1. **数据转换校验**：转换脚本输出后，随机抽 N 条 JSONL，解析 `<box>` token 还原像素框，与原标注框 IoU ≥ 0.99（验证坐标量化无损）。
  2. **训练 smoke**：`--max_steps 10 --max_num_tokens 4096` 跑通前向/反向/存盘/resume，确认 loss 有限、不 OOM。
  3. **推理 smoke**：`LocateAnythingWorker` 加载 checkpoint 对单张 RS 图 detect，确认输出 token 格式合法。
  4. **小目标评估**：按面积分桶 mAP，对比基座 vs 微调 checkpoint，量化 small 桶增益。
- **resume 一致性**：LocateAnything 流式打包支持 bit-wise 一致恢复（dataloader_state_rank{rank}.pt）；中断后续跑应自动从 checkpoint 恢复，无需手动指定。

______________________________________________________________________

## 10. Git 提交规范

- commit message 用 **gitmoji** 前缀：`✨ 新功能` / `🐛 修复` / `♻️ 重构` / `📝 文档` / `🔧 配置` / `🗃️ 数据转换` / `🚀 性能` / `🔥 删除`。
- 每个 commit 聚焦一个逻辑单元（数据转换 / 训练超参 / 切片推理 / 评估 各自分开）。
- **禁止提交**：`.env`、HF_TOKEN、模型权重、`work_dirs/` checkpoint、大体积遥感原图（数据集放数据盘，gitignore 排除）。
- 改动 `arguments.py` / recipe / shell 脚本时，同 commit 更新本文件第 8 节与 `document/`。

______________________________________________________________________

## 11. 关键技术备忘

### 11.1 模型加载与 backbone

- 基座：`nvidia/LocateAnything-3B`（Qwen2.5-3B-Instruct LLM + MoonViT-SO-400M 视觉编码器，max length 25K）。
- 模型类：`LocateAnythingForConditionalGeneration`（`eaglevl/model/locany/modeling_locateanything.py`），Processor：`LocateAnythingProcessor`（`eaglevl/utils/locany/`）。
- 推理模式：Fast(MTP)/Slow(NTP)/Hybrid（MTP 默认 + 格式异常时 NTP 回退）。

### 11.2 坐标系统（小目标关键约束）

- 归一化整数 `[0,1000]`，token `<0>`~`<1000>`；像素↔token：`round(p / dim * 1000)`。
- 框：`<ref>label</ref><box><x1><y1><x2><y2></box>`；点：`<box><x><y></box>`；无：`<box>none</box>`。
- **小目标坐标“比特”极少 → 切片是提升精度的根本手段**（见第 7 节）。

### 11.3 流式打包（Streaming Packing）

- 在线打包变长序列，Best-Fit + Big-Rocks-First 填充，`--per_device_train_batch_size` 固定 1，由 `--max_num_tokens` 控制有效 batch。
- 算法细节见 `document/STREAMING_PACKING.md`；OOM 时降 `max_num_tokens` / `packing_buffer_size`。

### 11.4 兄弟目录关系

- `Embodied/`（本项目）与 `Eagle/`、`Eagle2_5/` **零代码依赖**（已验证：全目录无 `../Eagle`/`Eagle2_5` 引用，训练脚本仅 import `eaglevl.*` + 第三方）。它们是 EAGLE 单仓库的兄弟模型发布，可忽略；保留利于 `git pull` 跟踪 LocateAnything 上游更新。

______________________________________________________________________

## 12. 关键约束总结

| 约束 | 规则 |
|---|---|
| 工作根 | `Embodied/`；所有命令、`pip install -e .` 在此执行 |
| 基座架构 | MoonViT + Qwen2.5 + PBD，**不替换**；微调只调权重/数据/推理 |
| 坐标系统 | `[0,1000]` 归一化整数（硬约束）；小目标靠切片提升有效精度 |
| 训练入口 | 统一 `locany_finetune_magi_stream.py`；LoRA（freeze+use_*_lora）/ Full（全解冻） |
| GPU | 仅 NVIDIA；Hopper/Blackwell 用 magi(32K+)，其余 sdpa(~4K)；MacBook 仅开发 |
| 项目核心 | **遥感域适配 + 小目标检测增强**；评估按面积分桶 mAP 量化小目标增益 |
| 包名 | import 统一 `eaglevl.*`；禁止裸 `model.*` |
| 日志 | tensorboard（默认）/ wandb；记录小目标分桶 mAP + recipe 构成 |
| 数据 | JSONL + Recipe JSON；遥感集需转 COCO/DOTA→LocateAnything 格式 |
| 防遗忘 | Recipe 混入少量原 LocateAnything-Data（repeat_time<1） |
| Git | gitmoji；禁提交权重/数据/token；改 arguments 须同步参数表 |
| 兄弟目录 | Eagle/Eagle2_5 无依赖，可忽略，保留以便上游同步 |
