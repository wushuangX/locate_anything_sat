# LocateAnything 模型结构理解与文档核验

> 本文档整理当前仓库中已有 Markdown 对 LocateAnything 模型结构的描述，并基于源码核验其正确性。
>
> 结论：**当前仓库没有一篇完整系统讲 LocateAnything 模型结构的 md。** 现有文档有高层介绍和零散线索，方向基本正确，但不足以完整理解模型内部结构；需要结合源码。

______________________________________________________________________

## 1. 当前仓库已有文档在哪里

| 文件 | 是否讲模型结构 | 判断 |
|---|---:|---|
| `Embodied/README.md` | **有，高层最好** | `## Method`、`Parallel Box Decoding`、L82–85 明确写 Moon-ViT + Qwen2.5 + MLP projector；L90–107 写 fast/slow/hybrid 与 NTP 回退；L252–258 写输出格式。**正确但很概略。** |
| `README.md` | 很少 | L161–163 的 model table 写 LocateAnything-3B = Qwen2.5-3B-Instruct + MoonViT-SO-400M + 25K；L78 写 PBD 高层。不是架构文档。 |
| `AGENTS.md` | 有开发视角摘要 | L18 写 MoonViT + Qwen2.5 + MLP + PBD；L19 写 `[0,1000]` 坐标；L31–38 指向核心源码；L393–401 写模型加载、模式、坐标。**实用，但不是结构解释。** |
| `SMALL_OBJECT_DESIGN.md` | 有局部深入 | §1.1 深入讲 MoonViT patch/merge 与小目标瓶颈。**不是完整模型结构文档。** |
| `Embodied/document/DATA_PREPARATION.md` | 数据格式 | L112–144 讲 `<ref>` / `<box>` / `<0>`~`<1000>`。正确，但只是数据/输出格式。 |
| `Embodied/document/TRAINING.md` | 训练参数 | 只零散提 `block_size`、attention backend、MLP layers。不是架构文档。 |
| `Embodied/document/RESULTS.md` | 结果/消融 | 有 PBD 消融结论，不讲结构。 |

另外，`Embodied/README.md` 嵌入了 `assets/images/method_overview.jpg` 架构图，但当前 checkout 里这个文件是 Git LFS pointer，无法从本地检查图片细节。

______________________________________________________________________

## 2. 这些文档理解是否正确

### 2.1 总体判断

**高层判断：基本正确，但不完整。没有发现根本性误解。**

`Embodied/README.md` 和 `AGENTS.md` 对核心结构的理解是对的：

- LocateAnything = **MoonViT 视觉编码器 + Qwen2/Qwen3 causal LLM + MLP projector + PBD/MTP 解码机制**。
- 坐标系统 = 归一化整数 `[0,1000]`，用 `<0>`~`<1000>` 坐标 token 表达。
- 推理模式 = fast / slow / hybrid。

源码核验位置：

| 内容 | 源码位置 |
|---|---|
| 顶层模型 | `Embodied/eaglevl/model/locany/modeling_locateanything.py:82-126` |
| 配置组合 | `Embodied/eaglevl/model/locany/configuration_locateanything.py:44-89` |
| MoonViT | `Embodied/eaglevl/model/moon_vit/modeling_vit.py:562-611` |
| MLP projector | `Embodied/eaglevl/model/locany/modeling_locateanything.py:121-126` |
| image token 替换 | `Embodied/eaglevl/model/locany/modeling_locateanything.py:197-243` |
| Qwen2 自定义 attention/mask | `Embodied/eaglevl/model/locany/modeling_qwen2.py:1085-1252` |
| inference fast/slow/hybrid | `Embodied/eaglevl/utils/locany/modeling_locateanything.py:349-512` |

### 2.2 不完整的部分

现有 md 没有系统解释这些关键点：

1. **MoonViT 细节**
   - patch size = 14
   - 27 层 transformer encoder
   - hidden size = 1152
   - 2×2 patch merger
   - 无 CLS token
   - 无 pixel shuffle

2. **MLP projector 结构**

   文档只说 “MLP projector”，源码实际是：

   ```python
   LayerNorm(vit_hidden_size * 4)
   Linear(vit_hidden_size * 4 -> llm_hidden_size)
   GELU
   Linear(llm_hidden_size -> llm_hidden_size)
   ```

   位置：`Embodied/eaglevl/model/locany/modeling_locateanything.py:121-126`。

3. **image token 如何进入 LLM**

   Processor 会把 `<image-1>` 展开成：

   ```text
   <image 1><img><IMG_CONTEXT>...<IMG_CONTEXT></img>
   ```

   `<IMG_CONTEXT>` 数量 = `(patch_h * patch_w) / 4`。然后模型 forward 中把这些 token 的 embedding 替换成 MoonViT+MLP 的视觉 embedding。

   位置：
   - `Embodied/eaglevl/utils/locany/processing_locateanything.py:375-474`
   - `Embodied/eaglevl/model/locany/modeling_locateanything.py:224-228`

4. **PBD 不是普通一个 head**

   文档说“单步原子预测框”没错，但实现上是 **MTP mask block + 自定义 attention mask + 6-token block 解码**。

   - 训练时 `get_targets_flag_with_mtp` 构造 anchor + `<text_mask>` 块；
   - 推理时 fast/hybrid 模式一次看 `n_future_tokens=6` 个 logits；
   - 异常时 hybrid 会回退 autoregressive。

   位置：
   - `Embodied/eaglevl/train/locany_finetune_magi_stream.py:254-479`
   - `Embodied/eaglevl/utils/locany/generate_utils.py:276-361`
   - `Embodied/eaglevl/utils/locany/modeling_locateanything.py:415-429`

5. **Qwen2 不是原版 HF Qwen2**

   `modeling_qwen2.py` 是 fork：加入 `magi` attention、stream packing、MTP block mask、`sub_sample_lengths`、`text_mask_token_id` 等逻辑。

   位置：`Embodied/eaglevl/model/locany/modeling_qwen2.py:1114-1251`。

### 2.3 一个文档不一致点：`block_size` 默认值

`block_size` 的“默认值”在文档与实际脚本中有语义不一致：

- 源码 dataclass 裸默认：`Embodied/eaglevl/train/arguments.py:103-105` 是 `default=4`；
- 两个训练 shell 都显式传：`--block_size 6`：
  - `Embodied/shell/locate-anything-lora-visual-prompt.sh:72`
  - `Embodied/shell/locate-anything-streaming.sh:51`
- `Embodied/document/TRAINING.md:195` 写默认 4 —— **符合 CLI 源码默认**；
- `AGENTS.md:349` 写默认 6 —— **符合本项目 shell 脚本实际使用值，但不符合 dataclass 裸默认**。

建议表述：**裸 CLI 默认是 4；本项目训练脚本固定传 6，实际推荐/项目默认用 6。**

______________________________________________________________________

## 3. 源码核验后的真实模型结构

### 3.1 总体数据流

```text
PIL image + prompt
    ↓
LocateAnythingProcessor
    - image rescale / patchify
    - <image-N> 展开成多个 <IMG_CONTEXT>
    ↓
MoonViT
    - 14×14 patch embed
    - 27-layer transformer encoder
    - 2×2 patch merger → token 数 /4，channel ×4
    ↓
MLP projector
    - LN → Linear → GELU → Linear
    - MoonViT hidden*4 → Qwen hidden
    ↓
Qwen2/Qwen3 causal LM
    - 输入 token embedding 中的 <IMG_CONTEXT> 被替换成视觉 embedding
    - 使用自定义 MTP/PBD attention mask
    ↓
PBD / MTP decoding
    - fast: 6-token block 并行解码
    - slow: 普通 autoregressive
    - hybrid: 先 MTP，异常时回退 AR，再切回 MTP
    ↓
<ref>label</ref><box><x1><y1><x2><y2></box>
```

### 3.2 MoonViT 视觉编码器

源码位置：`Embodied/eaglevl/model/moon_vit/modeling_vit.py`。

关键参数：

| 参数 | 值 | 来源 |
|---|---:|---|
| patch size | 14 | `modeling_vit.py:36` |
| layers | 27 | `modeling_vit.py:40` |
| heads | 16 | `modeling_vit.py:39` |
| hidden size | 1152 | `modeling_vit.py:41` |
| intermediate size | 4304 | `modeling_vit.py:42` |
| merge kernel | 2×2 | `modeling_vit.py:43` |

执行路径：

1. `MoonVisionPatchEmbed`：Conv2d，kernel=stride=14。
   - `modeling_vit.py:243-284`
   - `modeling_vit.py:264-266`
2. `MoonVitEncoder`：transformer encoder。
   - `modeling_vit.py:582-592`
3. `patch_merger`：2×2 patch fold/concat，token 数除以 4，channel 乘以 4。
   - `modeling_vit.py:533-559`
4. forward：patch_embed → encoder → patch_merger。
   - `modeling_vit.py:595-611`

注意：该实现不是 CLS 分类式 ViT，也不是 pixel-shuffle；它保留 patch 序列，最后做 2×2 merge。

### 3.3 Image processor 与 `<IMG_CONTEXT>` 展开

源码位置：
- `Embodied/eaglevl/utils/locany/image_processing_locateanything.py`
- `Embodied/eaglevl/utils/locany/processing_locateanything.py`

图像处理：

1. 如果 patch 数超过 `in_token_limit`，resize。
2. 尺寸 pad 到 `patch_size * merge_kernel = 28` 的倍数。
3. PIL → tensor。
4. normalize。
5. patchify 成 `(num_patches, 3, 14, 14)`。
6. 返回 `pixel_values` 和 `image_grid_hws`。

关键位置：
- resize/pad：`image_processing_locateanything.py:46-71`
- patchify：`image_processing_locateanything.py:79-86`
- preprocess：`image_processing_locateanything.py:88-126`

文本替换：

- Processor 把 `<image-1>` 替换为：

  ```text
  <image 1><img><IMG_CONTEXT>...<IMG_CONTEXT></img>
  ```

- `<IMG_CONTEXT>` 数量 = `(h * w) / (2 * 2)`，其中 `h,w` 是 patch grid 高宽。

位置：`processing_locateanything.py:391-398`。

### 3.4 MLP projector

源码位置：`Embodied/eaglevl/model/locany/modeling_locateanything.py:121-126`。

结构：

```python
self.mlp1 = nn.Sequential(
    nn.LayerNorm(vit_hidden_size * 4),
    nn.Linear(vit_hidden_size * 4, llm_hidden_size),
    nn.GELU(),
    nn.Linear(llm_hidden_size, llm_hidden_size),
)
```

含义：MoonViT `patch_merger` 后每个视觉 token 的维度是 `vit_hidden_size * 4`，MLP projector 把它映射到 Qwen hidden size，使其可以直接替换 LLM 的 token embedding。

### 3.5 视觉 embedding 注入 LLM

源码位置：`Embodied/eaglevl/model/locany/modeling_locateanything.py:197-243`。

步骤：

1. `input_ids` 先通过 Qwen embedding 得到 `input_embeds`；
2. `pixel_values` 经 MoonViT 得到 `vit_embeds`；
3. `vit_embeds` 过 `mlp1`；
4. 找到 `input_ids == self.image_token_index` 的位置；
5. 用 projected vision embedding 替换这些位置的原始 token embedding；
6. 替换后的 `inputs_embeds` 送入 Qwen2/Qwen3。

关键代码：

```python
selected = (input_ids == self.image_token_index)
input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds
```

位置：`modeling_locateanything.py:224-228`。

### 3.6 Qwen2/Qwen3 LLM 与自定义 attention

LocateAnything 支持 Qwen2 或 Qwen3：

- `configuration_locateanything.py:74-83`
- `modeling_locateanything.py:110-115`

当前主路径是 Qwen2 fork。它不是纯 HF 原版 Qwen2，关键自定义在 attention mask 和 MTP/PBD：

- `block_size`
- `causal_attn`
- `text_mask_token_id`
- `is_packing_mode`
- `sub_sample_lengths`
- MagiAttention range-plan
- SDPA 4D MTP packing mask

关键位置：`Embodied/eaglevl/model/locany/modeling_qwen2.py:1114-1251`。

attention 后端：

| 后端 | 作用 |
|---|---|
| `magi` | MagiAttention，支持长上下文和 range-based sparse attention plan |
| `sdpa` | PyTorch SDPA，支持 MTP 4D mask，但长序列能力弱 |
| `flash_attention_2` | 视觉端常用；文本端不支持 stream packing MTP masks |
| `eager` | fallback/调试路径 |

### 3.7 PBD / MTP 训练机制

训练核心在 `get_targets_flag_with_mtp`：

- 位置：`Embodied/eaglevl/train/locany_finetune_magi_stream.py:254-479`

思想：把 assistant 输出中的可监督 token 构造成 MTP block。每个 block 形如：

```text
输入:  [anchor_token, <text_mask>, <text_mask>, ...]
目标:  [真实 token1, 真实 token2, 真实 token3, ...]
```

检测/grounding 分支会尽量让 `<box> x1 y1 x2 y2 </box>` 这类结构作为一个 box-aligned atomic unit 被并行预测。

### 3.8 PBD / MTP 推理机制

推理版模型在：`Embodied/eaglevl/utils/locany/modeling_locateanything.py`。

关键模式：`utils/locany/modeling_locateanything.py:349-512`。

| 模式 | 行为 |
|---|---|
| `fast` | MTP only，从不回退 AR，最快但稳定性最低 |
| `slow` | AR only，普通自回归，最稳但慢 |
| `hybrid` | 默认：先 MTP，遇到格式异常/不可靠 box 时切到 AR，box end 后再切回 MTP |

MTP 解码核心：

- `sample_tokens`：`generate_utils.py:135-190`
- `decode_bbox_avg`：`generate_utils.py:276-361`
- `handle_pattern`：`generate_utils.py:408-486`

`n_future_tokens=6`，对应典型 box：

```text
<box> <x1> <y1> <x2> <y2> </box>
```

源码注释里 `handle_pattern` 写成 `x1 x2 y1 y2`，但输出文档统一是 `<x1><y1><x2><y2>`；后续设计时应以数据格式和评估逻辑为准，必要时再核生成输出解析代码。

### 3.9 特殊 token

训练常量：`Embodied/eaglevl/train/constants.py`。

| Token | 作用 |
|---|---|
| `<IMG_CONTEXT>` | 图像 token placeholder，后续被视觉 embedding 替换 |
| `<img>` / `</img>` | 图像 span 边界 |
| `<ref>` / `</ref>` | 类别/文本引用 span |
| `<box>` / `</box>` | box/point span |
| `<text_mask>` | MTP mask token |
| `<null>` | MTP block filler |
| `</c>` | 多类别分隔符 |
| `<0>`~`<1000>` | 坐标 token，归一化整数坐标 |

训练时加入 tokenizer：`Embodied/eaglevl/train/locany_finetune_magi_stream.py:1304-1328`。

______________________________________________________________________

## 4. 最终判断

1. **已有 md 能帮助入门，但不能满足“详细理解模型结构”。**
2. **最接近架构文档的是 `Embodied/README.md`，但它只到高层 method；`AGENTS.md` 是开发指南；`SMALL_OBJECT_DESIGN.md` 是小目标瓶颈分析。**
3. **现有理解总体正确，没有把架构方向讲反。**
4. **主要问题是缺失内部细节和一个 `block_size` 默认值语义不一致。**
5. 如果后续要做 OBB、小目标 coarse-to-fine、坐标表达扩展，建议先补一份正式 `ARCHITECTURE.md`，把：

```text
image/text input → processor → MoonViT → MLP → Qwen2 custom attention → PBD/MTP train/infer → output tokens
```

这条路径画清楚，并明确每个模块的代码入口。
