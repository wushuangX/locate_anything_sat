# 新 eval 服务器迁移记录

2026-10-10。用户提供新服务器并授权沿用此前 eval 部署、checkpoint/数据直传及 8001 端口可视化流程。

- SSH：`ssh -p 24047 root@175.155.64.171`，本机 SSH key 登录。
- GPU：单张 RTX 3090 24GB，驱动 580.173.02；数据盘初始余量约 402GB。
- 工作根：`/data/locate_anything_sat/Embodied`。
- 独立 `.venv` 使用预装 Python 3.12 / torch 2.12.1+cu130 / torchvision 0.27.1+cu130；未修改基础环境或其他项目。
- 固定推理依赖：transformers 4.57.1、tokenizers 0.22.0、peft 0.12.0、accelerate 1.5.2、numpy 1.26.4、shapely 2.1.2、Streamlit 1.65.0。
- 使用本机代码快照快速部署（沿用用户此前对 SCP 的明确授权）；不是 Git commit 部署。代码版本仍需后续统一提交追溯。
- 迁移 checkpoint：`/data/checkpoints/checkpoint-50500`，来源原训练 run 的完整 step50500；推理文件 7,907,277,492 bytes。原训练端在 `/data/eval_transfer/checkpoint-50500` 固定硬链接快照，防止 checkpoint 自动清理影响传输。
- 复用此前 `/data/dota_multitask_eval_subset`，train/val 源图隔离；2,200 条训练查询、3,330 条内部 val 查询，约 848 MiB tar。
- 模型/processor/worker 导入和 GPU BF16 SDPA 运算已通过；迁移的实际加载与生成结果另行记录。
- Streamlit 监听 `0.0.0.0:8001`；默认 checkpoint 与离线结果目录由 `MODEL_PATH` / `EVAL_RESULTS` 指定。
- 记录中不保存服务器密码。

进入环境：

```bash
cd /data/locate_anything_sat/Embodied
source .venv/bin/activate
```

Fast eval：

```bash
MODEL_PATH=/data/checkpoints/checkpoint-50500 \
OUT=/data/eval_results/ckpt50500_val_fast \
bash scripts/run_multitask_subset_eval.sh
```

可视化：

```bash
MODEL_PATH=/data/checkpoints/checkpoint-50500 \
EVAL_RESULTS=/data/eval_results/ckpt50500_val_fast \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/streamlit run scripts/multitask_viz_app.py \
  --server.address 0.0.0.0 --server.port 8001 --server.headless true
```

只使用本次创建的 `/data/locate_anything_sat`、`/data/checkpoints`、`/data/dota_multitask_eval_subset`、`/data/eval_results`；其他既有 `/data` 项目未修改。原训练进程和旧 eval 服务器均未停止或修改。

## 2026-10-11 最新 checkpoint 直传

- 12:33 训练日志推进至约 70135/100000，最新完整 checkpoint 为 70000。窗口 69001–70000 的平均 loss 为 0.1512592；旧 HBB 同窗口为 0.4980284。任务与协议不同，不能将 loss 差异解释为精度差异。
- 训练端 `nvidia-smi` 返回 NVML Unknown Error；训练日志仍持续更新，rank 0 显存记录约 17676 MB。未重启或更改训练进程。
- 固定推理快照 `/data/eval_transfer/checkpoint-70000`，不包含 optimizer、训练状态与 AppleDouble 文件。传输包 `/data/eval_transfer/checkpoint-70000.tar`，7907338240 bytes，SHA256 `77ae51ca895eae92d58c12d48db325135d1f60873cf096256ec7ce27b7cbc7a2`。
- 训练端独立后台 SFTP 直传到 eval `/data/checkpoints/checkpoint-70000.tar`，日志 `/data/eval_transfer/checkpoint-70000.transfer.log`。目标接收大小与校验状态是最终判断依据，不仅依据进程或上传端退出码。
- eval 自动等待完整接收、校验传输包、解压至 `/data/checkpoints/checkpoint-70000`，再核对全部推理文件。状态 `/data/checkpoints/checkpoint-70000.transfer-status.json`；只有 `stage=ready` 且 `all_files_verified=true` 才可使用。
- 旧 50500 checkpoint 保留；本次未同步仓库代码。checkpoint 自包含运行时代码属于模型产物，未覆盖仓库文件。
- 先前数据包两端 SHA256 已一致：`dbd12ba1a3b094b1528f9f86a5c41dec5cbb94de20c78dcf49a01ae259b8933d`。此记录不表示数据已解压或评估已启动。

## 2026-10-11 评估与可视化启动

- 70000 传输包 SHA256 和解压后全部推理文件校验通过，状态为 `ready`。代表性数据包已解压到 `/data/dota_multitask_eval_subset`。
- 评估代码经 Git 提交并推送，独立部署目录 `/data/locany_eval_git`，运行 commit `3e5c4b69b6148a63e6a232c67228831342fa2661`。启动前及启动后确认 Git 工作区干净；旧快照目录和训练 checkout 均未更新。复用旧目录的 `.venv`，通过 Git 本地 exclude 排除环境链接。
- checkpoint 自包含的 runtime 由模型目录加载；Git 部署的 worker、codec、评估与 UI 负责调用和解析。checkpoint 校验清单记录实际 runtime 内容，不能用 Git SHA 单独代表训练时的未提交 runtime。
- 22 条 smoke 查询（每个 T1–T11 一正一负）已完成并生成 `DONE`，目录 `/data/eval_results/ckpt70000_val_fast_smoke_20261011`。模型加载和推理成功，不将 smoke 结果当完整性能指标。
- 约 13:27 启动正式 Fast val 子集评估，全部 3330 条查询，进程启动 PID 82389，输出 `/data/eval_results/ckpt70000_val_fast_20261011`。已确认至少 20 条预测落盘；此处记录的是启动时状态。
- Streamlit 启动 PID 82219，监听 `0.0.0.0:8001`，默认 checkpoint 为 70000，离线结果指向上述正式评估目录。健康接口返回 `ok`；AppTest 检查 train/val × T1–T11 共 22 个页面全部通过。
- 批量评估期间可浏览已落盘结果；交互 GPU 推理默认禁用，可通过页面并行选项主动启用，会加载第二份模型并检查空闲显存。
- 当前所有 launcher 均从 `/data/locany_eval_git/Embodied` 执行；前文旧快照目录命令仅保留为历史部署记录。日志为输出目录中的 `eval.log` / `streamlit.log`，启动记录为 `launch.json` / `streamlit_launch.json`。
- 本轮为代表性 val 子集评估，未运行完整 121200 条 val，也未使用 test。最终指标在 `metrics.json`，完成标记为 `DONE`。
