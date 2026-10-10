# 多任务训练长期权重归档

当前训练不重启、不修改运行中的 Trainer。独立 `Embodied/scripts/archive_checkpoints.py` 每 15 秒检查一次 checkpoint；推理文件稳定 30 秒并通过权重分片大小、trainer step、runtime SHA256 校验后，将每 5000 步的 checkpoint 硬链接到独立目录。

- 训练目录：`/data/locate_anything_sat/Embodied/work_dirs/dota_v2_multitask_v1_full_20261008`
- 长期归档：`/data/checkpoint_archives/dota_v2_multitask_v1_full_20261008`
- 启用时额外保全 41500、42000、42500、43000；16500 原备份仍在 `/data/eval_transfer/checkpoint-16500`。
- 归档仅支持推理/评测，不包含优化器或 dataloader 恢复状态。恢复训练仍使用训练目录的完整 checkpoint。
- 每个归档权重约 7.9 GB。硬链接在创建时不复制权重，但训练删除原目录后仍占用磁盘；45000～100000 共 12 个计划归档，约 95 GB，加当前保全版本约 32 GB。
- 不修改归档文件内容；硬链接与源文件共享 inode。训练 checkpoint 文件保存后不再改写。
- `archive_complete.json` 标记已验证完成。锁文件防止同一归档目录有多个归档程序。
- 程序必须在服务器持续运行。服务器重启或程序退出后需重新启动；若某个 checkpoint 在停机期间已被训练清理，则不能补回。

脚本独立提交并经 Git remote 部署到隔离 checkout，不更新正在运行的训练代码目录。后台 PID 与日志保存在归档目录。已清理且无其他备份的早期权重不能从 loss 日志重建。

2026-10-10 已部署：脚本 commit `038e560`，隔离 checkout `/data/checkpoint_archive_tools`，后台 PID `72764`（以 `archive.pid` 和实际进程状态为准）。四个现存版本均通过校验并生成完成标记，43000 原文件与归档 inode 相同。训练启动进程 PID `26020` 保持不变。首次周期归档将是 checkpoint-45000。
