# HM3D v1 val：5 场景同观测分割对比

用户要求从 HM3D v1 val 随机抽取 5 个不同场景，每场景 1 个 episode，分别运行 OccuSG 和 Active room segmentation。沿用训练集对比的固定方法、权重和参数，不使用 val 结果调参，也不按运行结果替换样本。

抽样源：`/home/zyq/vlfm/data/datasets/objectnav/hm3d/v1/val/content/*.json.gz`。在全部 val 场景文件中均匀抽取 5 个，再在每个选中文件中均匀抽取一行，种子 `20261004`。选择文件 SHA256：`fb89209efe8f42d853220c1826eb16d8dca8f6656ad9700ae757931e153313c8`。

| 场景 | 源 episode ID | 源文件行号（从 0 开始） | 目标 |
| --- | --- | ---: | --- |
| 5cdEh9F2hJL | 0 | 66 | toilet |
| Nfvxx8J5NCo | 0 | 24 | chair |
| XB4GS9ShBRE | 0 | 95 | toilet |
| bxsVRursffK | 0 | 90 | chair |
| mv2HUxq3B53 | 0 | 85 | toilet |

原始 episode ID 不能单独标识样本；记录文件哈希、行号和 episode 内容哈希。实际场景资源版本由 Habitat 审计确认，episode 版本与 mesh 版本分别记录。

固定 VLFM baseline 录制每条完整轨迹，到 STOP 或 500 步终止，不填充静止帧。两种方法获得完全相同的 RGB、metric depth 和位姿。Active 保留每 10 帧更新及被动房间状态适配；OccuSG 保留原生点云、OctoMap、mapconversion 和 DuDe，以及显式可靠传输。参数细节和适配范围见 [训练集协议](room_segmentation_comparison.md)。

采样与录制 worktree：`/home/zyq/AV-Nav-worktrees/room-val-7b876c4`；回放与方法 worktree：`/home/zyq/AV-Nav-worktrees/room-val-0029f40`。全部使用独立运行目录和固定提交，不更新正在运行的 checkout。输入回放完成后逐帧核对 RGB 和深度，并检查楼层变化；原适配器对超过 0.3m 楼层变化明确拒绝，不将不同楼层混成同一张房间图。

当前状态：5 个样本已固定，资源审计和服务器 47 项测试完成，完整轨迹录制正在运行。短 smoke、失败和零输出均单独报告。无可靠 room footprint GT 时仅报告并排图、原生区域数和各自自由区标注比例，不声称准确率。
