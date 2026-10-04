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

2026-10-04 已完成全部 5 对分割运行，共享 1491 帧；10 个方法运行退出状态均为 0。每帧 RGB MAE 和归一化深度最大绝对误差均为 0。高度变化最大约 0.2021m，未超过固定适配器的 0.3m 拒绝阈值。资源审计确认实际 mesh 来自 HM3D-0.2 val，因此结果应描述为 **HM3D v1 ObjectNav val episode / HM3D-0.2 场景资源**。

| 场景 | 帧数 | VLFM 导航成功 | Active 区域数 | OccuSG 区域数 | Active 自身自由区标注比例 | OccuSG 自身自由区标注比例 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| 5cdEh9F2hJL | 177 | 是 | 2 | 8 | 67.6% | 85.8% |
| Nfvxx8J5NCo | 300 | 是 | 3 | 11 | 81.0% | 85.6% |
| XB4GS9ShBRE | 186 | 否 | 4 | 5 | 76.8% | 77.4% |
| bxsVRursffK | 328 | 否 | 1 | 6 | 57.8% | 79.1% |
| mv2HUxq3B53 | 500 | 否 | 2 | 6 | 67.0% | 86.4% |

区域数不是实际房间数。标注比例的分母是各方法原生地图的已知自由像素，两种方法的分母和建图范围不同，不是共同 GT 准确率。两者均有输出，没有零区域场景；Active 的门掩码触发帧数分别为 59、94、52、108、183，最终保留门数为 2、4、4、2、3，没有零触发场景。

完整并排图的坐标、起点、轨迹和区域颜色已检查，最终区域 ID 对应的调色板无冲突。观察到 OccuSG 在这 5 个场景都划分更多区域，几何区域通常更细；Active 的区域更粗，在 bxsVRursffK 中仅保留 1 个区域。没有可靠 room footprint GT，不能据此判断哪一种更准确，也不报告 mIoU 或 ARI。Active 是固定 ObjectNav 轨迹上的被动适配，不代表其自主探索性能。

3 条 VLFM 导航失败轨迹均保留，未重新抽样：XB4GS9ShBRE 和 mv2HUxq3B53 为 `never_saw_target_did_not_travel_stairs_feasible`，bxsVRursffK 为 `false_negative`；mv2HUxq3B53 达到 500 步预算。导航成败不等于分割成败。分割方法沿用训练集固定设置，没有根据这些 val 结果调参。

- [完整并排 PNG](../runs/room_val_20261004_review/final_comparison.png)
- [逐场景指标](results/room_segmentation_val_20261004/episodes.csv)
- [摘要与复现记录](results/room_segmentation_val_20261004/README.md)
- [启动失败和排除记录](room_segmentation_val_failed_attempts.json)

OccuSG 初次启动缺少 v2 消息包 underlay，未处理输入帧；加载原有 v2 setup 和 v3 local_setup 后，在新结果目录成功完成全部场景，没有改变分割参数。原始日志保留。内部 4 对预览仅用于对齐检查，不是最终结果。所有最终配置、提交、哈希、模型身份、构建身份和每条轨迹状态均已归档；原始 RGB-D、数组、权重和 PNG 保留在未跟踪的 runs 中。

服务器 47 项单元测试全部通过；本地 46 项通过、1 项因没有 OpenCV 跳过，相关项已在服务器通过。验证包含 val 抽样的可复现性、全场景范围、不同 scene 身份、源行保留，以及完整 Habitat 录制/回放和两种原生方法运行。CPU 运行存在并行负载，OccuSG 时间还包含 ROS 传输和等待，耗时不作为公平速度排名。
