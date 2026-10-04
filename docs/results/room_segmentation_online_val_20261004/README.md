# HM3D v1 val 在线房间分割结果

5 个固定不同场景、每场景一个 episode，保持 VLFM 导航策略，在评估每步的当前 RGB-D/位姿上同步运行 Active 和 OccuSG。全部完成：1491 步、2982 次方法确认、10 个方法正常退出。全部动作序列与先前 VLFM 评估一致，未使用旧动作驱动导航。

[在线协议与结果](../../room_segmentation_online_val.md)包含 episode 身份、地图与过程视频链接。`summary.json`、`episodes.csv` 为最终原生地图描述性指标；无可靠房间边界 GT，不声称准确率。

`provenance/online_step_audit.json` 保存全部步骤的发布、分割完成和导航完成时间，验证每步及跨步的在线顺序。RGB 和归一化深度与当前策略观测逐帧一致。`run_exits.json` 保存真实运行退出状态；`baseline_action_correspondence.json` 保存完整动作对照；`input_frame_sha256.json` 保存原始输入哈希。配置、源码提交、源数据身份、模型和构建身份、ROS 参数、环境与服务器测试均归档。

`video_manifest.json` 记录在评估中生成的每张可视化图的哈希，以及最终视频哈希和步号。视频仅编码这些已有图像，未重算分割。播放速度为每张图 1 秒，图像每 10 步及终止步保存，不是墙钟实时录像。5 个视频共 150 帧，全部逐帧解码并验证本地 SHA256。

原始在线目录：`/home/zyq/AV-Nav-worktrees/room-online-9301f80/runs/online_full5`。
原始最终报告：`/home/zyq/AV-Nav-worktrees/room-online-report-737aff0/runs/online_comparison5_final`。
服务器视频：`/home/zyq/AV-Nav-worktrees/room-online-results-20261004/runs/progress_videos_v2`。
本地视频与 PNG：`runs/room_online_val_20261004_review/`。

Episode 数据来自 HM3D v1 val；实际 mesh 为 HM3D-0.2 val。原始数据、数组、图片、视频和权重未提交。短诊断及第一次视频导出失败单独记载于 [运行记录](../../room_segmentation_online_attempts.json)。
