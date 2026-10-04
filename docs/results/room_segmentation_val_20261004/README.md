# HM3D v1 val room segmentation：5 场景结果

全部 5 个随机不同场景完成 Active 和 OccuSG 对比，共享 1491 帧。固定种子 20261004、每场景 1 个 episode；冻结训练集设置，没有使用 val 调参或重选场景。10 个方法运行均正常结束。

[实验报告](../../room_segmentation_comparison_val.md)包含结果解释、导航失败、适配范围与限制。[episodes.csv](episodes.csv)保存逐场景原生地图指标，[summary.json](summary.json)保存完整配对状态。区域数和自由区标注比例不是 GT 准确率。

复现记录位于 `provenance/`：资源审计含 episode 源文件、行号与内容哈希，输入记录含每帧 NPZ 哈希，method_summaries 含轨迹导航结果和方法状态，run_identity 含实际运行提交。模型、子模块、ROS 参数、构建二进制哈希、环境和服务器 47 项测试均保留。各运行目录固定提交，服务器原有主 checkout 未更新。

实际使用 HM3D v1 ObjectNav val episode / HM3D-0.2 val mesh。RGB MAE 最大值与归一化深度最大绝对误差均为 0。所有原始输入、地图数组和图像留在未跟踪 runs 中，未提交数据或权重。

原始报告：`/home/zyq/AV-Nav-worktrees/room-val-report-427d9a4/runs/val_comparison5_final`。
本地 PNG：[完整并排图](../../../runs/room_val_20261004_review/final_comparison.png)。

服务端初次 OccuSG 启动缺少消息包 underlay，重试改正环境加载而未改参数，详细记录见 [失败记录](../../room_segmentation_val_failed_attempts.json)。OccuSG 成功运行先完成 case 00，随后其余场景并行，最终 5 例全部完整。Active 的各场景并行处理。保留这些调度脚本以审计实际命令；脚本中的初次失败步骤不能作为一键成功启动脚本使用。
