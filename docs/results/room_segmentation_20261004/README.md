# 5 场景共享观测房间分割比较

5 个不同训练 calibration 场景，每场景随机 1 个 HM3D v1 ObjectNav episode；随机种子 20261004。
当前场景资产为 HM3D-0.2。两种方法均使用完整的同一组 880 帧 RGB-D 和仿真位姿，10 个方法回合全部完成。

结果见 [episodes.csv](episodes.csv)、[summary.json](summary.json) 和 [完整说明](../../room_segmentation_comparison.md)。
并排图保留在本地 `runs/room_compare_20261004_review/final_comparison.png` 及说明所列服务器目录，图像不提交到 Git。

Active 保留官方 DETR、门过滤、MapBuilder 和 Topomap；被动回放通过官方 update_topo 按已观测组件更新当前房间。
OccuSG 保留深度投影、OctoMap、mapconversion、DuDe 和区域跟踪，独立构建副本仅增加可选可靠点云订阅参数。
Active 原生建图范围 3m，OccuSG 点云 10m / OctoMap 8m；地图和自由区掩码不同。

区域数和各自自由区标注比例均为描述性指标，没有可验证的精确房间边界 GT，准确率项为空。
OccuSG 分区更多；LcAd9dhvVwh 中 OccuSG 标注范围较少，j6fHrce9pHR 中 Active 最终为零区域。
全部正式 Active 回合都有门掩码触发；不能将最后一个场景的零区域等同于零检测触发。
耗时包含不同管线开销及并行运行影响，不作为公平速度基准。

`provenance/` 保存源数据与资产哈希、880 帧文件哈希、权重哈希、方法源码哈希、构建修改、编译器与节点二进制哈希、
配置、环境和服务器单元测试。早期失败及排除结果见 [诊断记录](../../room_segmentation_failed_attempts.json)。
