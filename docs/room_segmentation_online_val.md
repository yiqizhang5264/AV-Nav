# HM3D v1 val：评估过程中在线增量房间分割

用户要求在 VLFM 评估过程中，依据当前场景观测更新 OccuSG 与 Active room segmentation，保持 VLFM 导航策略，分割结果不参与动作选择。此前的离线回放结果不能当作此在线实验结果。

沿用固定种子 20261004 抽取的 5 个不同 val 场景，每场景一个原始 episode；没有重新抽样。原始 ID 均为 `0`，身份由场景、源文件 SHA256、源行号、episode 内容 SHA256 联合确定。

| 场景路径 | 原始 episode ID | 源文件行号（从 0 开始） | 目标 |
| --- | --- | ---: | --- |
| hm3d/val/00853-5cdEh9F2hJL/5cdEh9F2hJL.basis.glb | 0 | 66 | toilet |
| hm3d/val/00880-Nfvxx8J5NCo/Nfvxx8J5NCo.basis.glb | 0 | 24 | chair |
| hm3d/val/00878-XB4GS9ShBRE/XB4GS9ShBRE.basis.glb | 0 | 95 | toilet |
| hm3d/val/00873-bxsVRursffK/bxsVRursffK.basis.glb | 0 | 90 | chair |
| hm3d/val/00876-mv2HUxq3B53/mv2HUxq3B53.basis.glb | 0 | 85 | toilet |

数据源 `/home/zyq/vlfm/data/datasets/objectnav/hm3d/v1/val/content/*.json.gz`，固定五场景样本 SHA256 为 `fb89209efe8f42d853220c1826eb16d8dca8f6656ad9700ae757931e153313c8`。场景资源实测来自 HM3D-0.2 val，episode 版本和 mesh 版本分别记录。

每个评估步的顺序如下：VLFM 根据当前观测计算原始动作；在线适配器通过 Habitat worker 获取同一时刻的原始 RGB、metric depth、相机和代理位姿，核对与策略输入相同；仅提交当前这一帧给两个独立分割进程；等待双方确认该帧更新完成，保存即时地图；随后执行该导航动作并获取下一帧。不存在完整轨迹回放，也不读取未来帧。分割进程与 VLFM 的随机数状态隔离，官方 backbone 保持固定提交 `584ed56008754fde7997d904983607def8328322`。

OccuSG 每帧更新原生点云、OctoMap、mapconversion 和 DuDe。Active 每帧进行门检测和建图，拓扑沿用训练集设定每 10 帧以及终止帧更新。没有改变方法阈值、门权重、地图范围和被动房间状态适配。原生范围不同，已知自由面积比例的分母不同，没有可靠房间边界 GT，不声称准确率。

这是一种同步在线增量运行：仿真等两个分割进程处理完成再前进，处理速度会影响墙钟评估速度，不承诺固定实时帧率。每步有原始观测、双方更新确认、发布和完成时间；每 10 帧及终止帧保存三联图（当前 RGB、Active 地图、OccuSG 地图），每帧更新 latest 地图，`live.png` 是当时的最新可视化。

5 个 episode 分别在独立 VLFM 进程中并行运行，每个进程按同一配置独立初始化，不强制沿用以前的动作序列。每条轨迹到原始 STOP 或 500 步预算结束。执行固定在 `/home/zyq/AV-Nav-worktrees/room-online-9301f80`；各进程使用 GPU 3 和独立 ROS domain 160–164。样本子任务保留原始选择身份和父样本哈希，仅将固定 5 个 episode 拆为单回合任务。

短诊断已完成：初次 5 帧收到 10 个方法确认；最终提交上的 2 帧诊断验证每帧 `发布 < 双方完成 < env.step 完成` 且下一帧发布在上一步完成之后。诊断不作为分割效果结果。服务器 51 项单元测试全部通过，本地 50 项通过、1 项 OpenCV 项在服务器通过。

当前状态：5 条完整在线评估正在运行，代码提交 `9301f80`、收集器提交 `737aff0`。原始输入、地图和即时 PNG 保留在未跟踪 `runs/`，小型配置和结果经 origin 归档。
