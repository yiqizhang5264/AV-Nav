# HM3D v1：OccuSG 与 Active room segmentation 小样本对比

## 当前状态

2026-10-04 用户确认先选 5 个不同场景，每场景 1 个随机 episode，先做分割对比。
源码已核查，抽样工具已实现；尚未抽取真实数据、实现两套回放适配器或运行分割实验，不能报告效果优劣。
服务器 `106.3.202.138:50001` 两次连接均在 SSH 密钥交换前被关闭，未能核查资源和运行进程。
本地上游源码仅在被忽略的 `tmp/room_upstream/` 中用于阅读，没有修改 VLFM 或主导航流程。

配置见 `configs/room_segmentation_compare.json`。这是准备配置，不是可直接启动完整比较的 runner。

## 固定方法与适配范围

- OccuSG：https://github.com/crcz25/OccuSG ，提交 `2bca2fa06af87fd9dd0542957039be8f580f0dca`。
  保留深度投影、OctoMap、mapconversion、incremental DuDe 和区域跟踪。
  需要递归检出该提交记录的子模块；不能用自行编写的形态学算法替代 DuDe 后称为 OccuSG。
  原项目使用 ROS2 Humble、Ubuntu 22.04 和 C++ 库，不能作为普通 Python 包直接导入。
  它本身不提供与 Active 相对应的自主探索策略。
- Active：https://github.com/FreeformRobotics/Active_room_segmentation ，提交
  `a941e3a5a1c8e16920a158fa0a9198c95be9c978`。
  保留 DETR 门检测、MapBuilder 深度建图、门过滤和 Topomap_construction 区域更新。
  上游面向 Gibson 和旧 Habitat API；需要显式 HM3D 回放适配器。
  `env/habitat/exploration_env.py` 的 `info['gt_map']` / `info['gt_exp']` 实际来自
  `mapper.update_map(depth, mapper_gt_pose)`，是观测地图，不是房间真值。
  使用仿真位姿这一条件须对两者统一记录。
  `requirements.txt` 对 numpy、Pillow、PyYAML 存在多个冲突版本，不能直接安装到既有 vlfm 环境。
  需要检查现有依赖和门检测权重；禁止使用随机权重生成正式结果。

## 抽样与数据身份

首批默认使用 HM3D v1 **训练集的 calibration 场景分区**，保持与现有 `scripts/make_split.py`
相同的场景分区规则。开发分区另行抽样，不用最终验证集调参。
先均匀随机选择 5 个不同 scene_id，再分别均匀随机选择 1 个 episode；种子 `20261004`。
保持 episode_id 的原始类型、前导零、起点、旋转和目标元数据。
不能复用现有按类别分层抽样脚本，因为它不保证每个 episode 属于不同场景。

服务器恢复后，在固定提交的专用 worktree 中执行：

```bash
/home/zyq/miniconda3/envs/vlfm/bin/python scripts/make_room_split.py \
  --source /home/zyq/vlfm/data/datasets/objectnav/hm3d/v1/train \
  --output-dir runs/room_compare_20261004_sample5 \
  --count 5 --seed 20261004 --partition calibration
```

已有输出目录会被拒绝，重试必须另取 run-id。`selection.json` 记录原始数据文件和 SHA256，
`episodes.json.gz` 禁用 Habitat 自动加载其他场景 content，避免意外扩大 episode 集合。
真实选择后另外保存场景资源路径、场景配置与文件 SHA256。episode 数据 v1 和场景资源版本分别核验。

## 同观测实验

1. 连接恢复后先只读检查运行进程、GPU、数据、权重、ROS/C++ 库；不更新正在运行的 checkout。
   代码和小摘要通过 origin 正常 push / fast-forward 同步；长任务使用固定提交的独立 worktree。
2. 为选中的 5 个 episode 收集固定官方 VLFM baseline 的动作轨迹，保持现有主干。
   不从结果中挑选成功回合，不用最短路径或完整地图指导分割。
   同一 episode 的 RGB、depth、相机内参、位姿和时间戳只录制一次，给两者相同的帧序列。
   500 动作上限或原始 STOP 时终止，不填充静止帧；若提前终止，后续检查点标为未到达。
   这评估的是 ObjectNav 轨迹上的房间分割，不保证探索完整房屋，不代表主动探索性能。
3. Habitat 和 GPU 推理使用 `/home/zyq/miniconda3/envs/vlfm/bin/python`。
   ROS2 进程使用与 ROS 兼容的运行环境；与 Habitat 录制阶段隔离，不改动既有 conda 环境。
   传感器和动作沿用 baseline 的解析后配置，保存配置本身，不能仅依据历史脚本推断。
4. OccuSG 回放 bridge 将 Habitat 坐标显式变换为 ROS 坐标，发布 depth、camera_info、TF/里程计，
   保留官方占据建图与 DuDe 更新。只评估 room segmentation 时可关闭无关的物体图节点，须记录配置。
5. Active 回放 adapter 调用官方门检测、深度建图、门过滤和拓扑更新，导出各顶点的 `room_exp`。
   不调用它自己的动作选择；这是同轨迹适配实验，不能称为完整自主探索复现。
   记录脱离自主动作后可能影响门扫描与区域更新的限制，先做单例 smoke 核验。
6. 共同记录 100/250/500 动作检查点及实际终止时刻；每个 episode 都列出成功、失败、门检测零触发和覆盖情况。

两者共享观测但保留不同建图流程，因此结果比较包含建图差异。
若以后额外共享同一张 2D occupancy grid，须作为单独的“区域分割模块”消融，不能替代完整 OccuSG 流程。

## 输出与评分

每个 episode 保存 RGB 示例、轨迹与已观测区域、两者 occupancy map 和分割区域的并排俯视图，
叠加房间边界、门和起点。标签颜色仅供显示，不以颜色相等判断同一房间。
统一世界坐标、栅格原点、分辨率、楼层和有效 mask；楼层不同的区域不能投影混合。
原始证据、数组、权重、视频留在 `runs/`，只提交小型 manifest、CSV 和报告。

先核验 `sim.semantic_scene.regions` 或其他可追溯房间标注是否真实提供 room instance 分区。
物体类别、navmesh 连通分量和 region AABB 都不能直接当作精确房间边界真值。
没有可靠真值时，提交定性并排图、区域数、门数、已观测覆盖及运行耗时，准确率项记为不可用。
如果人工标注，说明标注范围和流程，并保留审阅记录；不能用某个方法的输出充当 GT。

可靠 GT 存在时，两者必须用相同的 GT 已观测有效区域评分，不能各用自身成功分割的像素作分母。
预测未分配的 GT 有效像素保留为错误，禁止掩蔽掉；报告 GT mask 覆盖率。
报告一对一 Hungarian IoU 匹配下的房间 IoU（未匹配 GT 房间计 0）、ARI、过分割/欠分割、
预测与 GT 房间数、覆盖率及逐阶段运行时间。忽略无可靠标签的未知空间。
只比较两者共同完成的精确 scene/episode 对，失败回合仍在总表中单列。
5 场景用于小样本观察，不据此宣称总体显著优劣。

保存 AV-Nav/upstream/submodule 提交、代码 dirty 状态、数据/场景/权重哈希、完整启动命令、
传感器配置、参数、pip freeze、ROS/编译版本、每 episode 指标、检查点、退出码。
在适配器单例 smoke 成功前不启动批量比较；缩短 smoke 不能作为最终分割结果。

## 下一步所需条件

服务器 SSH 恢复后才能实际抽样、检查 room GT / ROS / 门权重，完成两套适配器和服务器 smoke。
这些工作目前未完成，暂无任何真实分割结果或比较指标。
