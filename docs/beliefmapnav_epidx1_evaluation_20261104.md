# BeliefMapNav episode 全量评估：全局索引 1（2026-11-04）

## 评估身份

- 全局 episode 索引：1，切片 `[1:2]`
- 场景：`TEEsavR23oF`
- content 文件内索引：1
- 数据中的 `episode_id`：`"0"`（该字段在同一文件中重复，不能单独作为身份）
- 数据集目标：`sofa`
- 策略内部目标名：`couch`
- 起点：`[1.34813, 0.166338, -6.12816]`
- 起始旋转：`[0, 0.16316, 0, -0.9866]`
- 最大步数：500
- Habitat-Sim / Habitat-Lab：0.2.5 / 0.2.5
- BeliefMapNav 官方提交：`c9800d09c2e0f0203a0efbee976d421afade7456`
- VLM：本地 `Qwen/Qwen3.5-9B`，snapshot `c202236235762e1c871ad0ccb60c8ee5ba337b9a`
- Qwen 设置：`enable_thinking=false`

该运行是 Qwen 替代模型诊断，不是论文 GPT-4o 基线，也不用于调参。

## 最终结果

| 指标 | 值 |
|---|---:|
| 退出码 | 0 |
| 用时 | 1545.91 秒（25 分 45 秒） |
| 最后一步 | 229 |
| Success | 0 |
| SPL | 0 |
| SoftSPL | 0.299239 |
| Reward | 8.220278 |
| Habitat `distance_to_goal` | 1.574307 m |
| `target_detected` | 1 |
| `stop_called` | 1 |
| 失败分类 | `false_positive` |

策略没有耗尽 500 步，而是在第 229 步主动 STOP。

动作统计：STOP 1、前进 119、左转 69、右转 41。模式统计：initialize 13、explore 186、navigate 31。官方动作枚举中动作 `2` 是左转、动作 `3` 是右转；此前的方向标签反置，现已更正，动作总数和所有评估指标不变。

## 事件时间线

1. 步骤 0–12：初始化环视。
2. 步骤 13–198：持续 explore，前 199 步没有一次 Qwen 请求。
3. 步骤 198 后：检测器首次产生 couch 候选，Qwen 对红框多座 sofa 返回 `Answer: yes`，策略切换为 navigate。
4. 步骤 213：画面出现红、绿两个候选。Qwen 选择红框，并将其判断为符合 couch。
5. 步骤 216：视角变化后，Qwen 认为红框是较小双座 recliner、旁边存在更大的 sofa，返回 `Answer: no`。
6. 步骤 217：再次比较两个候选，Qwen仍选择红框，并重新返回 `Answer: yes`。
7. 步骤 229：内部对象点距离降至 0.410478 m，低于代码硬编码的 0.6 m STOP 阈值，策略停止；Habitat 到合法目标视点的距离仍为 1.574307 m，因此失败。

## Qwen 调用

- OpenAI-compatible Chat Completions 请求：6
- 带 `Answer: yes`：3
- 带 `Answer: no`：1
- 颜色候选选择：2，均选择 `red`

视觉证据中，红框对象是明显的多座 sofa，绿色框是单座 recliner。因此这不是“把电视或椅子明显误认成沙发”的简单语义错误。但 Qwen 对红框在相邻视角下出现 `yes → no → yes` 波动，说明当前自然语言判据（尤其“若画面里有更大的 couch 则回答 no”）具有明显的视角敏感性。

## 主要问题判断

### 1. 检测门控使 VLM 很晚才参与

前 199 步没有任何 Qwen 请求。VLM 再强，也无法修复检测器没有召回候选的阶段。该 episode 中 86% 以上的动作发生在首次 VLM 调用之前，主要成本来自探索、3D 地图和检测链路。

### 2. 内部停止距离与 Habitat 成功距离不一致

代码根据对象点云中的目标点计算欧氏距离，并在 `cur_dis_to_goal <= 0.6` 时 STOP。本次内部距离为 0.410478 m，但 Habitat 指标仍为 1.574307 m。说明检测框深度投影、对象点云代表点、可通行落点和 Habitat 合法目标视点之间存在系统性偏差。

这是本次 `false_positive` 的直接原因。单纯替换更强 VLM 不能解决该停止逻辑。

### 3. 多候选提示本身不稳定

官方提示要求：如果画面里存在更大的 couch，则当前框回答 no。这会使同一个真实目标随着裁剪、遮挡和相机位置变化，在 yes/no 之间切换。更合理的做法应是保持候选身份并累计多视角证据，而不是让单帧相对大小决定是否删除检测。

### 4. VLM 证据被覆盖

虽然实际产生 6 次请求，运行目录最终只保留一个 `VLM/annotated_rgb_1.jpg`。文件在后续请求中被覆盖，无法逐请求复核输入图像与回答。这不满足可审计复现要求。

### 5. 对象体积计算存在非致命异常

日志中出现 20 次 `Error in compute_volume_and_density: No points given`。运行没有崩溃，但空点云仍进入体积/密度流程，说明对象候选或 voxel 聚合缺少严格的输入验证。

## 证据

服务器运行目录：

`/home/zyq/BeliefMapNav-repro/c9800d0/outputs/qwen35_h025_epidx1_full500_20261104`

目录包含 `stdout.log`、`stderr.log`、`exit_code.txt`、`final_result.txt`、230 张逐步合成图和 `provenance.txt`。关键哈希：

- 数据分片：`73c74356c2133973adbc0eea10b7064ae70f2032015884f08f773c02582589a6`
- 最终保留的 VLM 标注图：`0372720adb37a5adf6f7130b94a08c2a4a01470c7c115a4c55e8fbb6b2832f9c`
- 步骤 198 合成图：`8b2499443c427f719f184add87c882077605061092ad0660d6f8be937465d77e`
- 步骤 213 合成图：`1bce2a1d10e936aed93a24279228de3e02302eb014673cde831261385b29c9d3`
- 步骤 229 合成图：`ae063415dc992efd3f85dbafbb70edbf6c853f9b4062eeabfdb7b21f2ba85f75`

本地原始证据图只保存在被 Git 忽略的 `runs/` 下，没有提交到仓库。测试结束后专用 Qwen 服务已关闭，GPU 1 和 GPU 2 已释放。
