# BeliefMapNav 3-episode 小样本评估（GPU DINO + Qwen3.5-9B）

## 结论

2026-11-08 在服务器完成了 3 个 HM3D `val` episode 的完整 500-step 上限评估。进程正常退出（exit code 0），总耗时 3472.41 秒（57 分 52 秒）。结果为 1/3 成功：SR 33.33%，平均 SPL 0.1118，平均 soft-SPL 0.2111。

这是链路和失败模式诊断，不是论文 efficacy 结果。样本只有 3 个，VLM 使用本地 Qwen3.5-9B 而非论文的 GPT-4o，GroundingDINO 也已从旧 CPU/PyTorch 1.12 环境切换到 GPU/PyTorch 2.6；该结果不得与论文数字直接比较，也不得用于在 `val` 上调参。

## 固定环境与输入

- BeliefMapNav 上游提交：`c9800d09c2e0f0203a0efbee976d421afade7456`
- AV-Nav 运行辅助提交：`22c66dd21a460122dd8cb65a9afa698e01f8ea88`
- Habitat-Sim / Habitat-Lab：0.2.5
- Habitat 与策略 Python：`/home/zyq/venvs/beliefmapnav-c9800d0/bin/python`
- GroundingDINO：GPU 0，PyTorch 2.6.0+cu124，服务端口 14181
- Qwen：GPU 1，`Qwen/Qwen3.5-9B`，snapshot `c202236235762e1c871ad0ccb60c8ee5ba337b9a`
- Qwen 服务：vLLM 0.18.1，OpenAI-compatible `http://127.0.0.1:16000/v1`，`enable_thinking=false`
- 其他服务：BLIP2-ITM 14182、MobileSAM 14183、YOLOv7 14184
- 数据集：HM3D ObjectNav v1 `val`，场景 `TEEsavR23oF`，顺序索引 0:3
- 场景内容文件 SHA256：`73c74356c2133973adbc0eea10b7064ae70f2032015884f08f773c02582589a6`
- `val.json.gz` SHA256：`96950bd7d4ef4c6dfc7e89842d9232559b447d8cfdbf10fd2a644bb19550f95c`
- 结果目录：`/home/zyq/BeliefMapNav-repro/c9800d0/outputs/gpu_dino_qwen35_h025_smoke3_attempt2_20261108`

关键覆盖参数为：

```text
start_split=0
end_split=3
habitat_baselines.test_episode_count=3
habitat.environment.max_episode_steps=500
habitat_baselines.eval.video_option=[]
habitat_baselines.torch_gpu_id=0
habitat_baselines.rl.policy.use_vqa=true
```

## 逐 episode 结果

| 索引 | 目标 | 步数 | 结果 | distance_to_goal | SPL | soft-SPL | target_detected / stop_called | Qwen 请求 | 失败类型 |
|---:|---|---:|---|---:|---:|---:|---|---:|---|
| 0 | toilet | 500 | 失败 | 19.1298 m | 0 | 0 | 0 / 0 | 1 | `never_saw_target_did_not_travel_stairs_likely_infeasible` |
| 1 | sofa | 226 | 失败 | 1.5743 m | 0 | 0.2992 | 1 / 1 | 6 | `false_positive` |
| 2 | sofa | 307 | 成功 | 0.03445 m | 0.3353 | 0.3342 | 1 / 1 | 2 | `did_not_fail` |

聚合指标：

| 指标 | 值 |
|---|---:|
| Success | 0.3333 |
| SPL | 0.1118 |
| soft-SPL | 0.2111 |
| distance_to_goal | 6.9128 m |
| reward | 5.3431 |
| target_detected | 0.6667 |
| stop_called | 0.6667 |
| traveled_stairs | 0.0000 |

按 tqdm 累计时间估算，三个 episode 分别约为 29 分 16 秒、11 分 49 秒和 16 分 16 秒。动作总数为 1033。

## 可复现的行为与问题

1. 索引 0 在关闭 VQA 的诊断中曾于第 141 步因 DINO 候选直接误停。完整 Qwen 运行中，Qwen 在第 352 步把红框物体判断为 pedestal sink 而不是 toilet，回答 `no`，避免了该误报；但此后仍未召回真实目标，最终跑满 500 步。Qwen 减少了误报，却没有解决检测召回和探索覆盖问题。
2. 索引 0 曾进入 `get_close_to_stair` / `climb_stair` 并在换层逻辑后重新执行 12 次初始化旋转，但最终 `traveled_stairs=0`。策略状态切换不等于 Habitat 指标认可的实际跨层移动。
3. 索引 1 在第 195 步接受 sofa 候选并进入 navigate。相邻视角中 Qwen 的判断出现 `yes -> no -> yes`，且两候选选择路径额外调用 Qwen。最后在距目标约 1.57 m 时停止，仍被 Habitat 判为 false positive。这与此前同索引单例诊断一致，说明视角敏感和停止鲁棒性问题可重复出现。
4. 索引 2 在第 285 步接受 sofa 候选，第 306 步成功停止。它证明 GPU DINO、Qwen、3D 地图、规划和 Habitat 成功判据可以端到端走通，但不能用单个成功样本推断总体性能。
5. 起始初始化在三个 episode 中均为步骤 0--11 的 12 次转向，步骤 12 进入 explore，验证了 12-turn 适配。索引 0 的楼梯状态重置后又执行一次 12-turn 初始化，这是换层后的重新建图行为，不是起始阶段 390 度旋转问题复发。

## 服务统计与稳定性

- Qwen 共收到 9 次 `/v1/chat/completions` 请求，全部 HTTP 200；按 episode 分配为 1、6、2 次。
- GPU GroundingDINO 在本次运行窗口内收到 2014 次 `/gdino` 请求，全部 HTTP 200。
- 运行日志未发现 Traceback、APIConnectionError、CUDA OOM、HTTP 500 或请求超时。
- GroundingDINO GPU 版与旧 CPU 版并非逐位等价；此前固定证据图对比中 GPU 版少保留了一个低置信度框。因此本结果只能与同一 GPU DINO 环境下的后续实验比较。
- 评估结束后专用 Qwen tmux 已关闭，GPU 1 从约 32.1 GiB 降至 18 MiB。GPU GroundingDINO 正式服务继续保留在 GPU 0 的 tmux `beliefmapnav_gdino_gpu_14181` 中。

## 运行前失败尝试（不计入指标）

所有失败目录均保留，未覆盖正式结果：

1. `gpu_dino_h025_smoke3_novqa_20261108`：Hydra 覆盖键写错，在 episode 开始前退出。
2. `gpu_dino_h025_smoke3_novqa_attempt2_20261108`：`use_vqa=false` 只关闭部分 VQA 路径；索引 0 完成后，索引 1 第 211 步仍从双检测比较路径调用 OpenAI 接口，因 Qwen 未启动而退出。这证明官方 `use_vqa` 开关不完整。
3. `gpu_dino_qwen35_h025_smoke3_20261108`：把 YOLO 端口环境变量误写成 `YOLO_PORT`，而源码读取 `YOLOV7_PORT`；运行在第 0 步前被主动中止。
4. `gpu_dino_qwen35_h025_smoke3_attempt2_20261108`：修正为 `YOLOV7_PORT=14184` 后完成，本文所有指标均来自此目录。

## 下一步建议

先在训练场景构造目标存在/不存在的配对证据集，分别测量 DINO 召回、Qwen 接受率、相邻视角一致性和停止距离分布。当前 3 个样本已经显示两个独立瓶颈：目标候选长期不触发，以及候选通过后仍可能错误停止。只有在训练场景上固定这些诊断后，才应扩大开发集评估；不要根据本次 `val` 结果调阈值。
