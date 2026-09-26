# STRIVE + Qwen3.5-9B：HM3Dv2 增量续跑

## 状态

2026-09-26 已修复并启动增量续跑。已有完整episode不会重新评估。

- 固定AV-Nav提交：`4549f75f0574c1d453e9b7b9ce29f7e02fe743ac`
- 固定STRIVE提交：`1872d73b7db297705d251df73bf5f08ffed0d749`
- 当前任务：episode 46–49，1个worker。
- 后续任务：episode 62–999，2个worker；前一任务完成后自动启动。
- Qwen3.5-9B继续使用GPU0；STRIVE使用GPU1。

## 已复用episode

最终汇总将复用以下已经完整写出指标的episode：

| Episode | 来源 |
| --- | --- |
| 0–39 | `strive_qwen_hm3dv2_val1000_fe62c0a_20260924`的4个完整分片 |
| 40–44 | 旧suite的`shard_0040_0049_attempt_001`，取最早attempt，避免重复计数 |
| 45 | 修复500步边界后的定向验证，提交`a6ec8f8` |
| 50–59 | 旧suite的完整分片`shard_0050_0059_attempt_001` |
| 60 | 旧suite的`shard_0060_0069_attempt_001` |
| 61 | CPU预体素化修复后的定向验证，提交`ce758df` |

episode 46–49和62–999此前没有完整可用指标，因此由新任务补齐。最终合并时必须精确覆盖0–999，拒绝缺失和重复编号，并在逐episode表中保存来源suite、attempt和提交。

## 修复内容

1. Habitat达到500步后，STRIVE再次调用`env.step()`时正常结束episode，不再抛出断言。
2. 点云合并前在CPU按原0.05m体素分辨率去除非有限点并预下采样，避免Open3D CUDA体素哈希表在episode 61上OOM；固定STRIVE子模块保持不变。
3. episode 61定向验证成功：Success 1.0、SPL 0.737659、60步。
4. 只有分片全部指标已经写出且日志精确匹配Open3D CUDA缓存析构错误时，才接受清理阶段的`-6`退出。运行过程中的OOM仍视为失败。
5. 36项本地单元测试通过。

## 服务器路径

固定工作树：

```text
/home/zyq/AV-Nav-worktrees/strive-full-4549f75
```

补跑suite：

```text
runs/strive_qwen_hm3dv2_missing_0046_0049_4549f75_20260926
runs/strive_qwen_hm3dv2_missing_0062_0999_4549f75_20260926
```

启动日志：

```text
runs/strive_qwen_hm3dv2_resume_4549f75_20260926_launcher.log
```

当前配置和预检快照见[missing_0046_0049_manifest.json](missing_0046_0049_manifest.json)与[preflight.json](preflight.json)。

## 2026-09-26阶段性成功与失败

截至episode 47完成，共有60个唯一有效episode指标：43个成功、17个失败。该阶段成功率为71.67%，仅用于监控运行，不能作为HM3Dv2全量结果。

失败的操作性分类如下：

| 类型 | 数量 | Episode |
| --- | ---: | --- |
| 疑似假阳性或错误停止 | 8 | 9、23、28、30、36、39、40、54 |
| 未发现目标并探索超时 | 7 | 20、26、33、35、50、57、59 |
| 已发现候选但导航到500步仍未成功 | 1 | 45 |
| 未发现目标且提前结束 | 1 | 34 |

“疑似假阳性或错误停止”表示`Found Goal=True`、`success=0`、终点距离标注目标超过1m；仅凭指标不能继续区分检测器误检、VLM语义混淆、停在错误实例或数据集标注缺失，必须结合视频和候选证据复核。逐episode数据见[completed_episode_classification.csv](completed_episode_classification.csv)，结构化汇总见[failure_summary.json](failure_summary.json)。

## 视频保留

STRIVE为每个完成episode保存三段视频：`fps.mp4`为RGB第一视角轨迹，`depth.mp4`为深度轨迹，`metrics.mp4`为带Success、SPL、距离、步数和目标信息的俯视轨迹。后续未运行episode继续保存这三段视频。视频保留在服务器各suite的`shards/.../output/episode-<id>/`目录，不提交GitHub。

已有旧输出共发现68组episode视频（包含失败attempt产生的重复episode），204个MP4合计约484MB；单个episode通常约7MB。服务器当前剩余约27TB，保存剩余episode视频没有磁盘压力。最终汇总时将为0–999生成去重的视频索引，便于按失败类型逐个查看。
