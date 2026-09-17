# STRIVE + Qwen3.5-9B：HM3Dv2 val 全量评估

## 当前状态

- 状态：运行中。
- 正式启动时间：2026-09-17 22:38:33（Asia/Shanghai）。
- 启动验收：2026-09-17 22:47:06，两个 worker 均已完成并落盘首个正式 episode，随后继续运行。
- 评估范围：HM3Dv2 ObjectNav `val` 的全部 1000 个 episode。
- 运行方式：100 个分片，每片 10 个 episode，同时运行 2 个 worker；完成的分片可以原地续跑，不会覆盖已有 attempt。
- 本记录只说明任务已经启动以及运行环境通过检查。在 1000 个 episode 全部完成并汇总前，不报告 SR、SPL 等最终结论。

## 固定版本

| 项目 | 值 |
| --- | --- |
| AV-Nav commit | `bd665eb7b370db9fbf5650720027b7a6915acc5c` |
| STRIVE commit | `1872d73b7db297705d251df73bf5f08ffed0d749` |
| Habitat-Lab commit | `cb02f030655f9a475b379ec8d269979d9e17688d` |
| GroundingDINO commit | `cfd5d3a985b0249de009b67d04f37263e11cdf3d` |
| VLM | `Qwen/Qwen3.5-9B`，本地 OpenAI-compatible vLLM 服务 |
| 推理设置 | `disable_thinking=true`，正常请求最多 1024 tokens；结构化输出失败时使用只含最终字段的 512-token 重试 |
| 固定服务器工作树 | `/home/zyq/AV-Nav-worktrees/strive-full-bd665eb` |
| 正式 suite | `strive_qwen_hm3dv2_val1000_bd665eb_20260917` |

完整公开配置见 [manifest.json](manifest.json)，依赖和数据路径检查见 [preflight.json](preflight.json)。这些记录不含密钥。

## 数据集清单

- 36 个场景、1000 个 episode。
- 类别数：bed 165、chair 195、plant 152、sofa 187、toilet 166、tv_monitor 135。
- 36 个 content 文件的组合清单 SHA-256：`0c2e16c6aad0ec3e75c2c609dae8187cd51ec6f6dfc31a7ade2767c955bf8d3c`。
- 顶层 `val.json.gz` SHA-256：`e0f25e8224b7b3931bd1b505482e35d343ebad8a52988c72ae94c3bd6a6b5169`。
- HM3D 场景配置 SHA-256：`c16fa23b09e36d528c3f12c9799eaa9cf0d733dcd22c9f329c0508d551b640a4`。

逐文件哈希与 episode 数见 [dataset_manifest.json](dataset_manifest.json)。

## 启动前验证和故障记录

最初的正式尝试暴露了 Qwen 在 STRIVE 长结构化推理字段上超过 1024/2048 token 的问题，因此该尝试没有纳入正式结果。随后依次限制推理步骤、字符串和列表长度，并将失败重试改为 STRIVE 实际使用的最终字段。固定提交上的 episode 1 验证成功：`success=1.0`、`SPL=0.2106939120948465`、`distance_to_goal=0.15037854`、203 steps。该单 episode 只用于运行稳定性检查，不作为全量效果结论，也不会混入正式 suite。

正式 suite 的预检结果为 `ready=true`，并核对了 AV-Nav、STRIVE、Habitat-Lab、GroundingDINO、数据文件和场景配置的版本或哈希。Qwen 服务运行在 GPU0，两个 STRIVE worker 运行在 GPU1。启动时 GPU 型号记录见 [gpus_at_launch.txt](gpus_at_launch.txt)。

两个 worker 的首条正式记录分别为 episode 0（toilet，success 1.0，SPL 0.470888，128 steps）和 episode 10（sofa，success 1.0，SPL 0.359023，187 steps）。到启动验收时，每个 worker 都出现过 2 次长度超限，4 次失败请求均由最终字段重试成功恢复，没有导致 episode 或进程退出。这里的两条结果只证明正式评估链路能够持续运行，不作为模型效果估计。结构化状态见 [startup_status.json](startup_status.json)，原始小型指标快照和不含提示词、图像的 VLM 调用元数据保存在同目录。

## 服务器路径与监控

正式结果目录：

```text
/home/zyq/AV-Nav-worktrees/strive-full-bd665eb/runs/strive_qwen_hm3dv2_val1000_bd665eb_20260917
```

只读查看进程和阶段性汇总：

```bash
ps -p 455179 -o pid,ppid,stat,etime,cmd
cd /home/zyq/AV-Nav-worktrees/strive-full-bd665eb
/home/zyq/miniconda3/envs/strive/bin/python scripts/summarize_strive_full_eval.py \
  --suite-id strive_qwen_hm3dv2_val1000_bd665eb_20260917
```

不要在任务运行期间更新这个工作树。任务异常退出后，使用完全相同的 suite 参数重新运行，程序会跳过已经完整成功的分片并为未完成分片创建新的 attempt。

## 完成标准

只有满足以下条件才把全量评估标记为完成：100 个分片均以退出码 0 结束；`metrics.csv` 精确覆盖 episode 0–999 且没有重复或缺失；没有失败分片；保存总 SR、SPL、distance-to-goal 和分目标类别指标；保留失败 episode 与 VLM 调用错误计数。
