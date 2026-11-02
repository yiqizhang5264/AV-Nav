# OCS train-5：闭环启动记录

日期：2026-11-02。算法冻结提交 `5ac0795`，队列监督器提交 `d65847e`。不更新运行中的 checkout。

## 验收与资源

- 算法及数据管线服务器测试 174/174 通过；顺序监督器另 11/11 通过。
- 固定 12 条 CPU 回放通过，详情见 `OCS_CPU12_RESULTS_20261102.md`。
- 固定 train-5 已逐条核对原始任务 payload、起始位姿、目标集合、场景资产，且通过真实 Habitat dataset loader 检验。
- GPU 2 在启动前连续三次检测均为 0% 利用率、16 MiB、无计算进程；GPU 0/1/3 有其他任务，不使用。
- DINO 仍按既有兼容路径运行在 CPU；BLIP2ITM、MobileSAM、YOLOv7 与单环境仿真/PointNav 使用物理 GPU 2。四服务真实推理健康检查全部 `ok`。
- 专用端口：14181 / 14182 / 14183 / 14184；服务 PID 初始为 143946 / 143947 / 143948 / 143949。这些 PID 只适用于此次记录，不能当作未来的固定停止名单。

## 第一次尝试及修复

目录：`/home/zyq/vlfm_evidence_runs/ocs_train_smoke5_20261102`。

原版尚未执行动作即因顶视图 `_draw_goals_aabb` 实例索引越界退出，0/5 完成；监督器正确停止，未启动 anchor_only 与 OCS。

根因不是缺少语义网格或目标列表错误，而是新工作区没有 `external/vlfm/data/scene_datasets` 资源链接。虽然任务 GLB 使用绝对路径、能够被找到，上游相对路径的 `hm3d_annotated_basis.scene_dataset_config.json` 没有加载，进而找错了语义描述文件名。补上指向 `/home/zyq/vlfm/data/scene_datasets` 的链接后，日志确认 `5biL7VEkByM.semantic.txt` 与 `.semantic.glb` 成功加载。

没有关闭包围框显示、修改成功半径、目标 ID、任务、算法或步数。旧目录和错误日志全部保留。

## 当前重试

目录：`/home/zyq/vlfm_evidence_runs/ocs_train_smoke5_20261102_retry_sceneassets`。

监督器启动时间：2026-11-02 09:43:40（北京时间），初始 PID 150575。按 `original → anchor_only → ocs` 顺序运行，各组 5 条、相同数据 SHA、seed=17、预算 500 步。每组只有退出码为 0、5 条完整结果及任务身份完全匹配，才会进入下一组；失败立即停队列。

本文件记录启动，不将排队或进程存在称为评估完成。最终状态以 `queue/summary.json` 及各条件 `completion.json` 为准。

## 资产指纹

| 资产 | SHA256 |
|---|---|
| train-5 JSON gzip | `099070e6ef898d8222708b3f4a42b80702d3e433afdfc8460b7dd33e98c5227b` |
| HM3D annotated basis 场景配置 | `c16fa23b09e36d528c3f12c9799eaa9cf0d733dcd22c9f329c0508d551b640a4` |
| dummy policy | `6b58d91a6fd2a0372f8dce6df7da80048482429d93fc53f7f6f6618a76991990` |
| PointNav | `ecb6f217fad7abed04dea5db36f1a88cf1d49e58943be4d283ba3de64c2ac2c2` |
| MobileSAM | `6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f` |
| YOLOv7 | `b370120a414bf32b5d65fc808e5a32c8d9b3c63902d1bc41894fc9d86eccf9cb` |
| GroundingDINO | `3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799` |

运行目录内的 manifest 还冻结逐项方法配置、入口源码哈希、完整命令、环境变量与任务身份。BLIP2ITM 使用原环境已有的离线模型缓存，不下载或切换模型版本。
