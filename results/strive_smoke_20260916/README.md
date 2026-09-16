# STRIVE Gemini 端到端 smoke（2026-09-16）

本记录只证明固定版本 STRIVE 在服务器上的完整工程链路可运行，不作为论文效果结果。

## 成功运行

- 服务器目录：`/home/zyq/AV-Nav-worktrees/strive-84da6f7/runs/strive_gemini36_e2e_20260916T122713Z`
- AV-Nav：`84da6f70107b87fa715fefa0f96d5009e9740433`
- STRIVE：`1872d73b7db297705d251df73bf5f08ffed0d749`
- 独立 worktree：`/home/zyq/AV-Nav-worktrees/strive-84da6f7`
- Python：`/home/zyq/miniconda3/envs/strive/bin/python`
- GPU：0
- 数据：HM3D v0.2，ObjectNav HM3D v2 validation，episode 0
- VLM：`gemini-3.6-flash`
- 进程退出码：0
- 运行时间：约 498.31 秒

`metrics.csv` 中的单回合结果：

| target | success | SPL | steps | start-goal distance | travelled distance | final distance |
|---|---:|---:|---:|---:|---:|---:|
| toilet | 1.0 | 0.470888 | 128 | 4.591166 m | 9.750004 m | 0.021273 m |

运行完成了 Habitat 初始化、Gemini 请求、GroundingDINO、SAM、拓扑节点和房间构建、规划、
导航停止及指标/视频/点云写出。原始证据保留在上述服务器运行目录，未提交到 Git。

## 运行前诊断

1. `/home/zyq/AV-Nav/runs/strive_gemini_e2e_20260916T122115Z`：未启用代理，
   Hugging Face tokenizer 请求出现 `Network is unreachable`，退出码 1。
2. `/home/zyq/AV-Nav/runs/strive_gemini_proxy_e2e_20260916T122259Z`：启用
   `proxy_on` 后依赖加载成功，但上游写死的 `gemini-2.5-flash` 对新用户返回 404，退出码 1。
3. AV-Nav 新增运行时入口，通过 `STRIVE_GEMINI_MODEL=gemini-3.6-flash` 覆盖模型名；
   STRIVE 子模块保持固定且无修改。模型名写入每个运行目录的 `gemini_model.txt`。

服务器原检出中仍有 VLFM 进程运行，因此没有对其执行 `git pull`。成功 smoke 使用固定提交的
独立 worktree，避免改变正在运行实验的代码。

## 解释限制

这是单回合工程 smoke。Success 和 SPL 不能代表 STRIVE 的复现精度，也不能与 VLFM 或
拟议主动视觉方法做效果比较。正式复现需要冻结评估集和模型版本，运行完整 episode 集，记录
API 失败/重试与费用，并汇总 SR、SPL 等指标。
