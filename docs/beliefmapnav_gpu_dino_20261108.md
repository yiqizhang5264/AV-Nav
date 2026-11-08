# BeliefMapNav GPU GroundingDINO 切换记录（2026-11-08）

## 环境

- GPU：NVIDIA RTX 6000 Ada（SM89）
- 服务端口：`14181`
- 启动脚本：`scripts/run_beliefmapnav_gpu_dino.sh`
- Python：`/home/zyq/venvs/beliefmapnav-c9800d0/bin/python`
- PyTorch：`2.6.0+cu124`
- GroundingDINO：`/home/zyq/BeliefMapNav-repro/GroundingDINO-torch26`
- 配置：`GroundingDINO_SwinT_OGC.py`
- 权重：`groundingdino_swint_ogc.pth`

原 CPU 服务使用 PyTorch 1.12.1+cu113。该 CUDA 版本无法在 RTX 6000 Ada 上执行 GroundingDINO，真实请求报错 `nvrtc: invalid value for --gpu-architecture (-arch)`。GPU 服务因此使用已应用 `adapters/beliefmapnav/groundingdino_torch26.patch` 并重新编译扩展的 PyTorch 2.6 环境。

## 切换前验证

临时 GPU 服务先在 GPU 2、端口 `15181` 启动，正式 CPU 服务 `14181` 保持在线。四服务真实请求检查全部通过。

固定黑图、`chair .` 查询：

| 服务 | 延迟 | 输出 |
|---|---:|---|
| CPU / PyTorch 1.12 | 57.957 s | 空检测 |
| GPU / PyTorch 2.6 | 0.626 s | 空检测 |

两者响应 SHA256 相同；该首次 CPU 请求包含明显的运行时编译开销。

episode 证据图、`couch .` 查询：

| 服务 | 延迟 | 检测结果 |
|---|---:|---|
| CPU / PyTorch 1.12 | 2.220 s | 2 个 couch 框，置信度 0.820741 / 0.505864 |
| GPU / PyTorch 2.6 | 0.131 s | 1 个 couch 框，置信度 0.820311 |

主框的位置和置信度基本一致，但第二个低置信度框没有在 GPU 环境中保留。因此该切换显著提升速度，但不是逐位等价变更；正式结果必须记录 DINO 环境，不得与旧 CPU 诊断直接混合汇总。
