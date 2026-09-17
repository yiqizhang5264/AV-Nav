# STRIVE + Qwen3.5-9B：wcojb4TFT35 / episode 40

## 结论

该案例在 STRIVE + Qwen3.5-9B 下被成功挽救。GroundingDINO 多次把墙上镜子作为 `tv_monitor` 候选送入复核，但 Qwen 在第 13、37、66 步均将其判为 `mirror`，机器人没有在镜子前停止。随后系统在第 214 步发现真实电视候选，并在第 232 步主动复核为 `tv_monitor`，最终成功停止。

这与同一 HM3Dv1 诊断案例的 VLFM 结果形成直接对照：VLFM 在第 100 步把语义实例 434（标注为 `mirror`）当作电视并失败，终点距离为 11.00635 m；本次 STRIVE 运行成功率为 1，终点距离为 0.02055 m。

## 最终指标

| 指标 | 结果 |
|---|---:|
| success | 1.0 |
| SPL | 0.4013376758 |
| distance_to_goal | 0.0205530878 m |
| episode steps | 249 |
| start-to-goal geodesic distance | 10.0686473846 m |
| travel distance | 24.9999956787 m |
| Found Goal | True |
| wall-clock time | 627.58 s |

STRIVE 的单 episode 数据集在过滤后只有一条记录，因此输出目录和 `metrics.csv` 中写作 `episode-0`；它对应诊断清单中的 `wcojb4TFT35 / episode_id=40`（原始场景文件数组索引 40）。

## 关键视觉判定

| 步数 | 候选 | Qwen 判定 | 作用 |
|---:|---:|---|---|
| 13 | 0 | `mirror` | 拦截镜子误检 |
| 37 | 0 | `mirror` | 再次拦截同类镜子误检 |
| 66 | 0 | `wall` | 排除非目标表面 |
| 66 | 1 | `mirror` | 第三次拦截镜子误检 |
| 103 | 0 | `chair` | 排除非目标候选 |
| 123 | 0 | `stove` | 排除非目标候选 |
| 156 | 0 | `microwave` | 排除非目标候选 |
| 214 | 0 | `tv_monitor` | 发现真实电视 |
| 214 | 1 | `wine` | 排除非目标候选 |
| 214 | 2 | `tv_monitor` | 第二个真实电视候选 |
| 232 | 主动复核 | `tv_monitor`, `flag=True` | 确认真实电视并允许停止 |

原始文本判定保存在 `decisions/`。这里不提交图像、视频、数据集或模型权重。

## VLM 运行统计

- 模型：`Qwen/Qwen3.5-9B`，本地 vLLM OpenAI-compatible endpoint。
- thinking：关闭。
- 首次结构化输出上限：1024 tokens。
- 共 16 次 HTTP 尝试，对应 14 个成功的逻辑调用；其中 2 次规划调用触发长度上限，确定性回退均成功。
- 成功响应累计：26,660 prompt tokens、1,932 completion tokens。
- 所有 HTTP 尝试累计 VLM 等待时间：82.277 s。
- 最长成功 completion：449 tokens。

失败调用没有可用 usage 字段，因此 token 合计只统计成功响应。详细逐调用遥测见 `vlm_calls.jsonl`，只包含请求哈希、字符数、时延和 token 计数，不包含提示词或密钥。

## 可复现信息

- AV-Nav commit：`4e74b53d55939f1bf457a98e59b24132673f5864`
- STRIVE commit：`1872d73b7db297705d251df73bf5f08ffed0d749`
- Habitat-Lab commit：`cb02f030655f9a475b379ec8d269979d9e17688d`
- GroundingDINO commit：`cfd5d3a985b0249de009b67d04f37263e11cdf3d`
- 诊断单条 split SHA256：`54df82dfb6ca71ae195234dae1c8e89c6e2d6d31c37135980de9365d764b5fb3`
- STRIVE 路径适配后的运行 split SHA256：`47cb619f25874ce16d524d7dd1c1ddffb006054e85aa5ca86e6f380675bb5bd5`
- 场景 basis 文件 SHA256：`075e243b372458df4055a55e27c93885ba44ae8c36b4bc4446e0393c60c14540`
- 服务器固定 worktree：`/home/zyq/AV-Nav-worktrees/strive-vlm-4e74b53`
- 服务器运行目录：`/home/zyq/AV-Nav-worktrees/strive-vlm-4e74b53/runs/strive_qwen_wcojb4TFT35_ep40_deterministic_retry_20260917T025700Z`

路径适配只把场景 ID 前缀从 `hm3d/val/` 改为 STRIVE/Habitat v0.2 注册表所需的 `hm3d_v0.2/val/`；对应 `.basis.glb` 文件哈希相同。

## 运行故障记录

1. 初次运行因 split 使用 `hm3d/val/` 而 STRIVE 场景注册表要求 `hm3d_v0.2/val/`，在算法启动前失败。
2. 路径修正后的无长度上限运行在约第 103 步出现 14,774-token 长输出，触及 16,384-token 上下文上限并失败。这次运行已经在第 13、37、66 步正确拒绝镜子。
3. 首版 1024-token 限制能快速截断长输出，但回退新增了第二条 system 消息，Qwen 返回 HTTP 400。随后改为合并现有 system 消息。
4. 合并 system 消息后的运行仍有一次图像请求在首请求和回退中都达到 1024 tokens。最终版仅在异常回退中使用 `temperature=0`，并允许最多 2048 tokens。
5. 最终版完整运行退出码为 0；两次长度异常都在回退中成功恢复。

这些失败目录均未覆盖，保留在对应服务器固定 worktree 的 `runs/` 中。
