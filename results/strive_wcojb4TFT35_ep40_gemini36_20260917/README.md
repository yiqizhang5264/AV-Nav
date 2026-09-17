# STRIVE + Gemini 3.6 Flash：wcojb4TFT35 / episode 40

## 结论

Gemini 3.6 Flash 完整运行成功，并挽救了这个 VLFM 镜子误检案例。GroundingDINO 在第 13、37、66 步把镜子区域作为 `tv_monitor` 候选送入复核；Gemini 三次均返回 `mirror`，机器人没有在镜子前停止。系统在第 214 步找到真实电视，并在第 232 步主动复核为 `tv_monitor`，最终成功停止。

## 最终指标

| 指标 | Gemini 3.6 Flash | Qwen3.5-9B |
|---|---:|---:|
| success | 1.0 | 1.0 |
| SPL | 0.4013376758 | 0.4013376758 |
| distance_to_goal | 0.0205530878 m | 0.0205530878 m |
| episode steps | 249 | 249 |
| travel distance | 24.9999956787 m | 24.9999956787 m |
| Found Goal | True | True |
| wall-clock time | 1282.18 s | 627.58 s |
| VLM HTTP attempts | 14 | 16（含 2 次失败后回退） |
| VLM latency sum | 756.682 s | 82.277 s |

两个模型得到完全相同的导航轨迹指标。Gemini 本次总耗时约为 Qwen 的 2.04 倍，VLM 请求累计时延约为 Qwen 的 9.20 倍。Gemini 的 14 次调用全部成功，但有三次明显的长尾请求，分别耗时 346.266、154.742 和 152.999 秒；这三次合计 654.007 秒，几乎解释了两次 episode 的总时长差。

这只是同一诊断 episode 的单次对照，不能据此推断两个模型在完整验证集上的总体成功率或平均速度。

## 关键视觉判定

| 步数 | 候选 | Gemini 判定 | Qwen 判定 | 作用 |
|---:|---:|---|---|---|
| 13 | 0 | `mirror` | `mirror` | 拦截镜子误检 |
| 37 | 0 | `mirror` | `mirror` | 再次拦截镜子误检 |
| 66 | 0 | `wall` | `wall` | 排除非目标表面 |
| 66 | 1 | `mirror` | `mirror` | 第三次拦截镜子误检 |
| 103 | 0 | `chair` | `chair` | 排除非目标候选 |
| 123 | 0 | `oven` | `stove` | 均排除为非目标 |
| 156 | 0 | `microwave` | `microwave` | 排除非目标候选 |
| 214 | 0 | `tv_monitor` | `tv_monitor` | 发现真实电视 |
| 214 | 1 | `refridgerator` | `wine` | 均排除为非目标 |
| 214 | 2 | `tv_monitor` | `tv_monitor` | 第二个真实电视候选 |
| 232 | 主动复核 | `tv_monitor`, `flag=True` | `tv_monitor`, `flag=True` | 确认目标并允许停止 |

Gemini 在两个非关键候选上的具体标签与 Qwen 不同，但二者的目标/非目标判断一致，因此没有改变规划轨迹或指标。原始判定文本保存在 `decisions/`。

## Gemini 调用统计

- 模型：`gemini-3.6-flash`，Google OpenAI-compatible endpoint。
- 共 14 次调用，全部成功。
- 累计 prompt tokens：44,027。
- 累计可见 completion tokens：1,201。
- API 报告的累计 total tokens：50,506；它大于 prompt 与可见 completion 之和，报告中不将差额自行解释为某一种 token 类型。
- 累计调用时延：756.6815 秒。
- 最大单次时延：346.2658 秒。
- 最大可见 completion：305 tokens。

逐调用遥测见 `vlm_calls.jsonl`。其中只记录请求哈希、字符数、时延和 token 计数，不包含提示词、图片或 API 密钥。

## 可复现信息

- AV-Nav 算法 commit：`4e74b53d55939f1bf457a98e59b24132673f5864`
- STRIVE commit：`1872d73b7db297705d251df73bf5f08ffed0d749`
- Habitat-Lab commit：`cb02f030655f9a475b379ec8d269979d9e17688d`
- GroundingDINO commit：`cfd5d3a985b0249de009b67d04f37263e11cdf3d`
- 运行 split SHA256：`47cb619f25874ce16d524d7dd1c1ddffb006054e85aa5ca86e6f380675bb5bd5`
- 服务器固定 worktree：`/home/zyq/AV-Nav-worktrees/strive-vlm-4e74b53`
- 成功运行目录：`/home/zyq/AV-Nav-worktrees/strive-vlm-4e74b53/runs/strive_gemini36_wcojb4TFT35_ep40_proxy_20260917T034000Z`

该 worktree 与 Qwen 最终实验使用同一算法提交。输出中的 `episode-0` 是过滤后单条数据集的内部索引，对应诊断清单中的 `wcojb4TFT35 / episode_id=40`。

## 启动故障记录

1. 第一次启动没有设置 Hugging Face 离线模式，初始化阶段尝试访问不可达的 `huggingface.co`，尚未调用 Gemini 即被终止。
2. 第二次使用本地缓存，但服务器无法直连 Google API，第一条 Gemini 请求无响应后被终止。
3. 第三次先执行 `proxy_on`，并让实验进程继承代理环境。预检通过，14 次 Gemini 调用全部成功，episode 退出码为 0。

失败运行目录均保留在服务器，未覆盖成功结果。仓库中不保存密钥、图像、视频、数据集或权重。
