# STRIVE 中 Gemini 与 Qwen3.5-9B 的配对比较

当前阶段只比较 STRIVE 固定框架中的 VLM，不接入 VLFM。STRIVE 上游提交、Habitat 数据、
检测与分割模型、episode 顺序、提示词、结构化输出 schema 和导航参数保持一致。

## 对照组

| 组别 | 后端 | 模型 |
|---|---|---|
| Gemini | Google OpenAI-compatible API | `gemini-3.6-flash` |
| Qwen | 本地 vLLM OpenAI-compatible API | `Qwen/Qwen3.5-9B` |

两组均通过 `scripts/run_strive_upstream.py` 进入固定的 STRIVE 子模块。适配器只替换 API
地址和模型名，不更改 STRIVE 提示词或导航逻辑。

Qwen 服务使用独立环境 `/home/zyq/miniconda3/envs/strive-qwen`。服务器的 NVIDIA 570
驱动对应 CUDA 12.8，因此固定使用 vLLM 0.18.1；启动脚本还会优先加载该环境中的
`libstdc++`，并默认关闭 Hugging Face Xet 下载。模型缓存可通过 `HF_HOME` 指向仓库外的
持久目录，实验目录只保存日志、PID 和公开配置。

Qwen3.5 默认使用 thinking 模式，而 STRIVE 的调用要求短结构化结果。Qwen 组设置
`STRIVE_VLM_DISABLE_THINKING=1`，适配器按官方接口传递
`chat_template_kwargs.enable_thinking=false`。该设置会写入公开运行配置；Gemini 组不使用此参数。

## 分阶段验证

1. 接口 smoke：文本房间选择与图像目标核验均须通过结构化输出解析。
2. Episode 0 配对 smoke：确认两组在同一回合都能完整结束并生成指标。
3. 固定小规模配对集：按六个 HM3D ObjectNav 类别均衡抽取，先运行每类 2 个、共 12 个
   episode。该阶段用于发现模型兼容性和明显退化，不作为最终性能结论。
4. 若 12 回合没有系统性失败，再冻结更大的配对集。两组必须使用完全相同的 episode ID；
   不根据其中一组结果更换样本。

## 记录指标

- 导航：SR、SPL、DTG、步数和行走距离。
- VLM：调用次数、文本/图像调用、成功率、结构化输出失败、每次延迟和 token 用量。
- 核验：假阳性拒绝、真目标保留、重新观察后的决策变化。
- 资源：峰值显存、episode 总时间，以及 Gemini API 用量。

`vlm_calls.jsonl` 仅记录请求哈希、是否含图像、延迟、token 和错误类型，不保存提示词、图像
或密钥。请求哈希用于核对两组是否收到相同输入；由于导航轨迹可能分叉，不能假定后续请求
天然配对。

## 结果表述

论文原设置是 Gemini 2.0 Flash。当前 Gemini 3.6 Flash 与 Qwen3.5-9B 都属于模型替换实验，
不能称为对论文数值的严格复现。单 episode 和 12 episode 阶段只报告工程可用性与初步趋势。
