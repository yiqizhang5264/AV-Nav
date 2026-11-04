# BeliefMapNav 本地 Qwen3.5-9B 适配记录（2026-11-04）

## 结论

服务器已有的 `Qwen/Qwen3.5-9B` 已通过 vLLM 0.18.1 以 OpenAI 兼容接口部署，并成功接入 BeliefMapNav 官方提交 `c9800d09c2e0f0203a0efbee976d421afade7456`。Habitat-Sim 和 Habitat-Lab 均为 0.2.5。

这只是国产模型替代诊断，不是论文 GPT-4o 基线。20 步导航 smoke 正常退出，但该 episode 的 `target_detected=0`，因此导航过程中 VQA 为零触发；不能用其 `SR=0` 或 `SPL=0` 评价 Qwen。

## 部署身份

- 模型：`Qwen/Qwen3.5-9B`
- Hugging Face snapshot：`c202236235762e1c871ad0ccb60c8ee5ba337b9a`
- 本地缓存：`/home/zyq/AV-Nav/runs/qwen_hf_cache/hub/models--Qwen--Qwen3.5-9B`
- 推理环境：`/home/zyq/miniconda3/envs/strive-qwen`
- PyTorch：2.10.0+cu128
- vLLM：0.18.1
- 服务地址：`http://127.0.0.1:16000/v1`
- 服务模型名：`Qwen/Qwen3.5-9B`
- 最大上下文：16384
- GPU memory utilization：0.65
- 模型 GPU：物理 GPU 1
- Habitat GPU：物理 GPU 2

模型权重加载耗时 247.80 秒，vLLM 报告权重占用 17.66 GiB；完成 KV cache 和 CUDA graph 初始化后，GPU 1 总占用约 32.3 GiB。测试结束后专用 Qwen tmux 已关闭，GPU 1 恢复到约 18 MiB。

## 显式适配器

AV-Nav 提交 `36285cf` 新增：

- `adapters/beliefmapnav/official_c9800d0_model_env.patch`
- `adapters/beliefmapnav/official_c9800d0_qwen35_no_thinking.patch`

新增环境变量：

- `BELIEFMAPNAV_LLM_MODEL`：默认仍为 `gpt-4o`，所以不改变官方基线默认行为。
- `BELIEFMAPNAV_DISABLE_THINKING`：仅显式启用时向兼容接口传递 `chat_template_kwargs.enable_thinking=false`。

对应 patch SHA256：

- model env：`f49c6d9dcce27a4cf90e7d2156bbdbd55fe1c92daa10e3b23b7b98f6663c1e63`
- no thinking：`9b188fc750fdc44a47640e517c55334333e6e54e55fe0d33f1ccaadc721df657`

补丁在官方源码上依次应用后通过 `py_compile`。本地完整单元测试为 75 passed、1 skipped。

## 发现的 Qwen 兼容性问题

Qwen3.5-9B 默认启用 thinking。对同一张 Habitat 室内图像：

- 160 completion tokens：输出被截断，未出现要求的 `Answer:`。
- 512 completion tokens：仍因长度结束，未出现 `Answer:`。
- 设置 `enable_thinking=false` 后：2.545 秒正常结束，74 completion tokens，输出包含 `Answer: yes`。

随后直接调用 BeliefMapNav 的 `OpenAI_API.detection_refinement(image, "chair")`，返回内容可被现有 `Answer:` 解析逻辑识别。这说明禁用 thinking 是本地 Qwen 适配的必要条件，而不是可选性能优化。

单帧 probe 只验证接口和输出格式，不代表目标确认准确率。原始图像不进入 Git；服务器记录仅保存图像 SHA256 `2013d96d1b1c04d10728b678211b8899f9e8085f74c6b6c47d689d44c23a6c14`、响应摘要和耗时。

## Habitat 0.2.5 诊断

- 输出目录：`/home/zyq/BeliefMapNav-repro/c9800d0/outputs/qwen35_h025_smoke20_20261104`
- episode：索引 0，场景 `TEEsavR23oF`
- 步数上限：20
- 退出码：0
- episode 用时：139.93 秒（不包括 Qwen 服务冷启动）
- `success=0`、`spl=0`、`target_detected=0`
- 导航期间 Qwen 请求数：0

运行进入 3D 信息增益和 explore 路径，动作序列与此前相同的固定 episode 诊断一致。由于没有检测到目标，VQA 没有自然触发；该运行只能证明 Qwen 适配没有破坏 Habitat/BeliefMap 主链路。

服务器证据目录包含不可覆盖的 `stdout.log`、`stderr.log`、`exit_code.txt`、`final_result.txt`、逐帧输出和 `provenance.txt`。HM3D val manifest SHA256 为 `96950bd7d4ef4c6dfc7e89842d9232559b447d8cfdbf10fd2a644bb19550f95c`。

## 下一步实验边界

先在训练场景构造带人工标签的目标确认开发样本，比较 `VQA_OFF` 与 Qwen3.5-9B 的误报率、漏报率、解析成功率、耗时和显存。只有在开发样本上确认目标复核有正收益后，才运行多 episode 导航对照。GPT-4o 若以后可用，应作为独立论文基线，不与 Qwen 替代结果混合汇总。
