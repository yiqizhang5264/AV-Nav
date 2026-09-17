# STRIVE：Gemini 3.6 Flash 与 Qwen3.5-9B 初步配对比较

本记录只比较固定 STRIVE 框架中的 VLM，不接入 VLFM。它是单 episode 工程与初步效果证据，
不能替代多 episode 统计结论。STRIVE 论文原配置是 Gemini 2.0 Flash；这里的两个模型都属于
替换实验。

## 固定条件

- STRIVE：`1872d73b7db297705d251df73bf5f08ffed0d749`，上游子模块无修改。
- 数据：HM3D ObjectNav v2 validation，episode 0，目标 `toilet`。
- Qwen 运行入口：AV-Nav `e75f5c7346ce59874d080ab6355762f9f4f10226`。
- Qwen：`Qwen/Qwen3.5-9B` BF16，vLLM 0.18.1，最大上下文 16384，GPU0 提供服务；
  STRIVE 在 GPU1 运行。
- Qwen3.5 默认 thinking 会使短结构化请求产生过长输出。本实验按模型官方接口设置
  `chat_template_kwargs.enable_thinking=false`，该配置已写入 `qwen_vlm_runtime.json`。
- 提示词、Pydantic 输出 schema、视觉检测与分割组件、导航配置均由同一固定 STRIVE
  提供。适配层只替换 API 地址、模型名和上述 Qwen 输出模式。

Qwen 的独立文本与图像结构化接口测试均通过，用时 2.44 秒。文本正确选择含床的房间 2，
图像正确判断测试图以红色为主。

## Episode 0 结果

| VLM | Success | SPL | 步数 | 起点到目标 | 行走距离 | 最终 DTG | episode 计时 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gemini 3.6 Flash（此前完整运行） | 1.0 | 0.470888 | 128 | 4.591166 m | 9.750004 m | 0.021273 m | 498.31 s |
| Qwen3.5-9B | 1.0 | 0.470888 | 128 | 4.591166 m | 9.750004 m | 0.021273 m | 303.49 s |

两组在这个 episode 的最终导航指标逐项一致。Qwen 的 episode 计时少 194.82 秒，约低
39.1%。该时间差是单回合观测，不能直接外推为稳定加速比；此前成功的 Gemini 运行早于
逐调用遥测代码，因此完整运行的 Gemini 调用延迟不可追溯。

## VLM 调用记录

Qwen 完整回合共调用 VLM 11 次，其中图像 7 次、文本 4 次，11 次全部成功。总 VLM 延迟
81.92 秒，均值 7.45 秒，中位数 2.45 秒，最大值 41.59 秒；累计 18078 个 prompt token、
3876 个 completion token。

为补齐 Gemini 遥测而发起的当前提交配对运行，在第 5 次请求被 Google 返回 429。前 4 次
成功调用的延迟合计 65.91 秒，均值 16.48 秒，中位数 9.93 秒，最大值 42.17 秒。失败原因
是 `gemini-3.6-flash` 免费层每日 20 次请求额度耗尽。这个部分运行没有最终导航指标，不能
把其部分延迟与 Qwen 的完整 11 次总延迟直接比较。

请求哈希显示两组前两个输入完全相同，第三个输入已经不同，说明两条轨迹的中间状态发生
分歧；后续请求不再天然一一对应。哈希只用于比对输入，不保存提示词、图像或密钥。

## 失败与诊断记录

- Qwen 首次接口测试使用默认 thinking，图像结构化请求长时间生成；人工终止并记录退出码
  143。关闭 thinking 后相同测试 2.44 秒通过。
- 第一次当前提交 Gemini 配对运行没有继承代理，在首个 API 请求处等待；终止并记录退出码
  143。
- 继承代理后的 Gemini 配对运行正常完成 4 次调用，第 5 次因每日免费额度耗尽退出，进程
  退出码 1。
- Qwen 完整运行退出码 0。启动时 Hugging Face 联网探测失败后使用了本地 BERT 缓存，未
  影响 episode 指标。

## 当前判断

Qwen3.5-9B 可以直接承担 STRIVE 的文本推理和图像核验接口。在已完成的 episode 0 上，它
保持了 Gemini 3.6 Flash 的成功结果，并以更短的 episode 时间完成；这足以支持继续做固定
小规模配对实验，还不足以声称 Qwen 在 STRIVE 上与 Gemini 等效或更优。

下一步应在 Gemini 额度恢复或启用计费后，用提交 `e75f5c7` 重跑 episode 0 以补齐完整遥测，
随后冻结按六个目标类别各 2 个 episode 的 12 回合配对集。两组必须使用完全相同的 episode
ID，报告 SR、SPL、DTG、调用失败率、延迟和 token，不根据任一模型结果更换样本。
