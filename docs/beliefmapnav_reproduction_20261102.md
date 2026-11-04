# BeliefMapNav 复现记录（2026-11-02）

## 结论

已在服务器上使用 **Habitat-Sim 0.2.5** 和 Habitat-Lab 0.2.5 打通 BeliefMapNav 的 20 步诊断链路，包括 RGB-D 观测、SAM2、RedNet、CLIP、3D voxel map、信息增益计算、frontier 选择和局部动作。该运行是兼容性诊断，不是论文 SR/SPL 的复现结果。

官方提交不能按 README 原样运行。依次确认了硬编码权重路径、PyTorch/SAM2 依赖冲突、Ada GPU 与旧 CuPy/自定义 CUDA 扩展冲突，以及 `state.py` 的确定性 `NameError`。为保留上游，所有修改均以显式 patch 记录，没有改动 `external/vlfm`。

完整效能复现目前还缺两个条件：

1. 论文行为所需的可用 GPT-4o 兼容 API 凭据；仓库中只有占位符，凭据不能进入 Git 或运行日志。
2. 一个与论文相符且能在 RTX 6000 Ada 上工作的 GroundingDINO GPU 依赖组合。当前诊断改用固定提交 `4875db77d7f489b7672f1716c1ee01236258bd29` 的 CPU DINO 适配器，因此速度和显存不能与论文比较。

## 固定输入与环境

- BeliefMapNav 官方源码：`c9800d09c2e0f0203a0efbee976d421afade7456`
- 服务器源码快照：`/home/zyq/BeliefMapNav-repro/c9800d0`
- 源码归档 SHA256：`bcdfa3281340288d660acaadb5c79cd8f0e029ba6aaccf28874c4f8cb0b4fc22`
- Python：3.9
- Habitat-Sim：0.2.5，官方 headless/bullet conda 包 SHA256 `54e0ed9c6c179a23277a427aa6db9a665099519166c64c7a61359cfaca36753d`
- Habitat-Lab：0.2.5，提交 `17ec6b3c1d6e0f71d2834e62b52a308b567ad843`
- Habitat-Lab 源码归档 SHA256：`7f7c30a81b476590d48b1b375190bb779dd222f6bf4c8f4e0f897642d7e0f1a8`
- 官方要求的 `env_utils.py` 行为已应用；文件 SHA256：`dd08a085f0d97b3b123fa4839139470d9f3d0f7056187020a6761f39695ef07d`
- 诊断主进程：PyTorch 2.6.0+cu124、CuPy 13.4.1/cuda12x；这是为了验证源码链路的兼容层，不是论文的精确环境
- 最终合成配置：前进 0.25 m、转向 30°、500 步、Success 距离 0.1 m
- HM3D val manifest SHA256：`96950bd7d4ef4c6dfc7e89842d9232559b447d8cfdbf10fd2a644bb19550f95c`
- 首个场景分片 `TEEsavR23oF.json.gz` SHA256：`73c74356c2133973adbc0eea10b7064ae70f2032015884f08f773c02582589a6`
- 诊断 episode：ID `0`，场景 `00800-TEEsavR23oF`，目标 `toilet`

下载权重均保存在服务器任务目录或缓存中，不进入 Git。已记录的权重哈希包括：

- RedNet：`f94d1c62a73bc05690ae29200d3dbd033ff243e7ce91755d1cd928bde844f995`
- CLIP ViT-B/32：`40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`
- SAM2.1 base：`f1a9cf2dd69d84bb463b5ad98246d03e2d47a130a9295db0ec967e6cd95e2e47`

## 可审计适配

- `adapters/beliefmapnav/official_c9800d0_runtime.patch`：把 SAM2 作者机器绝对路径和 OpenAI 占位配置改为环境变量。
- `adapters/beliefmapnav/official_c9800d0_import_time.patch`：修复进入信息增益计算后必现的 `time` 未定义。
- `adapters/beliefmapnav/official_c9800d0_cpu_dino_timeout.patch`：CPU DINO 诊断时把 HTTP 单请求超时从 1 秒改为 60 秒，避免重复请求风暴；不属于效能基线。
- `adapters/beliefmapnav/groundingdino_torch26.patch`：仅用于诊断环境中重编译 GroundingDINO CUDA 扩展；不属于论文环境。
- `adapters/beliefmapnav/official_c9800d0_complete_split.patch`：把首段从 `[0:99]` 修正为 `[0:200]`，并在启动任务前验证八段连续覆盖 `[0, 2000)`。
- `adapters/beliefmapnav/official_c9800d0_unknown_category_schema.patch`：将未知类别的三层属性包装成与六类先验相同的 `attributes`/`probability` 字典；生成属性使用层内均匀概率，目标名称概率为 1.0，并拒绝空或畸形响应。
- `adapters/beliefmapnav/official_c9800d0_initialization_12_turns.patch`：在第 12 次 30° 左转时标记初始化完成，消除第 13 次额外旋转。

最终成功诊断中记录的 patch SHA256：

- runtime：`5f785afe35ee113ed68e953da92279a5f1be344e8e8d84e2b5c30595e2b611f4`
- CPU DINO timeout：`a1dcd86e2196cb3b4415ef8a8cb5f57e78cca96eb65ea9fc212cbc88174e800a`
- GroundingDINO torch 2.6：`31ceb89a17e92b948c43be3d4fb15e795964d5e177ed3336b6b6e07d20e9cde5`
- import time：`20cf3e2dfa0521f13bf8f3c8dce5fa56a5fbac0ef8ac43e5dbb7c6a71fa5afb6`
- complete split：`33e7575500338540c154c9b80618797c423508a972f96754e8377b2e34804eea`
- unknown category schema：`fda10c9eb2f0dbeb31471bb41770e0a611820e21b718d32311069f18c246b7e1`
- 12-turn initialization：`8a9b348698a079ccc867538d7c862ce841685822c914727620d72449d7711565`

## 运行证据

所有目录唯一命名且未覆盖。核心记录如下：

| 运行 | 结果 | 证据目录 |
|---|---|---|
| 官方源码 + Habitat-Sim 0.2.5 | 失败：SAM2 权重写死为作者机器路径 | `/home/zyq/BeliefMapNav-repro/c9800d0/outputs/repro_unmodified_h025_probe1_20261102` |
| runtime patch + PyTorch 1.12 | 失败：SAM2 需要 `scaled_dot_product_attention` | `.../outputs/repro_adapted_h025_smoke1_20261102` |
| PyTorch 2.6 兼容诊断 | 越过 SAM2；旧 CuPy/DINO CUDA 栈不支持 Ada 架构 | `.../outputs/repro_adapted_h025_smoke2_torch26_20261102` |
| CPU DINO，5 步 | exit 0；只覆盖初始化旋转，42.05 秒 | `.../outputs/repro_adapted_h025_smoke3_cpu_dino_20261102` |
| 20 步，未修 `time` | 第 13 步进入探索后 `NameError: time is not defined` | `.../outputs/repro_adapted_h025_smoke4_planner_20261102` |
| 20 步，修复 `time` | exit 0；覆盖 3D 信息增益与探索动作，201.72 秒 | `.../outputs/repro_adapted_h025_smoke5_import_time_20261102` |

最终 20 步诊断的动作序列是：步骤 0–12 全部左转，步骤 13–15 左转，步骤 16–17 前进，步骤 18 右转，步骤 19 前进。episode 因人为 20 步上限结束，`success=0`、`SPL=0`、`target_detected=0`。这些数字不是效能结论。

测试中观察到 Open3D range map 从约 72 fragments 动态扩到 1592 fragments；兼容环境峰值显存约 32 GB。由于使用 CPU DINO、PyTorch 2.6 且仅单 episode，这些资源数据不能与论文声称的约 13 GB 直接比较，但说明官方实现存在明显的动态分配和部署敏感性。

任务专用 VLM tmux 会话已在测试后关闭；没有停止或修改服务器上原有的 VLFM 服务和其他用户进程。

## 已确认的实现与论文弱点

### 1. 官方安装说明内部矛盾

README 要求 PyTorch 1.12.1，却使用了依赖 PyTorch 2.x API 的 SAM2；`requirements.txt` 又固定 PyTorch 2.0.1。README 要安装不存在的 `environment.txt`（仓库实际提供 `requirements.txt`），评测命令还写成 `.scripts/launch_multi_eval.sh`。权重清单没有列出源码必需的 SAM2。

这不是轻微版本漂移：按 README 的 PyTorch 1.12 环境，第一帧 SAM2 就会崩溃。

### 2. 官方提交含必现运行错误

`vlfm/mapping/openfusion/state.py` 只在可选的 `torchshow` 导入成功时才顺带 `import time`。未安装可选调试包时，策略进入 `compute_info_gain()` 后调用 `time.time()`，必现 `NameError`。这说明论文核心的 3D 信息增益路径没有在干净发布环境中做过基本 smoke test。

### 3. “开放词汇/zero-shot”实现实际上是六类闭集优先

HM3D 的六个类别（couch、tv、bed、toilet、potted plant、chair）拥有手写房间、位置、共现物和距离先验。未知类别才调用 GPT-4o，但 `_get_object_attributes()` 返回 list，后续代码却按 `{"attributes": ..., "probability": ...}` 的 dict 读取，未知类别路径会类型错误。因此当前实现不能可靠支持其宣称的任意开放词汇目标。

### 4. 论文行为依赖未公开、非确定的 GPT-4o 调用

默认策略配置启用 VQA，检测后的目标复核/多目标选择会调用 GPT-4o；未知类别的属性和距离也调用 GPT-4o。官方提交没有可用凭据，也没有保存模型快照、请求/响应或确定性缓存。即使其余代码相同，API 模型更新也会改变结果。

不能用 `use_vqa=false` 的结果冒充论文基线；那只能作为明确标注的消融诊断。

### 5. 官方 2000-episode 启动脚本漏评 101 个 episode

脚本使用 Python 风格的尾端不包含切片，但区间从 `0:99` 后直接跳到 `200:400`。八段总计只覆盖 1899 个索引，漏掉 99–199 共 101 个 episode。若论文 HM3D 2000-episode 数字由另一套未公开脚本生成，当前仓库无法据此复核；若由该脚本生成，汇总口径本身就有误。

### 6. 论文与代码配置存在漂移

- 论文写 45,000 个 0.25 m voxels；发布代码实际打印 `block_count=50000`。
- 论文报告 HSSD 结果，但仓库只提供 HM3D/MP3D 评测配置，没有可直接复现 HSSD 的配置。
- 论文的 400-episode ablation 只说固定随机种子，没有给出种子和 episode ID；代码中的 simulated annealing、Python random 和 CuPy 随机性也未形成完整的可审计种子链。

### 7. 初始化和运行开销偏高

首个 episode 实测步骤 0–12 连续执行 13 次 30° 左转，即 390°，比完整环视多一次动作。官方动作枚举中 `2` 是左转、`3` 才是右转；此前把动作 `2` 记为右转属于报告表述错误，现已更正。之后每步还组合了 SAM2、RedNet、CLIP 多尺度特征、DINO/YOLO、Open3D raycasting 和 simulated annealing。论文没有报告每步延迟、吞吐或组件耗时，使得“效果提升是否值得计算代价”无法判断。

### 8. 论文自身结果已经暴露核心瓶颈

论文失败分析中约 48.32% 来自检测器误检/漏检，约 36.78% 来自跨楼层、楼梯和局部规划问题。后者尤其削弱“3D belief map 能解决多楼层导航”的核心叙事。消融中加入 object-level landmarks 后，SPL 还从 Scene+Region 的 32.0 降到 31.6，说明更细粒度先验并非稳定增益。

## 下一阶段建议

1. 用户在服务器当前 shell 中设置 `OPENAI_API_KEY` 和（如需要）`OPENAI_BASE_URL`，不要写入仓库、脚本或聊天记录。
2. 新建固定提交的专用 worktree/环境，优先重建 PyTorch 2.0.1 + 匹配 CUDA 的 GPU DINO；先做 10 个 development episode，不使用最终 validation 调参。
3. 保存逐 episode 指标、episode ID、请求缓存、环境 freeze、权重哈希和耗时；验证无崩溃后再运行完整 2000 episode。
4. 同时做三个明确标注的诊断消融：关闭 GPT VQA、移除手写六类先验、修复未知类别结构。这样可以区分 3D belief map 本身、人工先验和 GPT 复核分别贡献多少。
5. 修复官方切分脚本后重新生成精确 episode 清单；最终验证集只评一次，不根据结果调参。
