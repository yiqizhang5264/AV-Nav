# STRIVE 复现执行记录（2026-09-16）

## 当前结论

STRIVE 官方代码、隔离环境、官方权重、HM3D v2 episode 数据和 HM3D v0.2 场景均已部署到服务器。
视觉模型组件与 Habitat 单 episode 初始化分别通过。完整 STRIVE episode 尚未启动，唯一缺少的运行输入是
`GEMINI_API_KEY`；预检已按失败退出记录，不能把它计为有效实验。

## 固定提交

- AV-Nav：`e27a0d1f14a8f7a01c51f18ec145a97c460df910`（Habitat 冒烟运行时）。
- STRIVE：`1872d73b7db297705d251df73bf5f08ffed0d749`。
- Habitat-Sim：`1075d95dde5605957aa3ce3792718b8edabb784f`。
- Habitat-Lab：`cb02f030655f9a475b379ec8d269979d9e17688d`。
- MMDetection：`cfd5d3a985b0249de009b67d04f37263e11cdf3d`。

## 资源校验

- SAM ViT-H：`a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e`。
- MM-GroundingDINO Swin-L：`34dcdc535a0eb020deda881fe8b4c49b324defe764d85f94d8282fb287cd0208`。
- `objectnav_hm3d_v2.zip`：`f551a0d8560804dc385c52f9aedb646c70917da36c1902d848b4fbc1615515d0`。
- HM3D v2 val：36 个场景分片，1000 episodes；入口文件 SHA-256 为
  `e0f25e8224b7b3931bd1b505482e35d343ebad8a52988c72ae94c3bd6a6b5169`。

## 服务器运行记录

- `runs/strive_hm3dv2_preflight_20260916/`：退出码 2。代码、依赖、权重、数据和场景检查通过；
  `GEMINI_API_KEY=false`，因此未冒充完整运行。
- `runs/strive_components_20260916/`：失败，定位到缺少 `fairscale`。
- `runs/strive_components_fairscale_20260916/`：失败，定位到缺少 `transformers`。
- `runs/strive_components_transformers_20260916/`：失败，未固定的 Transformers 5.17 删除了
  MMDetection 3.3 使用的 API。
- `runs/strive_components_transformers4352_20260916/`：退出码 0。GroundingDINO 前向推理通过，
  SAM 嵌入形状 `(1, 256, 64, 64)`，CUDA 峰值分配 7,398,238,720 bytes。
- `runs/strive_habitat_hm3dv2_20260916/`：退出码 0。成功重置 episode 25、加载
  `00861-GLAQ4DNUx5U`，得到 RGB、depth、objectgoal、compass 和 GPS 观测。

所有 `runs/` 内容仅保存在服务器且不进入 Git。安装问题及修复已经固化在脚本中，后续新环境无需重复人工排查。
