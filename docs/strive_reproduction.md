# STRIVE 复现记录

## 固定版本

- STRIVE：`igzat1no/STRIVE`，提交 `1872d73b7db297705d251df73bf5f08ffed0d749`。
- 作者 Habitat-Sim 分支 `release/v0.3.2`：提交 `1075d95dde5605957aa3ce3792718b8edabb784f`。
- 作者 Habitat-Lab 分支 `release/v0.3.2`：提交 `cb02f030655f9a475b379ec8d269979d9e17688d`。
- STRIVE 以 `external/strive` Git 子模块纳入 AV-Nav。上游代码保持原样；安装、预检与启动逻辑放在主仓库脚本中。

## 与现有 VLFM 环境隔离

STRIVE 官方代码要求 Python 3.12、PyTorch 2.5.1、作者的 Habitat 0.3.2 分支、SAM ViT-H 与 MMDetection GroundingDINO。现有 `/home/zyq/miniconda3/envs/vlfm` 是 Python 3.9、PyTorch 1.12.1 和 Habitat 0.2.4，不能用于等价复现。服务器使用独立环境 `/home/zyq/miniconda3/envs/strive`，依赖源码位于忽略的 `runs/strive_deps`。

## 服务器安装

```bash
cd /home/zyq/AV-Nav
git pull --ff-only
git submodule update --init --recursive external/strive
bash scripts/setup_strive_server.sh
```

权重、数据和密钥不进入 Git。服务器私有配置文件为 `/home/zyq/.config/av-nav/strive.env`：

```bash
export GEMINI_API_KEY="..."
export HABITAT_LAB_PATH="/home/zyq/AV-Nav/runs/strive_deps/habitat-lab"
export SAM_CHECKPOINT="/path/to/sam_vit_h_4b8939.pth"
export GROUNDING_DINO_PATH="/home/zyq/AV-Nav/runs/strive_deps/mmdetection"
export GROUNDING_DINO_CHECKPOINT="/path/to/grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth"
export HM3D_DATA_PATH="/path/to/strive-compatible-hm3d-root"
export STRIVE_GPU=0
```

`HM3D_DATA_PATH` 必须包含作者代码写死的相对路径：

- `objectnav_hm3d_v2/val/val.json.gz`
- `scene_datasets/hm3d_v0.2/hm3d_annotated_basis.scene_dataset_config.json`

不要用现有 HM3Dv1 episode 文件冒充 HM3Dv2。预检会记录数据文件 SHA-256。

## 单回合冒烟运行

```bash
cd /home/zyq/AV-Nav
bash scripts/run_strive_smoke.sh strive_hm3dv2_smoke1
```

原始输出保存在 `runs/strive_hm3dv2_smoke1/`，不推送 Git。提交哈希、依赖清单、数据哈希、控制台日志和退出码一并保存。冒烟运行只验证工程链路，不能作为论文复现结果。完整复现必须使用论文对应数据、模型、提示词、episode 数及官方指标，并报告 API 模型版本和调用失败。

## 当前边界

服务器已有 HM3D v0.2 场景和 HM3Dv1 ObjectNav episodes，但尚未发现作者所需路径下的 HM3Dv2 episodes；已有 GroundingDINO Swin-T 权重也不是 STRIVE 指定的 MM-GroundingDINO Swin-L。Gemini 密钥在检查时未设置。这些资源必须补齐后，正式 STRIVE episode 才能运行。
