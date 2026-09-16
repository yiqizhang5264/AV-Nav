# STRIVE 复现记录

## 固定版本

- STRIVE：`igzat1no/STRIVE`，提交 `1872d73b7db297705d251df73bf5f08ffed0d749`。
- 作者 Habitat-Sim `release/v0.3.2`：提交 `1075d95dde5605957aa3ce3792718b8edabb784f`。
- 作者 Habitat-Lab `release/v0.3.2`：提交 `cb02f030655f9a475b379ec8d269979d9e17688d`。
- MMDetection 3.3.0：提交 `cfd5d3a985b0249de009b67d04f37263e11cdf3d`。
- STRIVE 以 `external/strive` Git 子模块纳入 AV-Nav，上游算法代码保持原样。

服务器 `/usr/local/cuda` 指向 CUDA 12.8，而作者 Habitat-Sim fork 将 nvcc 写死为 CUDA 12.2。构建脚本自动应用
`patches/strive/habitat-sim-configurable-cuda.patch`，只允许外部传入编译器路径，不改变仿真或导航逻辑。

## 与 VLFM 环境隔离

STRIVE 官方代码要求 Python 3.12、PyTorch 2.5.1、作者的 Habitat 0.3.2 fork、SAM ViT-H 和
MM-GroundingDINO Swin-L。现有 `vlfm` 环境是 Python 3.9、PyTorch 1.12.1 和 Habitat 0.2.4，
不能用于等价复现。服务器使用独立环境 `/home/zyq/miniconda3/envs/strive`，依赖源码位于不跟踪的
`runs/strive_deps`。

## 服务器安装与资源准备

```bash
cd /home/zyq/AV-Nav
git pull --ff-only
git submodule update --init --checkout external/strive
bash -ic 'proxy_on; cd /home/zyq/AV-Nav; STRIVE_CONDA_CHANNEL=https://repo.anaconda.com/pkgs/main bash scripts/setup_strive_server.sh'
bash -ic 'proxy_on; cd /home/zyq/AV-Nav; bash scripts/download_strive_resources.sh'
```

环境构建已在服务器验证。MMCV 2.1.0 与 Habitat-Sim 0.3.2 都在本机源码编译，CUDA 可用。权重、
数据和密钥不进入 Git。资源默认保存到 `/home/zyq/AV-Nav/runs/strive_resources`；HM3D v0.2 场景
通过符号链接复用 `/home/zyq/vlfm/data/versioned_data/hm3d-0.2/hm3d`。

服务器私有配置文件为 `/home/zyq/.config/av-nav/strive.env`：

```bash
export GEMINI_API_KEY="..."
export HABITAT_LAB_PATH="/home/zyq/AV-Nav/runs/strive_deps/habitat-lab"
export SAM_CHECKPOINT="/home/zyq/AV-Nav/runs/strive_resources/weights/sam_vit_h_4b8939.pth"
export GROUNDING_DINO_PATH="/home/zyq/AV-Nav/runs/strive_deps/mmdetection"
export GROUNDING_DINO_CHECKPOINT="/home/zyq/AV-Nav/runs/strive_resources/weights/grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth"
export HM3D_DATA_PATH="/home/zyq/AV-Nav/runs/strive_resources/data"
export STRIVE_GPU=0
export STRIVE_GEMINI_MODEL="gemini-3.6-flash"
```

上游提交将 `gemini-2.5-flash` 写死在 `constants.py`。该模型对新 API 用户不可用时，
`scripts/run_strive_upstream.py` 在进程内应用 `STRIVE_GEMINI_MODEL`，不修改固定的 STRIVE
子模块。每次运行把实际模型名写入 `gemini_model.txt`。

HM3D v2 episode 数据来自 Habitat-Lab 官方 `objectnav_hm3d_v2.zip`，预期路径为：

- `objectnav_hm3d_v2/val/val.json.gz`
- `scene_datasets/hm3d_v0.2/hm3d_annotated_basis.scene_dataset_config.json`

预检记录代码提交、依赖提交、数据与权重 SHA-256，并且不输出 Gemini 密钥内容。

## 单 episode 冒烟运行

```bash
cd /home/zyq/AV-Nav
bash scripts/run_strive_smoke.sh strive_hm3dv2_smoke1
```

每次运行写入新的 `runs/<run-id>/`，保存预检、提交、依赖清单、控制台日志和退出码，拒绝覆盖已有目录。
单 episode 冒烟仅验证工程链路，不作为论文效果结果。正式复现还需固定 Gemini 模型版本、完整 episode
数量与官方指标，并报告 API 调用失败情况。
