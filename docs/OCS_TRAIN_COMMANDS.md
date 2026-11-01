# OCS train smoke5：复用路径与命令

核查日期：2026-11-01。这里只准备命令；不代表已经启动或完成实验。

## 数据核对

源数据 `/home/zyq/AV-Nav/runs/splits/train_smoke5.json.gz`，SHA256：

`099070e6ef898d8222708b3f4a42b80702d3e433afdfc8460b7dd33e98c5227b`

源选样 seed=17。五条源任务（加载后的 ID 须另核）：

| 场景 | 源 episode_id | 类别 |
|---|---|---|
| 5biL7VEkByM | 15 | toilet |
| NEVASPhcrxR | 12 | chair |
| nACV8wLu1u5 | 67 | plant |
| gQ3xxshDiCz | 39 | sofa |
| W9YAR9qcuvN | 108 | bed |

`content_scenes_path={data_path}/__no_external_content__/{scene}.json.gz` 已禁用额外 scene content。任务文件中包括五条完整 payload 和对应 goals_by_category，不用仅数字 ID 匹配。

## 只读核验与生成不可变副本

变量 `REPO` 必须指本轮独立、固定 commit 的新工作区，下面的 `RUNROOT` 必须不存在，或用不同子目录避免覆盖。

```bash
PY=/home/zyq/miniconda3/envs/vlfm/bin/python
REPO=/home/zyq/AV-Nav-worktrees/REPLACE_WITH_FROZEN_OCS_WORKTREE
RUNROOT=/home/zyq/vlfm_evidence_runs/ocs_train_smoke5_20261101
cd "$REPO"
CUDA_VISIBLE_DEVICES='' "$PY" scripts/prepare_ocs_train.py \
  --dataset /home/zyq/AV-Nav/runs/splits/train_smoke5.json.gz \
  --source-manifest /home/zyq/AV-Nav/runs/splits/train_smoke5.json.manifest.json \
  --source-train /home/zyq/vlfm/data/datasets/objectnav/hm3d/v1/train \
  --scenes-dir /home/zyq/vlfm/data/scene_datasets \
  --output-dir "$RUNROOT/tasks" --verify-loader
```

脚本核对完整 task payload、源 row、目标集合、资产存在，并调用 Habitat dataset loader 核实实际加载的五条身份；不启动 simulator 或模型。

## 视觉服务

启服务之前再次确认 GPU 0/1 没有其他 compute process，不能只看瞬时低利用率。GPU 2/3 不在本轮资源范围内。由 root 统一执行，不同代理不重复启动。

现成模型路径在 `/home/zyq/vlfm/data`。独立 upstream checkout 需要复用其 data/yolov7 链接，不能替换已评估源码。

```bash
export GROUNDING_DINO_PORT=14181
export BLIP2ITM_PORT=14182
export SAM_PORT=14183
export YOLOV7_PORT=14184
"$PY" scripts/start_services.py --gpu 0 \
  --dino-config /home/zyq/vlfm/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py
"$PY" scripts/check_services.py
```

DINO 默认 CPU，避开现成旧 CUDA11.3/Ada 的 NVRTC 兼容问题。其它视觉服务放 GPU0，PointNav/模拟器用 GPU1。服务 PID、日志与健康检查必须保存；成功 fork 不等于模型已就绪。

## 准备并执行三个同键条件

```bash
"$PY" scripts/run_vlfm_ocs.py prepare --mode original \
  --run-dir "$RUNROOT/original" --dataset "$RUNROOT/tasks/hm3d_train_smoke5.json.gz" \
  --vlfm-root "$REPO/external/vlfm" --scenes-dir /home/zyq/vlfm/data/scene_datasets \
  --gpu 1 --seed 17 --port-base 14181
"$PY" scripts/run_vlfm_ocs.py execute --run-dir "$RUNROOT/original" --resources-ready
```

`anchor_only`、`ocs` 仅替换 mode 和独立 run-dir，其余保持完全相同。先 original 通过真实输入/输出核验，再执行实验条件；不因为 baseline 成败换任务。

当前 runner 冻结的关键 Hydra overrides：

```text
habitat_baselines.evaluate=true
habitat_baselines.num_environments=1
habitat_baselines.torch_gpu_id=0
habitat.seed=17
habitat.environment.max_episode_steps=500
habitat.environment.iterator_options.shuffle=False
habitat.environment.iterator_options.group_by_scene=False
habitat_baselines.test_episode_count=5
habitat_baselines.eval.split=train
habitat.dataset.split=train
habitat.dataset.data_path=<每个run内同字节hm3d_train_tasks.json.gz>
habitat.dataset.scenes_dir=/home/zyq/vlfm/data/scene_datasets
habitat_baselines.eval.video_option=[]
```

沿用默认 HM3D policy/config（HabitatITMPolicyV2），不得注入 room_online、STRIVE、主动复看等无关 adapter。五条不同训练场景都是预先选定的工程冒烟任务；不据此宣称全量 SR 提升。
