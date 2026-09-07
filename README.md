# AV-Nav

基于 VLFM 的预算约束主动视觉验证实验。先在 VLFM 上验证，再与 DSVR-Nav 结合。

详细协议：[实验实施方案](docs/experiment_plan.md)。当前为 MVP：阈值未校准，导航效果尚待实验验证。

## 目录与环境

- 本地：`D:\zyq\AV_nav`
- 服务器：`/home/zyq/AV-Nav`，`ssh -p 50001 zyq@106.3.202.138`
- 服务器 Python：`/home/zyq/miniconda3/envs/vlfm/bin/python`
- 上游：[VLFM](https://github.com/bdaiinstitute/vlfm)，固定提交 `584ed56008754fde7997d904983607def8328322`，MIT 许可。

## 本地检查

```powershell
python -m pip install -e .
python -m unittest discover -s tests -v
git status
git add av_nav scripts configs docs tests
git commit -m "Describe the experiment change"
git push
```

## 服务器准备

```bash
cd /home/zyq/AV-Nav
git pull --ff-only
bash scripts/bootstrap_server.sh
source scripts/server.env.example
"$VLFM_PYTHON" -m unittest discover -s tests -v
"$VLFM_PYTHON" scripts/start_services.py --gpu 3 \
  --dino-config /home/zyq/miniconda3/envs/vlfm/lib/python3.9/site-packages/groundingdino/config/GroundingDINO_SwinT_OGC.py
```

端口已占用时先检查 `runs/services/pids.json` 和日志，不启动第二套相同端口服务。首次模型加载后确认服务就绪。

服务器原环境的 CUDA 11.3 与 RTX Ada 在 GroundingDINO 的 NVRTC 运算上不兼容，因此启动脚本默认仅将 DINO 放到 CPU。BLIP2、SAM、YOLOv7 与导航仍使用 GPU。所有对照共用同一服务设置；不改动现有 conda 环境。以后升级到兼容环境时可显式传 `--dino-device cuda` 并重新验证。

```bash
"$VLFM_PYTHON" scripts/check_services.py
```

## 训练集冒烟测试

```bash
"$VLFM_PYTHON" scripts/make_split.py \
  --source /home/zyq/vlfm/data/datasets/objectnav/hm3d/v1/train \
  --output runs/splits/hm3d_train_smoke5.json.gz --count 5
"$VLFM_PYTHON" scripts/launch_detached.py \
  --config configs/baseline.json --run-id baseline_train_smoke5 \
  --episodes 5 --max-steps 100 --split train \
  --dataset /home/zyq/AV-Nav/runs/splits/hm3d_train_smoke5.json.gz --gpu 3
```

对主动版本将配置替换为 `configs/active.json`，并使用新的 run-id。已有数据清单与 run-id 不覆盖。100 步测试仅验证工程链路；正式实验使用一致的完整任务预算。

```bash
tail -f runs/baseline_train_smoke5/console.log
"$VLFM_PYTHON" scripts/summarize.py runs/active_train_smoke5/episodes.jsonl \
  --baseline runs/baseline_train_smoke5/episodes.jsonl \
  --output results/smoke5_summary.json
```

## 双向 GitHub 同步

本地提交推送 → 服务器拉取运行 → 服务器提交结果摘要推送 → 本地拉取。服务器 origin 使用已有 SSH 身份，本地 origin 使用 HTTPS 凭据管理器。不存储服务器密码或 GitHub token。

```bash
python scripts/sync.py status
python scripts/sync.py pull
git add results/smoke5_summary.json
git commit -m "Record smoke evaluation summary"
python scripts/sync.py push
```

原始 `runs/`、数据、权重和视频不推送到 GitHub。同步工具拒绝脏工作区与非快进合并；先审查并提交，禁止强推覆盖另一端。运行期间不要更新工作目录，长实验用单独 worktree 固定代码。

## 实现边界

`baseline` 直接运行官方策略；其他策略共享裁剪 BLIP2 ITC 间隔验证器和候选入口。随机、最近与主动策略共享可行视点集合。MVP 使用局部点云关联、二维可见性和启发式距离，不宣称完整三维恢复或真实信息增益。原地组采用有界转向观测，不能称为完全静止视频。`unsuccessful_target_stops` 不等同于语义假阳性率。

每个运行记录提交、配置、数据哈希、逐回合指标、事件和退出码。仅完整成功退出且 episode 集一致的结果可用于配对比较。

`configs/diagnostic_force_review.json` 放宽工程诊断预算并强制进入模糊区间，用于执行路径检查，禁止纳入方法性能表。默认验证组会将候选裁剪保存到各自 `runs/.../evidence/`，便于训练集人工标签审计；图像不进入 Git。
