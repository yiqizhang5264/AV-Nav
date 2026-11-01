# OCS train-5 closed-loop preparation

This launcher changes only exploration selection. It uses the pinned upstream
VLFM evidence overlay, and retains target navigation and STOP behavior. Modes
are `original`, `anchor_only`, and `ocs`; none is called a bug-fix baseline.
OCS engineering defaults are diagnostic, not train-calibrated final parameters.

## Prerequisites

- Use a dedicated fixed-commit AV-Nav worktree, with pinned VLFM commit
  `584ed56008754fde7997d904983607def8328322`.
- Online VLFM needs `data/dummy_policy.pth`, `data/pointnav_weights.pth`, detector
  weights and a `yolov7` resource link. The CPU-only audit checkout does not have
  all of these. Do not mutate that audit checkout to run online evaluation.
- Python: `/home/zyq/miniconda3/envs/vlfm/bin/python`.
- Filtered train tasks: `/home/zyq/AV-Nav/runs/splits/train_smoke5.json.gz`.
- Scene assets: `/home/zyq/vlfm/data/scene_datasets`.
- DINO config: `/home/zyq/vlfm/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py`.
- Explicitly map DINO/ITM/SAM/YOLO to ports `14181/14182/14183/14184` for this
  run; never assume a listening server is the expected model service.

## Resource gate

Before starting services, run on each intended physical GPU:

```bash
PY=/home/zyq/miniconda3/envs/vlfm/bin/python
"$PY" scripts/check_gpu_idle.py --gpu 0 --samples 3 --interval 5
```

The guard requires no compute processes, utilization at most 5%, and memory
usage at most 1 GiB in every sample. It performs no launch/kill operation and is
not a reservation. Recheck immediately before launching. A user-owned service
started after this gate is expected to make that GPU non-idle; verify its PID
and configuration instead of trying to pass the idle gate again.

## Freeze each condition

Run in the new fixed-commit worktree. `UPSTREAM` must have complete online
resources; `RUN_ROOT` must be a new nonexisting experiment directory.

```bash
PY=/home/zyq/miniconda3/envs/vlfm/bin/python
UPSTREAM=/home/zyq/AV-Nav-worktrees/NEW_FIXED_WORKTREE/external/vlfm
RUN_ROOT=/home/zyq/vlfm_evidence_runs/ocs_train5_NEW_DATE
for MODE in original anchor_only ocs; do
  "$PY" scripts/run_vlfm_ocs.py prepare \
    --mode "$MODE" --run-dir "$RUN_ROOT/$MODE" \
    --dataset /home/zyq/AV-Nav/runs/splits/train_smoke5.json.gz \
    --vlfm-root "$UPSTREAM" \
    --scenes-dir /home/zyq/vlfm/data/scene_datasets \
    --gpu 0 --seed 17 --port-base 14181
done
```

Preparation loads no models. It copies the filtered task JSON to a filename
containing `hm3d` (required by upstream dataset inference), freezes source/task/
settings hashes, validates scene files, and saves full source identities.

## Execute explicitly

Only after tests, CPU reconstruction, resource checks, service health checks,
and frozen manifest review:

```bash
"$PY" scripts/run_vlfm_ocs.py execute \
  --run-dir "$RUN_ROOT/original" --resources-ready
```

Then run paired conditions sequentially with the same five tasks, seed, model
services and 500-step budget. Every attempt uses a new directory. Completion
requires zero exit status and exact physical task-identity multiset agreement;
counting five `result.json` files is insufficient. Source IDs may be re-numbered
by Habitat, so identity compares scene, target and initial pose rather than only
episode ID.

`ocs_events.jsonl` distinguishes predicted reset conditions from actual reset
callbacks. Target-mode calls preserve the original goal, `stop` argument and
returned action. This document is a preparation recipe, not evidence that any
episode has run or that efficacy improved.
