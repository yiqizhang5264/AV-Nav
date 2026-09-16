#!/usr/bin/env bash
set -euo pipefail

ROOT="${AV_NAV_ROOT:-/home/zyq/AV-Nav}"
PYTHON="${STRIVE_PYTHON:-/home/zyq/miniconda3/envs/strive/bin/python}"
ENV_FILE="${STRIVE_ENV_FILE:-/home/zyq/.config/av-nav/strive.env}"
RUN_ID="${1:-strive_smoke_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="$ROOT/runs/$RUN_ID"

if [[ -e "$RUN_ROOT" ]]; then
  echo "refusing to overwrite $RUN_ROOT" >&2
  exit 2
fi
mkdir -p "$RUN_ROOT"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  source "$ENV_FILE"
  set +a
fi

"$PYTHON" "$ROOT/scripts/strive_preflight.py" \
  --root "$ROOT" --output "$RUN_ROOT/preflight.json"

git -C "$ROOT" rev-parse HEAD > "$RUN_ROOT/av_nav_commit.txt"
git -C "$ROOT/external/strive" rev-parse HEAD > "$RUN_ROOT/strive_commit.txt"
"$PYTHON" -m pip freeze > "$RUN_ROOT/environment.txt"

cd "$ROOT/external/strive"
export CUDA_VISIBLE_DEVICES="${STRIVE_GPU:-0}"
set +e
"$PYTHON" objnav_benchmark_with_process_obs.py \
  --eval_episodes 1 \
  --start_episode 0 \
  --save_dir "../../../runs/$RUN_ID/output" \
  > "$RUN_ROOT/console.log" 2>&1
CODE=$?
set -e
printf '%s\n' "$CODE" > "$RUN_ROOT/exit_code.txt"
exit "$CODE"
