#!/usr/bin/env bash
set -euo pipefail

ROOT="${AV_NAV_ROOT:-/home/zyq/AV-Nav}"
PYTHON="${STRIVE_PYTHON:-/home/zyq/miniconda3/envs/strive/bin/python}"
ENV_FILE="${STRIVE_ENV_FILE:-/home/zyq/.config/av-nav/strive.env}"
RUN_ID="${1:-strive_components_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="$ROOT/runs/$RUN_ID"

if [[ -e "$RUN_ROOT" ]]; then
  echo "refusing to overwrite $RUN_ROOT" >&2
  exit 2
fi
mkdir -p "$RUN_ROOT"

set -a
source "$ENV_FILE"
set +a
export CUDA_VISIBLE_DEVICES="${STRIVE_GPU:-0}"

git -C "$ROOT" rev-parse HEAD > "$RUN_ROOT/av_nav_commit.txt"
git -C "$ROOT/external/strive" rev-parse HEAD > "$RUN_ROOT/strive_commit.txt"
cp "$ROOT/runs/strive_resources/resource_manifest.sha256" \
  "$RUN_ROOT/resource_manifest.sha256"

set +e
"$PYTHON" "$ROOT/scripts/strive_component_smoke.py" \
  > "$RUN_ROOT/console.log" 2>&1
CODE=$?
set -e
printf '%s\n' "$CODE" > "$RUN_ROOT/exit_code.txt"
cat "$RUN_ROOT/console.log"
exit "$CODE"
