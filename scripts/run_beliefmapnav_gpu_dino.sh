#!/usr/bin/env bash
set -euo pipefail

PYTHON="${BELIEFMAPNAV_DINO_PYTHON:-/home/zyq/venvs/beliefmapnav-c9800d0/bin/python}"
AV_NAV_ROOT="${AV_NAV_ROOT:-/home/zyq/AV-Nav}"
VLFM_ROOT="${BELIEFMAPNAV_VLFM_ROOT:-/home/zyq/AV-Nav-worktrees/vlfm-ocs-continue-4875db7/external/vlfm}"
DINO_ROOT="${BELIEFMAPNAV_DINO_ROOT:-/home/zyq/BeliefMapNav-repro/GroundingDINO-torch26}"
CONFIG="${BELIEFMAPNAV_DINO_CONFIG:-${DINO_ROOT}/groundingdino/config/GroundingDINO_SwinT_OGC.py}"
WEIGHTS="${BELIEFMAPNAV_DINO_WEIGHTS:-/home/zyq/vlfm/data/groundingdino_swint_ogc.pth}"
PORT="${GROUNDING_DINO_PORT:-14181}"
GPU="${BELIEFMAPNAV_DINO_GPU:-0}"

for path in "$PYTHON" "$AV_NAV_ROOT" "$VLFM_ROOT" "$DINO_ROOT" "$CONFIG" "$WEIGHTS"; do
  if [[ ! -e "$path" ]]; then
    echo "Required GroundingDINO resource is missing: $path" >&2
    exit 1
  fi
done

export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONPATH="${AV_NAV_ROOT}:${VLFM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

cd "$DINO_ROOT"
exec "$PYTHON" -u -m av_nav.dino_service \
  --port "$PORT" \
  --config "$CONFIG" \
  --weights "$WEIGHTS" \
  --device cuda
