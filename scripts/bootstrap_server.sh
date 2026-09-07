#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLFM_SOURCE="${VLFM_SOURCE:-/home/zyq/vlfm}"
git -C "$ROOT" submodule update --init --recursive
for item in data yolov7; do
  if [[ ! -e "$ROOT/external/vlfm/$item" ]]; then
    ln -s "$VLFM_SOURCE/$item" "$ROOT/external/vlfm/$item"
  elif [[ "$item" == data && -d "$ROOT/external/vlfm/data" && ! -L "$ROOT/external/vlfm/data" ]]; then
    # Upstream tracks PointNav weights under data; link only missing assets.
    for asset in "$VLFM_SOURCE/data/"*; do
      name="$(basename "$asset")"
      [[ -e "$ROOT/external/vlfm/data/$name" ]] || ln -s "$asset" "$ROOT/external/vlfm/data/$name"
    done
  fi
done
mkdir -p "$ROOT/runs"
echo "Ready: $ROOT; reuse existing vlfm interpreter via scripts/server.env.example"
