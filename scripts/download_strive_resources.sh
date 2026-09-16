#!/usr/bin/env bash
set -euo pipefail

ROOT="${AV_NAV_ROOT:-/home/zyq/AV-Nav}"
RESOURCE_ROOT="${STRIVE_RESOURCE_ROOT:-$ROOT/runs/strive_resources}"
WEIGHT_DIR="$RESOURCE_ROOT/weights"
DOWNLOAD_DIR="$RESOURCE_ROOT/downloads"
DATA_ROOT="$RESOURCE_ROOT/data"
SCENE_SOURCE="${STRIVE_HM3D_SCENE_SOURCE:-/home/zyq/vlfm/data/versioned_data/hm3d-0.2/hm3d}"

mkdir -p "$WEIGHT_DIR" "$DOWNLOAD_DIR" "$DATA_ROOT/scene_datasets"

download() {
  local url="$1"
  local output="$2"
  if [[ -f "$output" ]]; then
    return
  fi
  curl -L --fail --retry 5 --retry-delay 3 --continue-at - \
    --output "$output.part" "$url"
  mv "$output.part" "$output"
}

download \
  "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth" \
  "$WEIGHT_DIR/sam_vit_h_4b8939.pth"
download \
  "https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-l_pretrain_obj365_goldg/grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth" \
  "$WEIGHT_DIR/grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth"
download \
  "https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2.zip" \
  "$DOWNLOAD_DIR/objectnav_hm3d_v2.zip"

printf '%s  %s\n' \
  "a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e" \
  "$WEIGHT_DIR/sam_vit_h_4b8939.pth" \
  "34dcdc535a0eb020deda881fe8b4c49b324defe764d85f94d8282fb287cd0208" \
  "$WEIGHT_DIR/grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth" \
  "f551a0d8560804dc385c52f9aedb646c70917da36c1902d848b4fbc1615515d0" \
  "$DOWNLOAD_DIR/objectnav_hm3d_v2.zip" \
  | sha256sum --check --strict

if [[ ! -f "$DATA_ROOT/objectnav_hm3d_v2/val/val.json.gz" ]]; then
  unzip -tq "$DOWNLOAD_DIR/objectnav_hm3d_v2.zip"
  unzip -oq "$DOWNLOAD_DIR/objectnav_hm3d_v2.zip" \
    -d "$DATA_ROOT"
fi

SCENE_LINK="$DATA_ROOT/scene_datasets/hm3d_v0.2"
test -f "$SCENE_SOURCE/hm3d_annotated_basis.scene_dataset_config.json"
if [[ -L "$SCENE_LINK" ]]; then
  test "$(readlink -f "$SCENE_LINK")" = "$(readlink -f "$SCENE_SOURCE")"
elif [[ -e "$SCENE_LINK" ]]; then
  echo "refusing to replace existing $SCENE_LINK" >&2
  exit 2
else
  ln -s "$SCENE_SOURCE" "$SCENE_LINK"
fi

sha256sum \
  "$WEIGHT_DIR/sam_vit_h_4b8939.pth" \
  "$WEIGHT_DIR/grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth" \
  "$DOWNLOAD_DIR/objectnav_hm3d_v2.zip" \
  "$DATA_ROOT/objectnav_hm3d_v2/val/val.json.gz" \
  "$SCENE_LINK/hm3d_annotated_basis.scene_dataset_config.json" \
  > "$RESOURCE_ROOT/resource_manifest.sha256"

echo "STRIVE resources prepared under $RESOURCE_ROOT"
