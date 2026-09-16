#!/usr/bin/env bash
set -euo pipefail

ROOT="${AV_NAV_ROOT:-/home/zyq/AV-Nav}"
CONDA="${CONDA_EXE:-/home/zyq/miniconda3/bin/conda}"
ENV_PREFIX="${STRIVE_ENV_PREFIX:-/home/zyq/miniconda3/envs/strive}"
DEPS_ROOT="${STRIVE_DEPS_ROOT:-/home/zyq/AV-Nav/runs/strive_deps}"
STRIVE="$ROOT/external/strive"

STRIVE_COMMIT="1872d73b7db297705d251df73bf5f08ffed0d749"
HABITAT_SIM_COMMIT="1075d95dde5605957aa3ce3792718b8edabb784f"
HABITAT_LAB_COMMIT="cb02f030655f9a475b379ec8d269979d9e17688d"
MMDETECTION_COMMIT="cfd5d3a985b0249de009b67d04f37263e11cdf3d"

test "$(git -C "$STRIVE" rev-parse HEAD)" = "$STRIVE_COMMIT"
mkdir -p "$DEPS_ROOT"

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  CONDA_ARGS=(create -y -p "$ENV_PREFIX" python=3.12 pip)
  if [[ -n "${STRIVE_CONDA_CHANNEL:-}" ]]; then
    CONDA_ARGS+=(--override-channels -c "$STRIVE_CONDA_CHANNEL")
  fi
  "$CONDA" "${CONDA_ARGS[@]}"
fi

PYTHON="$ENV_PREFIX/bin/python"
"$PYTHON" -m pip install --upgrade pip setuptools wheel
"$PYTHON" -m pip install -r "$STRIVE/requirements.txt"
"$PYTHON" -m pip install git+https://github.com/facebookresearch/segment-anything.git
"$PYTHON" -m pip install openmim
# openmim 0.3.9 pulls setuptools 60.2.0, which imports the removed
# pkgutil.ImpImporter symbol on Python 3.12. Setuptools 81+ then removes the
# pkg_resources module that openmim still imports, so keep the compatible band.
"$PYTHON" -m pip install --upgrade 'setuptools>=70,<81'
"$ENV_PREFIX/bin/mim" install mmengine
"$PYTHON" -m pip install fairscale
# Python 3.12 has no OpenMMLab wheel for this old MMCV release. Build in the
# current environment so setup can see torch and the compatible setuptools.
export CUDA_HOME="${STRIVE_CUDA_HOME:-/usr/local/cuda}"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export MMCV_WITH_OPS=1
export MAX_JOBS="${MAX_JOBS:-8}"
"$PYTHON" -m pip install ninja
"$PYTHON" -m pip install --no-build-isolation 'mmcv==2.1.0'

if [[ ! -d "$DEPS_ROOT/habitat-sim/.git" ]]; then
  git clone --branch release/v0.3.2 https://github.com/zwandering/habitat-sim.git "$DEPS_ROOT/habitat-sim"
fi
git -C "$DEPS_ROOT/habitat-sim" checkout --detach "$HABITAT_SIM_COMMIT"
test "$(git -C "$DEPS_ROOT/habitat-sim" rev-parse HEAD)" = "$HABITAT_SIM_COMMIT"
HABITAT_PATCH="$ROOT/patches/strive/habitat-sim-configurable-cuda.patch"
if git -C "$DEPS_ROOT/habitat-sim" apply --reverse --check "$HABITAT_PATCH" 2>/dev/null; then
  : # already applied
else
  git -C "$DEPS_ROOT/habitat-sim" apply --check "$HABITAT_PATCH"
  git -C "$DEPS_ROOT/habitat-sim" apply "$HABITAT_PATCH"
fi
if ! "$PYTHON" -c 'import habitat_sim' >/dev/null 2>&1; then
  "$PYTHON" -m pip install cmake ninja
  rm -rf "$DEPS_ROOT/habitat-sim/build"
  HEADLESS=1 WITH_CUDA=1 \
    CMAKE_ARGS="${CMAKE_ARGS:-} -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_CUDA_COMPILER=$CUDACXX" \
    "$PYTHON" -m pip install --no-build-isolation "$DEPS_ROOT/habitat-sim"
fi

if [[ ! -d "$DEPS_ROOT/habitat-lab/.git" ]]; then
  git clone --branch release/v0.3.2 https://github.com/zwandering/habitat-lab.git "$DEPS_ROOT/habitat-lab"
fi
git -C "$DEPS_ROOT/habitat-lab" checkout --detach "$HABITAT_LAB_COMMIT"
test "$(git -C "$DEPS_ROOT/habitat-lab" rev-parse HEAD)" = "$HABITAT_LAB_COMMIT"
"$PYTHON" -m pip install -e "$DEPS_ROOT/habitat-lab/habitat-lab"

if [[ ! -d "$DEPS_ROOT/mmdetection/.git" ]]; then
  git clone https://github.com/open-mmlab/mmdetection.git "$DEPS_ROOT/mmdetection"
fi
git -C "$DEPS_ROOT/mmdetection" checkout --detach "$MMDETECTION_COMMIT"
test "$(git -C "$DEPS_ROOT/mmdetection" rev-parse HEAD)" = "$MMDETECTION_COMMIT"
"$PYTHON" -m pip install --no-build-isolation -v -e "$DEPS_ROOT/mmdetection"

"$PYTHON" - <<'PY'
import habitat
import habitat_sim
import mmdet
import open3d
import segment_anything
import torch
print("python dependencies imported")
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("habitat", habitat.__file__)
print("habitat_sim", habitat_sim.__file__)
print("mmdet", mmdet.__version__)
PY
