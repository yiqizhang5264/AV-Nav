#!/usr/bin/env bash
set -euo pipefail

ENV_PREFIX="${QWEN_VLLM_ENV_PREFIX:-/home/zyq/miniconda3/envs/strive-qwen}"
CONDA="${CONDA_EXE:-/home/zyq/miniconda3/bin/conda}"
VLLM_VERSION="${QWEN_VLLM_VERSION:-0.29.0}"

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  "$CONDA" create -y -p "$ENV_PREFIX" python=3.12 pip
fi

"$ENV_PREFIX/bin/python" -m pip install "vllm==$VLLM_VERSION"
"$ENV_PREFIX/bin/python" - <<'PY'
import torch
import vllm
print({"vllm": vllm.__version__, "torch": torch.__version__, "cuda": torch.version.cuda})
PY
