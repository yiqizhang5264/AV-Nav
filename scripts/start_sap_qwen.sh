#!/usr/bin/env bash
# Dedicated, existing Qwen environment; never modify VLFM dependencies.
set -euo pipefail
OUT="$1"
mkdir "$OUT"
PY=/home/zyq/miniconda3/envs/strive-qwen/bin/python
MODEL=/home/zyq/AV-Nav/runs/qwen_hf_cache/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a
export CUDA_VISIBLE_DEVICES="${SAP_QWEN_GPU:-2}"
export LD_LIBRARY_PATH="/home/zyq/miniconda3/envs/strive-qwen/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HUB_OFFLINE=1
"$PY" -m pip freeze > "$OUT/environment.txt"
find -L "$MODEL" -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum > "$OUT/model_sha256.txt"
ARGS=("$PY" -m vllm.entrypoints.openai.api_server --model "$MODEL"
 --served-model-name Qwen/Qwen3.5-9B --host 127.0.0.1 --port 8000
 --dtype bfloat16 --gpu-memory-utilization 0.8 --max-model-len 16384
 --reasoning-parser qwen3 --max-num-seqs 4)
printf '%q ' "${ARGS[@]}" > "$OUT/command.txt"
exec "${ARGS[@]}" > "$OUT/server.log" 2>&1
