#!/usr/bin/env bash
set -euo pipefail

PYTHON="${QWEN_VLLM_PYTHON:-/home/zyq/miniconda3/envs/strive-qwen/bin/python}"
MODEL="${QWEN_MODEL:-Qwen/Qwen3.5-9B}"
SERVED_MODEL="${QWEN_SERVED_MODEL:-Qwen/Qwen3.5-9B}"
HOST="${QWEN_HOST:-127.0.0.1}"
PORT="${QWEN_PORT:-8000}"
GPU="${QWEN_GPU:-0}"
GPU_MEMORY="${QWEN_GPU_MEMORY_UTILIZATION:-0.65}"
MAX_MODEL_LEN="${QWEN_MAX_MODEL_LEN:-16384}"

export CUDA_VISIBLE_DEVICES="$GPU"
exec "$PYTHON" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "$SERVED_MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --dtype bfloat16 \
  --gpu-memory-utilization "$GPU_MEMORY" \
  --max-model-len "$MAX_MODEL_LEN"
