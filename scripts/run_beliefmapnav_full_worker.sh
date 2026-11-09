#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 WORKER_NAME PHYSICAL_GPU SUITE_DIR START:END [START:END ...]" >&2
  exit 2
fi

WORKER_NAME="$1"
PHYSICAL_GPU="$2"
SUITE_DIR="$3"
shift 3
RANGES=("$@")

BELIEFMAP_ROOT="${BELIEFMAP_ROOT:-/home/zyq/BeliefMapNav-repro/c9800d0}"
BELIEFMAP_PYTHON="${BELIEFMAP_PYTHON:-/home/zyq/venvs/beliefmapnav-c9800d0/bin/python}"
SAM2_WEIGHTS="${BELIEFMAPNAV_SAM2_WEIGHTS:-${BELIEFMAP_ROOT}/data/sam2.1_b.pt}"
QWEN_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:16000/v1}"
QWEN_MODEL="${BELIEFMAPNAV_LLM_MODEL:-Qwen/Qwen3.5-9B}"
SAVE_VIDEO="${BELIEFMAPNAV_SAVE_VIDEO:-0}"

if [[ "$SAVE_VIDEO" != "0" && "$SAVE_VIDEO" != "1" ]]; then
  echo "BELIEFMAPNAV_SAVE_VIDEO must be 0 or 1, got: $SAVE_VIDEO" >&2
  exit 2
fi

for required in "$BELIEFMAP_ROOT" "$BELIEFMAP_PYTHON" "$SAM2_WEIGHTS"; do
  if [[ ! -e "$required" ]]; then
    echo "Required resource is missing: $required" >&2
    exit 2
  fi
done

mkdir -p "$SUITE_DIR/workers" "$SUITE_DIR/shards"
WORKER_LOG="$SUITE_DIR/workers/${WORKER_NAME}.log"
WORKER_STATUS="$SUITE_DIR/workers/${WORKER_NAME}.status"

if [[ -e "$WORKER_STATUS" ]]; then
  echo "Refusing to overwrite existing worker status: $WORKER_STATUS" >&2
  exit 2
fi

exec > >(tee -a "$WORKER_LOG") 2>&1

echo "worker=$WORKER_NAME"
echo "physical_gpu=$PHYSICAL_GPU"
echo "ranges=${RANGES[*]}"
echo "started_at=$(date --iso-8601=seconds)"

for range in "${RANGES[@]}"; do
  if [[ ! "$range" =~ ^([0-9]+):([0-9]+)$ ]]; then
    echo "Invalid range: $range" >&2
    exit 2
  fi
  start="${BASH_REMATCH[1]}"
  end="${BASH_REMATCH[2]}"
  if (( start < 0 || end > 2000 || start >= end )); then
    echo "Range must be a non-empty subset of [0, 2000): $range" >&2
    exit 2
  fi

  count=$((end - start))
  shard_name=$(printf '%04d_%04d' "$start" "$end")
  shard_dir="$SUITE_DIR/shards/$shard_name"
  attempt_dir="$shard_dir/attempt_001"
  complete_marker="$shard_dir/complete.txt"

  if [[ -e "$shard_dir" ]]; then
    echo "Refusing to overwrite existing shard directory: $shard_dir" >&2
    exit 2
  fi
  mkdir -p "$attempt_dir"
  video_option_override="habitat_baselines.eval.video_option=[]"
  if [[ "$SAVE_VIDEO" == "1" ]]; then
    mkdir -p "$attempt_dir/videos"
    video_option_override="habitat_baselines.eval.video_option=[disk]"
  fi

  {
    echo "worker=$WORKER_NAME"
    echo "physical_gpu=$PHYSICAL_GPU"
    echo "start_split=$start"
    echo "end_split=$end"
    echo "expected_episodes=$count"
    echo "beliefmap_root=$BELIEFMAP_ROOT"
    echo "beliefmap_python=$BELIEFMAP_PYTHON"
    echo "qwen_base_url=$QWEN_BASE_URL"
    echo "qwen_model=$QWEN_MODEL"
    echo "save_video=$SAVE_VIDEO"
    echo "video_dir=$attempt_dir/videos"
    echo "grounding_dino_port=14181"
    echo "blip2itm_port=14182"
    echo "sam_port=14183"
    echo "yolov7_port=14184"
    echo "started_at=$(date --iso-8601=seconds)"
  } > "$attempt_dir/provenance.txt"

  echo "Starting shard $range on physical GPU $PHYSICAL_GPU at $(date --iso-8601=seconds)"
  set +e
  (
    cd "$BELIEFMAP_ROOT"
    export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU"
    export GROUNDING_DINO_PORT=14181
    export BLIP2ITM_PORT=14182
    export SAM_PORT=14183
    export YOLOV7_PORT=14184
    export BELIEFMAPNAV_SAM2_WEIGHTS="$SAM2_WEIGHTS"
    export OPENAI_API_KEY="${OPENAI_API_KEY:-local-dummy}"
    export OPENAI_BASE_URL="$QWEN_BASE_URL"
    export BELIEFMAPNAV_LLM_MODEL="$QWEN_MODEL"
    export BELIEFMAPNAV_DISABLE_THINKING=1
    export start_split="$start"
    export end_split="$end"
    export result_path="$attempt_dir"
    export gpu_id=0
    export gpu_num=1
    export HYDRA_FULL_ERROR=1

    /usr/bin/time -f 'ELAPSED_SECONDS=%e' "$BELIEFMAP_PYTHON" -u -m vlfm.run \
      "habitat_baselines.test_episode_count=$count" \
      habitat.environment.max_episode_steps=500 \
      "$video_option_override" \
      "habitat_baselines.video_dir=$attempt_dir/videos" \
      habitat_baselines.torch_gpu_id=0 \
      habitat_baselines.rl.policy.use_vqa=true \
      > "$attempt_dir/stdout.log" \
      2> "$attempt_dir/stderr.log"
  )
  exit_code=$?
  set -e
  printf '%s\n' "$exit_code" > "$attempt_dir/exit_code.txt"

  if (( exit_code != 0 )); then
    printf 'failed shard=%s exit_code=%s at=%s\n' \
      "$range" "$exit_code" "$(date --iso-8601=seconds)" > "$WORKER_STATUS"
    echo "Shard $range failed with exit code $exit_code" >&2
    exit "$exit_code"
  fi

  actual_count=$(grep -c '^episode stats:' "$attempt_dir/final_result.txt" || true)
  if (( actual_count != count )); then
    printf 'failed shard=%s expected=%s actual=%s at=%s\n' \
      "$range" "$count" "$actual_count" "$(date --iso-8601=seconds)" > "$WORKER_STATUS"
    echo "Shard $range exited zero but recorded $actual_count/$count episodes" >&2
    exit 3
  fi

  {
    echo "range=$range"
    echo "episodes=$actual_count"
    echo "exit_code=0"
    echo "completed_at=$(date --iso-8601=seconds)"
  } > "$complete_marker"
  echo "Completed shard $range with $actual_count episodes at $(date --iso-8601=seconds)"
done

printf 'complete worker=%s at=%s\n' "$WORKER_NAME" "$(date --iso-8601=seconds)" > "$WORKER_STATUS"
echo "Worker $WORKER_NAME completed all assigned shards"
