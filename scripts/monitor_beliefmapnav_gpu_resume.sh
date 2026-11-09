#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "Usage: $0 PHYSICAL_GPU SUITE_DIR WORKER_SCRIPT WORKER_NAME START:END [START:END ...]" >&2
  exit 2
fi

PHYSICAL_GPU="$1"
SUITE_DIR="$2"
WORKER_SCRIPT="$3"
WORKER_NAME="$4"
shift 4
RANGES=("$@")

POLL_SECONDS="${BELIEFMAPNAV_GPU_POLL_SECONDS:-300}"
STABLE_CHECKS="${BELIEFMAPNAV_GPU_STABLE_CHECKS:-2}"
STABLE_SECONDS="${BELIEFMAPNAV_GPU_STABLE_SECONDS:-60}"
MAX_IDLE_MEMORY_MIB="${BELIEFMAPNAV_GPU_IDLE_MEMORY_MIB:-1024}"
MAX_IDLE_UTILIZATION="${BELIEFMAPNAV_GPU_IDLE_UTILIZATION:-5}"
BELIEFMAP_ROOT="${BELIEFMAP_ROOT:-/home/zyq/BeliefMapNav-repro/c9800d0}"
EXPECTED_POLICY_SHA256="${BELIEFMAPNAV_POLICY_SHA256:-}"
RESUME_OF="${BELIEFMAPNAV_RESUME_OF:-}"
BELIEFMAP_COMMIT="${BELIEFMAPNAV_COMMIT:-unknown}"
ADAPTER_COMMIT="${BELIEFMAPNAV_ADAPTER_COMMIT:-unknown}"
AV_NAV_COMMIT="${AV_NAV_COMMIT:-unknown}"
DATASET_SHA256="${BELIEFMAPNAV_DATASET_SHA256:-unknown}"
QWEN_SNAPSHOT="${BELIEFMAPNAV_QWEN_SNAPSHOT:-unknown}"
HABITAT_SIM_VERSION="${BELIEFMAPNAV_HABITAT_SIM_VERSION:-unknown}"
REQUIRED_PORTS=(14181 14182 14183 14184 16000)

for integer in \
  "$PHYSICAL_GPU" "$POLL_SECONDS" "$STABLE_CHECKS" "$STABLE_SECONDS" \
  "$MAX_IDLE_MEMORY_MIB" "$MAX_IDLE_UTILIZATION"; do
  if [[ ! "$integer" =~ ^[0-9]+$ ]]; then
    echo "Expected a non-negative integer, got: $integer" >&2
    exit 2
  fi
done

if (( STABLE_CHECKS < 1 )); then
  echo "BELIEFMAPNAV_GPU_STABLE_CHECKS must be at least 1" >&2
  exit 2
fi
if [[ ! -r "$WORKER_SCRIPT" ]]; then
  echo "Worker script is missing or not readable: $WORKER_SCRIPT" >&2
  exit 2
fi
if [[ -e "$SUITE_DIR" ]]; then
  echo "Refusing to reuse existing suite directory: $SUITE_DIR" >&2
  exit 2
fi

POLICY_FILE="$BELIEFMAP_ROOT/vlfm/policy/base_objectnav_policy.py"
if [[ ! -f "$POLICY_FILE" ]]; then
  echo "BeliefMapNav policy file is missing: $POLICY_FILE" >&2
  exit 2
fi
if [[ -n "$EXPECTED_POLICY_SHA256" ]]; then
  actual_policy_sha256=$(sha256sum "$POLICY_FILE" | awk '{print $1}')
  if [[ "$actual_policy_sha256" != "$EXPECTED_POLICY_SHA256" ]]; then
    echo "Policy SHA256 mismatch: expected $EXPECTED_POLICY_SHA256, got $actual_policy_sha256" >&2
    exit 2
  fi
fi

gpu_uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader \
  | awk -F', ' -v index="$PHYSICAL_GPU" '$1 == index {print $2}')
if [[ -z "$gpu_uuid" ]]; then
  echo "GPU index does not exist: $PHYSICAL_GPU" >&2
  exit 2
fi

mkdir -p "$SUITE_DIR"
MONITOR_LOG="$SUITE_DIR/gpu_wait.log"
MONITOR_STATE="$SUITE_DIR/gpu_wait.status"
SUITE_MANIFEST="$SUITE_DIR/suite_manifest.txt"
exec > >(tee -a "$MONITOR_LOG") 2>&1

{
  echo "suite=$(basename "$SUITE_DIR")"
  echo "state=queued_waiting_for_gpu"
  echo "queued_at=$(date --iso-8601=seconds)"
  echo "worker_count=1"
  echo "habitat_gpu=$PHYSICAL_GPU"
  echo "save_video=0"
  echo "video_option=none"
  echo "ranges=${RANGES[*]}"
  echo "resume_of=$RESUME_OF"
  echo "beliefmap_commit=$BELIEFMAP_COMMIT"
  echo "adapter_commit=$ADAPTER_COMMIT"
  echo "av_nav_commit=$AV_NAV_COMMIT"
  echo "policy_sha256=$(sha256sum "$POLICY_FILE" | awk '{print $1}')"
  echo "habitat_sim=$HABITAT_SIM_VERSION"
  echo "dataset_sha256=$DATASET_SHA256"
  echo "qwen_snapshot=$QWEN_SNAPSHOT"
} > "$SUITE_MANIFEST"

{
  echo "state=waiting"
  echo "physical_gpu=$PHYSICAL_GPU"
  echo "gpu_uuid=$gpu_uuid"
  echo "poll_seconds=$POLL_SECONDS"
  echo "stable_checks=$STABLE_CHECKS"
  echo "stable_seconds=$STABLE_SECONDS"
  echo "max_idle_memory_mib=$MAX_IDLE_MEMORY_MIB"
  echo "max_idle_utilization=$MAX_IDLE_UTILIZATION"
  echo "worker_script=$WORKER_SCRIPT"
  echo "worker_name=$WORKER_NAME"
  echo "ranges=${RANGES[*]}"
  echo "beliefmap_root=$BELIEFMAP_ROOT"
  echo "policy_sha256=$(sha256sum "$POLICY_FILE" | awk '{print $1}')"
  echo "started_at=$(date --iso-8601=seconds)"
} > "$MONITOR_STATE"

gpu_is_idle() {
  local memory_used utilization process_count
  read -r memory_used utilization < <(
    nvidia-smi --id="$PHYSICAL_GPU" \
      --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits \
      | awk -F', ' '{print $1, $2}'
  )
  process_count=$(
    nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader \
      | awk -F', ' -v uuid="$gpu_uuid" '$1 == uuid {count++} END {print count + 0}'
  )
  echo "$(date --iso-8601=seconds) gpu=$PHYSICAL_GPU memory_mib=$memory_used utilization=$utilization compute_processes=$process_count"
  (( process_count == 0 )) \
    && (( memory_used <= MAX_IDLE_MEMORY_MIB )) \
    && (( utilization <= MAX_IDLE_UTILIZATION ))
}

services_are_ready() {
  local port
  for port in "${REQUIRED_PORTS[@]}"; do
    if ! timeout 3 bash -c "</dev/tcp/127.0.0.1/$port" 2>/dev/null; then
      echo "$(date --iso-8601=seconds) service_port_unavailable=$port"
      return 1
    fi
  done
}

echo "Waiting for physical GPU $PHYSICAL_GPU ($gpu_uuid)"
while true; do
  stable=0
  while (( stable < STABLE_CHECKS )); do
    if gpu_is_idle && services_are_ready; then
      stable=$((stable + 1))
      echo "$(date --iso-8601=seconds) stable_idle_check=$stable/$STABLE_CHECKS"
      if (( stable < STABLE_CHECKS )); then
        sleep "$STABLE_SECONDS"
      fi
    else
      stable=0
      break
    fi
  done

  if (( stable == STABLE_CHECKS )); then
    break
  fi
  sleep "$POLL_SECONDS"
done

{
  echo "state=launching"
  echo "physical_gpu=$PHYSICAL_GPU"
  echo "gpu_uuid=$gpu_uuid"
  echo "worker_name=$WORKER_NAME"
  echo "ranges=${RANGES[*]}"
  echo "policy_sha256=$(sha256sum "$POLICY_FILE" | awk '{print $1}')"
  echo "launching_at=$(date --iso-8601=seconds)"
} > "$MONITOR_STATE"

echo "GPU $PHYSICAL_GPU is stably idle and all services are reachable; launching evaluation"
exec bash "$WORKER_SCRIPT" "$WORKER_NAME" "$PHYSICAL_GPU" "$SUITE_DIR" "${RANGES[@]}"
