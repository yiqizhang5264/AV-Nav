"""Read-only guard: low utilization alone is NOT an idle GPU.

No process is stopped and no job is launched. Use immediately before service
launch, after coordinating GPU use. A check is not a resource reservation.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import time


def query_idle(index: str, max_utilization=5, max_memory_mib=1024):
    command = ["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"]
    rows = list(csv.reader(io.StringIO(subprocess.check_output(command, text=True))))
    selected = next((r for r in rows if r[0].strip() == str(index)), None)
    if selected is None:
        raise ValueError(f"GPU {index} not found")
    uuid = selected[1].strip()
    process_text = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"], text=True)
    processes = [r for r in csv.reader(io.StringIO(process_text)) if r and r[0].strip() == uuid]
    utilization = int(selected[2].strip())
    memory_used = int(selected[3].strip())
    return {"gpu": str(index), "uuid": uuid, "utilization_percent": utilization,
            "memory_used_mib": memory_used, "memory_total_mib": int(selected[4]),
            "compute_processes": processes,
            "idle": not processes and utilization <= max_utilization and memory_used <= max_memory_mib}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", required=True)
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--interval", type=float, default=5)
    p.add_argument("--max-utilization", type=int, default=5)
    p.add_argument("--max-memory-mib", type=int, default=1024)
    args = p.parse_args()
    if args.samples < 1 or not 0 <= args.interval <= 30:
        p.error("samples must be positive; interval must be in [0,30]")
    if not 0 <= args.max_utilization <= 100 or args.max_memory_mib < 0:
        p.error("max-utilization must be in [0,100]; max-memory-mib must be nonnegative")
    records = []
    for i in range(args.samples):
        records.append(query_idle(args.gpu, args.max_utilization, args.max_memory_mib))
        if i + 1 < args.samples:
            time.sleep(args.interval)
    ready = all(r["idle"] for r in records)
    print(json.dumps({"ready": ready, "reservation": False, "samples": records}, indent=2))
    raise SystemExit(0 if ready else 3)


if __name__ == "__main__":
    main()
