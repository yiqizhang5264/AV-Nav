"""Resumable low-priority CPU audit; every failure retained, never a new rollout."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def case_priority(item):
    index, case = item
    terminal = {"stepout_nonstop": 0, "no_frontier_confirmed": 1}.get(case.get("terminal_class"), 2)
    return (case["dataset"] == "mp3d", terminal, index)


def audit_one(index, case, args, output):
    from frontier_region_audit import analyze_episode
    from summarize_frontier_probe import read_jsonl, verify_printed_scores
    import numpy as np

    started = time.monotonic()
    case_root = output / "cases" / f"{index:04d}_{case['dataset']}_{case['episode_key']}"
    case_root.mkdir(parents=True, exist_ok=True)
    attempts = sorted(case_root.glob("attempt_*"))
    attempt = case_root / f"attempt_{len(attempts) + 1:03d}"
    attempt.mkdir()
    identity = dict(case_index=index, dataset=case["dataset"], episode_key=case["episode_key"],
                    terminal_class=case.get("terminal_class"), target=case.get("original_category"),
                    source_trace_sha256=case.get("source_trace_sha256"),
                    manual_review_status="not_reviewed", no_new_navigation_rollout=True,
                    attempt=str(attempt))
    save_json(case_root / "status.json", dict(identity, status="running", started_utc=datetime.now(timezone.utc).isoformat()))
    try:
        if digest(case["trace_source"]) != case["source_trace_sha256"]:
            raise ValueError("source trace hash mismatch")
        if digest(case["config_source"]) != case["source_config_sha256"]:
            raise ValueError("source config hash mismatch")
        evidence = Path(case["trace_source"]).parent
        if digest(evidence / "result.json") != case["source_result_sha256"]:
            raise ValueError("source result hash mismatch")
        command = [sys.executable, str(Path(__file__).with_name("replay_frontier_probe.py")),
                   "--vlfm-root", str(Path(args.vlfm_root).resolve()),
                   "--selection", str(Path(args.selection).resolve()), "--case-index", str(index),
                   "--output-dir", str(attempt / "replay"), "--save-audit-map", "--cpu-threads", "1"]
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
                           OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
        save_json(attempt / "command.json", dict(argv=command, environment_overrides={key: environment[key] for key in
                  ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}))
        with (attempt / "console.log").open("w") as stream:
            completed = subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT,
                                       timeout=1800)
        if completed.returncode:
            raise RuntimeError(f"CPU replay exit {completed.returncode}; retained console.log and partial evidence")
        summary = json.loads((attempt / "replay/summary.json").read_text())
        result = summary["cases"][0]
        if not all(result.get(name) for name in ("validated_to_end", "complete_episode_reconstruction", "all_frontier_decisions_exact")):
            raise ValueError("non-exact/incomplete reconstruction; not accepted as an analyzed case")
        replay_folder = next((attempt / "replay").glob("*/result.json")).parent
        rows = read_jsonl(replay_folder / "steps.jsonl")
        original = read_jsonl(case["trace_source"])
        scores = verify_printed_scores(original, rows)
        with np.load(replay_folder / "final_audit_map.npz", allow_pickle=False) as arrays:
            map_context = {key: arrays[key] for key in arrays.files}
        audit = analyze_episode(rows, source_rows=original, map_context=map_context)
        save_json(attempt / "region_audit.json", audit)
        # Candidate cases retain exact step->RGB/map pointers, not generated visual evidence.
        review = []
        for event in audit.get("events", []):
            frames = []
            for phase, step in (("previous_visit", event["previous_region_step"]),
                                ("return_start", event["return_transport_start_step"]),
                                ("arrival", event["arrival_step"]),
                                ("followup_end", event["followup_end_step"])):
                frames.append(dict(phase=phase, step=step, rgb=str(evidence / "rgb" / f"{step:04d}.png"),
                                   maps_directory=str(evidence / "maps"),
                                   exact_decision=rows[step].get("decision")))
            review.append(dict(event_index=event["event_index"], category=event["category"], frames=frames,
                               manual_review_status="not_reviewed"))
        save_json(attempt / "review_evidence.json", review)
        record = dict(identity, status="analyzed", episode_category=audit["episode_category"],
                      audit_status=audit["status"],
                      spatial_return_events=audit.get("spatial_return_events", 0),
                      low_yield_return_events=audit.get("low_yield_return_events", 0),
                      exact_frontier_decisions=result["exact_frontier_goals"],
                      original_printed_scores_matched=scores,
                      low_yield_union_observed_actions=audit.get("low_yield_union_observed_actions", 0),
                      low_yield_union_path_m=audit.get("low_yield_union_path_m", 0),
                      elapsed_seconds=time.monotonic() - started,
                      region_audit=str(attempt / "region_audit.json"),
                      replay_result=str(replay_folder / "result.json"))
    except Exception as exc:
        record = dict(identity, status="unknown_replay_or_audit_error", error=f"{type(exc).__name__}: {exc}",
                      elapsed_seconds=time.monotonic() - started)
    save_json(case_root / "status.json", record)
    return record


def summarize(records, total):
    counts = {}
    for record in records:
        count = counts.setdefault(record["dataset"], Counter())
        count["all_completed_failures"] += 1
        count[record["status"]] += 1
        if record["status"] == "analyzed":
            count[record["episode_category"]] += 1
            count["audit_insufficient_evidence"] += int(record.get("audit_status") == "insufficient_evidence")
            count["with_any_spatial_return"] += int(record["spatial_return_events"] > 0)
            count["with_operational_low_yield_return"] += int(record["low_yield_return_events"] > 0)
    return dict(updated_utc=datetime.now(timezone.utc).isoformat(), all_failed_cohort=total,
                no_causal_recoverability_claim=True, automated_audit_is_not_manual_video_review=True,
                by_dataset={key: dict(value) for key, value in counts.items()}, cases=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--vlfm-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=1, choices=(1, 2))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-cases", type=int, help="Smoke only; unfinished cohort stays pending")
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists() and not args.resume:
        raise FileExistsError("New output required, or explicit --resume with matching fingerprints")
    output.mkdir(parents=True, exist_ok=True)
    selection = json.loads(Path(args.selection).read_text())
    scripts = Path(__file__).parent
    files = [Path(__file__), scripts / "frontier_region_audit.py", scripts / "replay_frontier_probe.py",
             scripts / "summarize_frontier_probe.py", scripts / "decision_trace.py", scripts / "depth_camera_filtering.py",
             scripts.parent / "docs/VLFM_ALL_FAILURE_REVISIT_PROTOCOL_20261010.md"]
    upstream = Path(args.vlfm_root).resolve()
    files += [upstream / name for name in ("vlfm/policy/itm_policy.py", "vlfm/policy/utils/acyclic_enforcer.py",
              "vlfm/mapping/obstacle_map.py", "vlfm/mapping/value_map.py", "vlfm/mapping/base_map.py",
              "vlfm/utils/geometry_utils.py", "vlfm/utils/img_utils.py")]
    fingerprints = dict(selection_sha256=digest(args.selection), code={str(p.resolve()): digest(p) for p in files},
                        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=scripts.parent, text=True).strip(),
                        vlfm_git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=upstream, text=True).strip())
    lock = output / "fingerprints.json"
    if lock.exists() and json.loads(lock.read_text()) != fingerprints:
        raise ValueError("Refuse mixing different manifest, code, or frozen protocol")
    if not lock.exists():
        save_json(lock, fingerprints)
    records, queue = {}, []
    for index, case in enumerate(selection["episodes"]):
        status_path = output / "cases" / f"{index:04d}_{case['dataset']}_{case['episode_key']}" / "status.json"
        old = json.loads(status_path.read_text()) if status_path.exists() else None
        if old and old["status"] == "analyzed":
            records[index] = old
        elif not case.get("strict_2d_input_eligible", False):
            records[index] = dict(case_index=index, dataset=case["dataset"], episode_key=case["episode_key"],
                                  status="unknown_input_or_floor_evidence", terminal_class=case.get("terminal_class"),
                                  reasons=case.get("input_replay_ineligibility_reasons", []) + case.get("geometry_ineligibility_reasons", []))
        else:
            records[index] = dict(case_index=index, dataset=case["dataset"], episode_key=case["episode_key"], status="pending")
            queue.append((index, case))
    queue.sort(key=case_priority)
    if args.max_cases is not None:
        queue = queue[:args.max_cases]
    save_json(output / "progress.json", summarize([records[i] for i in sorted(records)], len(records)))
    print("COHORT", len(records), "QUEUE", len(queue), "WORKERS", args.workers, flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        tasks = {pool.submit(audit_one, index, case, args, output): index for index, case in queue}
        for future in as_completed(tasks):
            index = tasks[future]
            records[index] = future.result()
            save_json(output / "progress.json", summarize([records[i] for i in sorted(records)], len(records)))
            print("CASE", index, records[index]["dataset"], records[index]["episode_key"], records[index]["status"],
                  records[index].get("episode_category", records[index].get("error")), flush=True)
    print("QUEUE_COMPLETE", len(queue), flush=True)


if __name__ == "__main__":
    main()
