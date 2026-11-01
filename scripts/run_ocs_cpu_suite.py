"""Bounded CPU-only historical probe suite, not an online navigation evaluation.

``run --detach`` freezes an immutable launch manifest and starts one supervisor.
Each selected episode is replayed in its own process, with at most two such
processes at once. Zero-exploration cases remain in the denominator. Summaries
distinguish full reconstruction from a limited interface smoke test.
"""
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


ROOT = Path(__file__).resolve().parents[1]
PINNED_VLFM = "584ed56008754fde7997d904983607def8328322"
SOURCE_FILES = (
    "scripts/run_ocs_cpu_suite.py", "scripts/cpu_ocs_probe.py",
    "scripts/replay_frontier_probe.py", "scripts/summarize_frontier_probe.py",
    "scripts/decision_trace.py", "av_nav/frontier_ocs.py",
)
THREAD_ENV = {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
              "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
              "NUMEXPR_NUM_THREADS": "1"}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def git_revision(path):
    result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                            text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"Cannot resolve git revision at {path}: {result.stderr}")
    return result.stdout.strip()


def validate_upstream(path):
    revision = git_revision(path)
    if revision != PINNED_VLFM:
        raise RuntimeError(f"VLFM must be pinned to {PINNED_VLFM}, got {revision}")
    # HEAD equality alone does not exclude edited/staged source. Untracked
    # resource symlinks and cached assets do not participate in this check.
    result = subprocess.run(["git", "-C", str(path), "diff", "HEAD", "--quiet"],
                            text=True, capture_output=True)
    if result.returncode == 1:
        raise RuntimeError("Pinned VLFM has tracked modifications")
    if result.returncode:
        raise RuntimeError(f"Cannot verify VLFM tracked source: {result.stderr}")
    return revision


def selected_indices(value, count):
    indices = list(range(count)) if value is None else [int(x) for x in value.split(",")]
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("case-indices must be nonempty and unique")
    if any(index < 0 or index >= count for index in indices):
        raise ValueError("case-index outside the frozen selection")
    return indices


def prepare(args):
    if not 1 <= args.workers <= 2:
        raise ValueError("CPU suite workers must be 1 or 2")
    if args.limit_steps is not None and args.limit_steps < 1:
        raise ValueError("limit-steps must be positive")
    if args.every_explore < 1:
        raise ValueError("every-explore must be positive")
    selection = Path(args.selection).resolve()
    upstream = Path(args.vlfm_root).resolve()
    cohort = read_json(selection)["episodes"]
    indices = selected_indices(args.case_indices, len(cohort))
    identities = [(cohort[i]["dataset"], cohort[i]["episode_key"]) for i in indices]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate physical episode identity in the selected cohort")
    source_hashes = {name: sha256(ROOT / name) for name in SOURCE_FILES}
    revision, upstream_revision = git_revision(ROOT), validate_upstream(upstream)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    frozen_selection = output / "selection.json"
    frozen_selection.write_bytes(selection.read_bytes())
    (output / "logs").mkdir()
    cases = []
    for index in indices:
        case = cohort[index]
        directory = output / f"case_{index:02d}"
        command = [sys.executable, str(ROOT / "scripts/cpu_ocs_probe.py"),
                   "--vlfm-root", str(upstream), "--selection", str(frozen_selection),
                   "--case-index", str(index), "--output-dir", str(directory),
                   "--every-explore", str(args.every_explore), "--with-ocs"]
        if args.limit_steps is not None:
            command += ["--limit-steps", str(args.limit_steps)]
        cases.append({"index": index, "dataset": case["dataset"],
                      "episode_key": case["episode_key"],
                      "stratum": case.get("ocs_stratum", case.get("pilot_stratum")),
                      "directory": str(directory), "command": command,
                      "stdout_path": str(output / "logs" / f"case_{index:02d}.log")})
    manifest = {"schema": "vlfm.ocs_cpu_suite.launch.v1", "created_at": utc_now(),
                "root": str(ROOT), "git_commit": revision,
                "source_sha256": source_hashes, "vlfm_root": str(upstream),
                "vlfm_commit": upstream_revision, "selection": str(frozen_selection),
                "selection_sha256": sha256(frozen_selection),
                "workers": args.workers, "limit_steps": args.limit_steps,
                "environment": THREAD_ENV, "cases": cases,
                "interpretation": "Frozen-input shadow selection only; no PointNav actions, GPU inference, new trajectory or SR estimate"}
    write_json(output / "manifest.json", manifest)
    return output


def validate_manifest(manifest):
    root = Path(manifest["root"])
    if git_revision(root) != manifest["git_commit"]:
        raise RuntimeError("Experiment checkout revision changed")
    if validate_upstream(manifest["vlfm_root"]) != manifest["vlfm_commit"]:
        raise RuntimeError("Pinned VLFM revision changed")
    for name, expected in manifest["source_sha256"].items():
        if sha256(root / name) != expected:
            raise RuntimeError(f"Experiment source changed: {name}")
    if sha256(manifest["selection"]) != manifest["selection_sha256"]:
        raise RuntimeError("Frozen selection changed")
    if not 1 <= manifest["workers"] <= 2:
        raise RuntimeError("Invalid frozen worker count")


def baseline_issues(summary, case, limit_steps):
    issues = []
    expected_status = "complete" if limit_steps is None else "complete_prefix"
    if summary.get("status") != expected_status:
        issues.append(f"probe_status={summary.get('status')}")
    acceptance = summary.get("baseline_acceptance", [])
    if len(acceptance) != 1:
        return issues + [f"baseline_acceptance_count={len(acceptance)}"]
    evidence = acceptance[0]
    if (evidence.get("dataset"), evidence.get("episode_key")) != (case["dataset"], case["episode_key"]):
        issues.append("baseline_identity_mismatch")
    decisions = evidence.get("frontier_decisions")
    if not isinstance(decisions, int) or decisions < 0:
        issues.append("invalid_frontier_decision_count")
    if evidence.get("exact_frontier_goals") != decisions:
        issues.append("baseline_goal_mismatch")
    if evidence.get("original_printed_scores_matched") != decisions:
        issues.append("baseline_printed_score_mismatch")
    if evidence.get("mismatches") != []:
        issues.append("baseline_mismatch_records")
    if limit_steps is None and not evidence.get("complete_episode_reconstruction"):
        issues.append("incomplete_full_episode_reconstruction")
    expected_steps = evidence.get("source_steps")
    if limit_steps is not None and isinstance(expected_steps, int):
        expected_steps = min(expected_steps, limit_steps)
    if evidence.get("replayed_steps") != expected_steps or expected_steps is None:
        issues.append("replayed_step_count_mismatch")
    if not summary.get("with_ocs"):
        issues.append("ocs_shadow_missing")
    return issues


def frozen_diagnostics(path):
    """Count the identity query once per sampled frozen state, not all variants.

    This supports legacy sampled-only logs and remains separate from newer
    all-exploration counters. Both old score-only and upstream-native geometry
    fallback reason labels are counted; this is not an all-exploration rate.
    """
    counts = Counter()
    reasons = Counter()
    if not Path(path).exists():
        return {"sampled_identity_queries": 0, "fallback_count": 0,
                "fallback_fraction": None, "reasons": {}, "missing_log": True}
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            identities = [item for item in row.get("variants", []) if item.get("variant") == "identity"]
            if len(identities) != 1 or "ocs" not in identities[0]:
                raise ValueError("Frozen state needs exactly one OCS identity query")
            diagnostics = identities[0]["ocs"]["diagnostics"]
            counts["sampled_identity_queries"] += 1
            reason = diagnostics.get("reason", "missing_reason")
            reasons[reason] += 1
            counts["fallback_count"] += int(reason.startswith("geometry_unavailable"))
    n = counts["sampled_identity_queries"]
    return {"sampled_identity_queries": n, "fallback_count": counts["fallback_count"],
            "fallback_fraction": counts["fallback_count"] / n if n else None,
            "reasons": dict(reasons), "missing_log": False}


def summarize(output):
    output = Path(output)
    manifest = read_json(output / "manifest.json")
    counts, variants, reasons = Counter(), {}, Counter()
    cases, accepted, rejected, pending, zero_exposure = [], [], [], [], []
    fallback_n, fallback_total = 0, 0
    all_ocs_calls, all_ocs_fallbacks = 0, 0
    all_ocs_known, all_ocs_unknown = [], []
    for case in manifest["cases"]:
        row = {key: case[key] for key in ("index", "dataset", "episode_key", "stratum")}
        exit_path = output / "logs" / f"case_{case['index']:02d}.exit.json"
        summary_path = Path(case["directory"]) / "summary.json"
        if not exit_path.exists():
            row.update(accepted=False, issues=["process_pending_or_running"], state="pending")
            pending.append(case["index"])
            cases.append(row)
            continue
        process = read_json(exit_path)
        row["process"] = process
        issues = [] if process.get("returncode") == 0 else [f"returncode={process.get('returncode')}"]
        if summary_path.exists():
            summary = read_json(summary_path)
            issues += baseline_issues(summary, case, manifest["limit_steps"])
            if summary.get("selection_sha256") != manifest["selection_sha256"]:
                issues.append("selection_hash_mismatch")
            row["probe_error"] = summary.get("error")
            row["counts"] = summary.get("counts", {})
            row["baseline_acceptance"] = summary.get("baseline_acceptance", [])
            try:
                diagnostics = frozen_diagnostics(Path(case["directory"]) / "frozen_probes.jsonl")
                row["sampled_ocs_diagnostics"] = diagnostics
                if diagnostics["missing_log"]:
                    issues.append("missing_frozen_probe_log")
                expected = summary.get("counts", {}).get("frozen_states_probed", 0)
                if diagnostics["sampled_identity_queries"] != expected:
                    issues.append("frozen_diagnostic_denominator_mismatch")
            except (ValueError, KeyError) as exc:
                issues.append(f"invalid_frozen_diagnostics={exc}")
                diagnostics = {}
            if not issues:
                accepted.append(case["index"])
                if not summary.get("counts", {}).get("explore_decisions", 0):
                    zero_exposure.append(case["index"])
                counts.update(summary.get("counts", {}))
                episode_counts = summary.get("counts", {})
                if "ocs_calls" in episode_counts and "ocs_geometry_fallback_calls" in episode_counts:
                    all_ocs_calls += episode_counts["ocs_calls"]
                    all_ocs_fallbacks += episode_counts["ocs_geometry_fallback_calls"]
                    all_ocs_known.append(case["index"])
                elif not episode_counts.get("explore_decisions", 0):
                    # No exploration is an observed zero, not missing OCS data.
                    all_ocs_known.append(case["index"])
                else:
                    all_ocs_unknown.append(case["index"])
                for name, values in summary.get("variants", {}).items():
                    variants.setdefault(name, Counter()).update(values)
                fallback_n += diagnostics.get("sampled_identity_queries", 0)
                fallback_total += diagnostics.get("fallback_count", 0)
                reasons.update(diagnostics.get("reasons", {}))
        else:
            issues.append("missing_probe_summary")
        if issues:
            rejected.append(case["index"])
        row.update(accepted=not issues, issues=issues, state="accepted" if not issues else "failed")
        cases.append(row)
    result = {"schema": "vlfm.ocs_cpu_suite.summary.v1", "updated_at": utc_now(),
              "status": "running" if pending else ("failed" if rejected else "complete"),
              "expected_cases": len(cases), "accepted_cases": accepted,
              "failed_cases": rejected, "pending_cases": pending,
              "zero_exploration_cases": zero_exposure,
              "full_episode_validation": manifest["limit_steps"] is None,
              "counts_accepted_cases_only": dict(counts),
              "variants_accepted_cases_only": {name: dict(value) for name, value in variants.items()},
              "ocs_sampled_identity_fallback": {"count": fallback_total, "denominator": fallback_n,
                  "fraction": fallback_total / fallback_n if fallback_n else None,
                  "sampling": "Frozen-state identity queries only, NOT all exploration decisions",
                  "reason_counts": dict(reasons)},
              "ocs_all_exploration_fallback": {"known_cases": all_ocs_known,
                  "unknown_cases": all_ocs_unknown, "count_known_cases": all_ocs_fallbacks,
                  "denominator_known_cases": all_ocs_calls,
                  "fraction": (all_ocs_fallbacks / all_ocs_calls
                               if all_ocs_calls and not all_ocs_unknown else None),
                  "all_accepted_cases_instrumented": not all_ocs_unknown,
                  "sampling": "All shadow exploration calls when probe supplies counters; absent counters are unknown, never zero"},
              "selection_sha256": manifest["selection_sha256"], "git_commit": manifest["git_commit"],
              "cases": cases, "actual_pointnav_executed": False, "new_navigation_rollout": False,
              "interpretation": manifest["interpretation"]}
    write_json(output / "suite_summary.json", result)
    return result


def run_case(case, environment):
    start = time.monotonic()
    record = {"started_at": utc_now(), "command": case["command"]}
    exit_path = Path(case["stdout_path"]).with_suffix(".exit.json")
    try:
        with Path(case["stdout_path"]).open("x", encoding="utf-8") as stdout:
            child = subprocess.Popen(case["command"], stdout=stdout, stderr=subprocess.STDOUT,
                                     env=environment, cwd=str(ROOT))
            record["pid"] = child.pid
            write_json(Path(case["stdout_path"]).with_suffix(".process.json"), record)
            record["returncode"] = child.wait()
    except Exception as exc:
        record.update(returncode=None, launch_error=f"{type(exc).__name__}: {exc}")
    record.update(finished_at=utc_now(), elapsed_seconds=time.monotonic() - start)
    write_json(exit_path, record)
    return record


def worker(output):
    output = Path(output)
    # Exclusive claim prevents accidental resume/parallel execution against the
    # same output. A failed run is rerun in a new directory, never overwritten.
    with (output / "worker_claim.json").open("x", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "started_at": utc_now()}, stream)
    manifest = read_json(output / "manifest.json")
    validate_manifest(manifest)
    environment = os.environ.copy()
    environment.update(manifest["environment"])
    summarize(output)
    with ThreadPoolExecutor(max_workers=manifest["workers"]) as executor:
        futures = {executor.submit(run_case, case, environment): case for case in manifest["cases"]}
        for future in as_completed(futures):
            record = future.result()
            case = futures[future]
            print(f"CASE {case['index']} returncode={record['returncode']} elapsed={record['elapsed_seconds']:.2f}s", flush=True)
            summarize(output)
    result = summarize(output)
    write_json(output / "worker_exit.json", {"pid": os.getpid(), "finished_at": utc_now(),
                                             "status": result["status"]})
    return 0 if result["status"] == "complete" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    run = sub.add_parser("run")
    run.add_argument("--selection", required=True)
    run.add_argument("--vlfm-root", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--case-indices", help="Comma-separated fixed selection indices; default all")
    run.add_argument("--workers", type=int, default=2)
    run.add_argument("--limit-steps", type=int)
    run.add_argument("--every-explore", type=int, default=20)
    run.add_argument("--detach", action="store_true")
    for name in ("worker", "summarize"):
        command = sub.add_parser(name)
        command.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.action == "summarize":
        result = summarize(args.output_dir)
        print(json.dumps({key: result[key] for key in ("status", "accepted_cases", "failed_cases", "pending_cases")}, indent=2))
        return 0
    if args.action == "worker":
        return worker(args.output_dir)
    output = prepare(args)
    if not args.detach:
        return worker(output)
    command = [sys.executable, str(Path(__file__).resolve()), "worker", "--output-dir", str(output)]
    kwargs = {"start_new_session": True} if os.name != "nt" else {
        "creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP}
    environment = os.environ.copy()
    environment.update(THREAD_ENV)
    with (output / "supervisor.log").open("x", encoding="utf-8") as stdout:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout,
                                   stderr=subprocess.STDOUT, cwd=str(ROOT), env=environment, **kwargs)
    record = {"pid": process.pid, "command": command, "started_at": utc_now(),
              "output_dir": str(output), "manifest_sha256": sha256(output / "manifest.json")}
    write_json(output / "supervisor.json", record)
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
