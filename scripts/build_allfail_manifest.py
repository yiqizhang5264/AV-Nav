"""Read-only census of completed VLFM failures, including audit-ineligible cases.

Run on the server (stdout JSON) or use --capture-remote locally. No simulator,
policy, model, or original evidence file is modified. Selection is by dataset,
scene and Habitat-loaded episode ID, never by outcome. Stairs and missing-mode
cases remain in the denominator and carry explicit eligibility reasons.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

DEFAULT_ROOT = "/home/zyq/vlfm_evidence_runs/original_vlfm_584ed56_avnav_1224d7e_20261003"
STEP = re.compile(r"Step:\s*(\d+)\s*\|\s*Mode:\s*(\w+)\s*\|\s*Action:\s*(\d+)")
END = re.compile(r"Logging episode\s+\d+\s+to\s+.*?/episode_logs/(\d+)_([A-Za-z0-9]+)\.json")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scene_name(value):
    return Path(value).name.removesuffix(".basis.glb").removesuffix(".glb")


def parse_console(path):
    found = {}
    indices, modes, actions, nf, edge = [], [], [], [], []
    if not path.is_file():
        return found
    for line_no, line in enumerate(path.open(errors="replace"), 1):
        if "No frontiers found during exploration, stopping." in line:
            nf.append(line_no)
        if "Reached edge of map, stopping." in line:
            edge.append(line_no)
        match = STEP.search(line)
        if match:
            indices.append(int(match[1]))
            modes.append(match[2])
            actions.append(int(match[3]))
        match = END.search(line)
        if match:
            key = (match[2], match[1])
            entry = {"step_count": len(indices), "contiguous": indices == list(range(len(indices))),
                     "modes": modes, "actions": actions, "nf_message_lines": nf,
                     "edge_message_lines": edge, "end_line": line_no,
                     "repeated_key": key in found}
            found[key] = entry
            indices, modes, actions, nf, edge = [], [], [], [], []
    return found


def semantic_result(result):
    """Ignore recorder paths/timing but not any metric or observed outcome."""
    return json.dumps({"metrics": result.get("metrics"),
                       "steps_recorded": result.get("steps_recorded"),
                       "failure_cause": result.get("failure_cause")}, sort_keys=True)


def classify_terminal(success, rows, console, valid_console, steps, budget):
    last_action = int(rows[-1]["action"]) if rows else None
    if success:
        return "success", False
    strict_nf = bool(valid_console and len(console["nf_message_lines"]) == 1
                     and console["modes"][-1:] == ["explore"] and last_action == 0)
    if strict_nf:
        return "no_frontier_confirmed", True
    if console and console["edge_message_lines"] and last_action == 0:
        return "map_edge_exception_stop", False
    if steps >= budget and last_action != 0:
        return "stepout_nonstop", False
    if last_action == 0:
        if valid_console and console["modes"][-1:] == ["navigate"]:
            return "target_navigation_stop_failure", False
        return "other_stop_failure", False
    return "other_or_unknown_failure", False


def existing_steps(directory, extension):
    if not directory.is_dir():
        return set()
    return {int(p.stem) for p in directory.iterdir()
            if p.is_file() and p.suffix == extension and p.stem.isdigit()}


def scan_dataset(root, dataset):
    import yaml
    print(f"SCAN {dataset}: index completed records", file=sys.stderr, flush=True)
    snapshot_started = datetime.now(timezone.utc).isoformat()
    grouped = defaultdict(list)
    errors = []
    for shard in sorted((root / dataset / "shards").glob("*")):
        for attempt in sorted((shard / "attempts").glob("*")):
            for epdir in sorted((attempt / "evidence/episodes").glob("*")):
                result_path = epdir / "result.json"
                if not result_path.is_file():
                    continue
                try:
                    meta = json.loads((epdir / "episode.json").read_text())
                    result = json.loads(result_path.read_text())
                    key = (scene_name(meta["scene_id"]), str(meta["episode_id"]))
                    grouped[key].append((attempt, epdir, meta, result))
                except (ValueError, KeyError, OSError) as exc:
                    errors.append({"path": str(result_path), "error": repr(exc)})
    selected = {key: entries[-1] for key, entries in grouped.items()}
    duplicates = []
    for key, entries in sorted(grouped.items()):
        if len(entries) > 1:
            duplicate = {"scene": key[0], "loaded_episode_id": key[1],
                         "records": len(entries), "sources": [str(e[1]) for e in entries],
                         "semantic_result_conflict": len({semantic_result(e[3]) for e in entries}) != 1}
            duplicates.append(duplicate)
    attempts = sorted({entry[0] for entry in selected.values()})
    console = {at: parse_console(at / "console.log") for at in attempts}
    configs = {}
    for attempt in attempts:
        config_path = attempt / "hydra/.hydra/config.yaml"
        if config_path.is_file():
            config = yaml.safe_load(config_path.read_text())
            configs[attempt] = (config, sha256(config_path))
        else:
            configs[attempt] = ({}, None)
    counts = Counter(complete_unique_episodes=len(selected), complete_result_records=sum(map(len, grouped.values())),
                     duplicate_extra_records=sum(len(x)-1 for x in grouped.values()),
                     duplicate_conflicting_keys=sum(x["semantic_result_conflict"] for x in duplicates))
    all_category = Counter()
    episodes = []
    for index, (key, (attempt, epdir, meta, result)) in enumerate(sorted(selected.items())):
        success_value = result.get("metrics", {}).get("success")
        if success_value not in (0, 1, 0.0, 1.0):
            counts["unknown_success_value"] += 1
            errors.append({"path": str(epdir / "result.json"), "error": "success not binary", "success": success_value})
            continue
        success = bool(success_value)
        counts["success" if success else "failure"] += 1
        if success:
            continue
        if len(episodes) % 100 == 0:
            print(f"SCAN {dataset}: failure {len(episodes)} / completed {len(selected)}", file=sys.stderr, flush=True)
        reasons = []
        trace_path = epdir / "steps.jsonl"
        rows = []
        trace_hash = None
        try:
            data = trace_path.read_bytes()
            trace_hash = hashlib.sha256(data).hexdigest()
            rows = [json.loads(line) for line in data.splitlines() if line.strip()]
        except (ValueError, OSError) as exc:
            reasons.append("trace_unreadable_or_invalid")
            errors.append({"path": str(trace_path), "error": repr(exc)})
        steps = result.get("steps_recorded", 0)
        if len(rows) != steps or [r.get("step") for r in rows] != list(range(len(rows))):
            reasons.append("trace_count_or_contiguity_invalid")
        if not rows or rows[-1].get("done") is not True:
            reasons.append("trace_final_done_missing")
        c = console[attempt].get(key)
        valid_console = bool(c and c["contiguous"] and c["step_count"] == steps and not c["repeated_key"])
        if not valid_console:
            reasons.append("exact_console_mode_sequence_missing")
        elif rows and c["actions"] != [r["action"] for r in rows]:
            reasons.append("console_trace_action_mismatch")
            valid_console = False
        config, config_hash = configs[attempt]
        policy_cfg = config.get("habitat_baselines", {}).get("rl", {}).get("policy", {})
        if policy_cfg.get("name") not in ("HabitatITMPolicyV2", "HabitatITMPolicyV3"):
            reasons.append("policy_not_replay_supported")
        if not config_hash:
            reasons.append("missing_exact_config")
        budget = config.get("habitat", {}).get("environment", {}).get("max_episode_steps", 500)
        expected_steps = set(range(steps))
        availability = {}
        for folder, extension in (("rgb", ".png"), ("depth", ".npy"), ("sensors", ".npz")):
            present = existing_steps(epdir / folder, extension)
            missing = sorted(expected_steps - present)
            availability[folder] = {"present_expected": len(expected_steps & present), "missing_steps": missing}
            if missing:
                reasons.append(f"missing_{folder}_inputs")
        event_dir = epdir / "model_events"
        event_steps = Counter()
        if event_dir.is_dir():
            for event in event_dir.iterdir():
                match = re.fullmatch(r"(\d+)_(\d+)_blip2_itm", event.name)
                if match and (event / "event.json").is_file() and (event / "image.png").is_file():
                    event_steps[int(match[1])] += 1
        channels = len(policy_cfg.get("text_prompt", "").split("|"))
        bad_itm_steps = [step for step in range(steps) if event_steps[step] != channels]
        if bad_itm_steps:
            reasons.append("cached_itm_count_mismatch")
        category_set = sorted({r.get("policy_info", {}).get("target_object") for r in rows
                               if r.get("policy_info", {}).get("target_object")})
        category = category_set[0] if len(category_set) == 1 else "unknown_or_changed"
        all_category[category] += 1
        terminal, nf = classify_terminal(success, rows, c, valid_console, steps, budget)
        counts["terminal:" + terminal] += 1
        stairs = result.get("metrics", {}).get("traveled_stairs")
        counts["reported_stairs" if stairs else "no_reported_stairs"] += 1
        # Replay availability is not a geometry-validity certificate.
        counts["input_replay_eligible" if not reasons else "input_replay_ineligible"] += 1
        for reason in reasons:
            counts["ineligibility:" + reason] += 1
        geometry_reasons = []
        if stairs not in (0, 0.0):
            geometry_reasons.append("stairs_or_floor_ambiguity_without_world_height")
        if not rows:
            geometry_reasons.append("missing_pose_trace")
        conflict = len({semantic_result(e[3]) for e in grouped[key]}) != 1
        if conflict:
            geometry_reasons.append("duplicate_outcome_conflict")
        counts["strict_2d_input_eligible" if not (reasons or geometry_reasons) else "strict_2d_input_ineligible"] += 1
        basis = ".basis.glb" if dataset.startswith("hm3d") else ".glb"
        episodes.append({"dataset": dataset, "scene": key[0], "episode_id": key[1],
                         "episode_key": f"{key[0]}{basis}_{key[1]}",
                         "pilot_stratum": "all_completed_failures:" + terminal,
                         "original_success": False, "original_steps_recorded": steps,
                         "original_failure_at_500": steps == 500,
                         "original_strict_nf": nf, "terminal_class": terminal,
                         "original_last_action": rows[-1].get("action") if rows else None,
                         "original_last_mode": c["modes"][-1] if c and c["modes"] else None,
                         "original_mode_sequence": c["modes"] if valid_console else None,
                         "original_mode_counts": dict(Counter(c["modes"])) if c else {},
                         "original_category": category, "category_values": category_set,
                         "reported_stairs": stairs, "source_failure_cause_untrusted": result.get("failure_cause"),
                         "trace_source": str(trace_path), "source_trace_sha256": trace_hash,
                         "source_result_sha256": sha256(epdir / "result.json"),
                         "source_episode_meta_sha256": sha256(epdir / "episode.json"),
                         "config_source": str(attempt / "hydra/.hydra/config.yaml"),
                         "source_config_sha256": config_hash, "console_source": str(attempt / "console.log"),
                         "console_episode_end_line": c["end_line"] if c else None,
                         "console_nf_message_lines": c["nf_message_lines"] if c else [],
                         "console_edge_message_lines": c["edge_message_lines"] if c else [],
                         "input_replay_eligible": not reasons, "input_replay_ineligibility_reasons": reasons,
                         "strict_2d_input_eligible": not (reasons or geometry_reasons),
                         "geometry_ineligibility_reasons": geometry_reasons,
                         "input_availability": availability,
                         "cached_itm_event_count": sum(event_steps.values()),
                         "cached_itm_expected_channels": channels, "cached_itm_bad_steps": bad_itm_steps,
                         "duplicate_record_count": len(grouped[key]), "duplicate_result_conflict": conflict})
    # Failure class priority is independent of the hypothesized return mechanism.
    priority = {"stepout_nonstop": 0, "no_frontier_confirmed": 1, "target_navigation_stop_failure": 2}
    episodes.sort(key=lambda row: (priority.get(row["terminal_class"], 3), row["scene"], int(row["episode_id"])))
    counts["failure_steps"] = sum(row["original_steps_recorded"] for row in episodes)
    counts["replay_eligible_failure_steps"] = sum(row["original_steps_recorded"] for row in episodes if row["input_replay_eligible"])
    return {"dataset": dataset, "snapshot_started_utc": snapshot_started,
            "snapshot_finished_utc": datetime.now(timezone.utc).isoformat(),
            "full_dataset_expected_n": {"hm3dv1": 2000, "hm3dv2": 1000}.get(dataset),
            "partial_snapshot": dataset == "mp3d", "counts": dict(counts),
            "failure_categories": dict(all_category), "duplicate_keys": duplicates,
            "errors": errors, "episodes": episodes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--datasets", nargs="+", default=["hm3dv1", "hm3dv2", "mp3d"])
    parser.add_argument("--capture-remote", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.capture_remote:
        if not args.output:
            parser.error("--capture-remote requires --output")
        out = Path(args.output)
        if out.exists():
            raise SystemExit("Refusing to overwrite manifest")
        command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "yingu",
                   "/home/zyq/miniconda3/envs/vlfm/bin/python", "-u", "-", "--root", args.root,
                   "--datasets", *args.datasets]
        completed = subprocess.run(command, input=Path(__file__).read_text(encoding="utf-8"),
                                   stdout=subprocess.PIPE, text=True, encoding="utf-8", check=True)
        document = json.loads(completed.stdout)
        document["builder_source_sha256"] = sha256(__file__)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(document, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(out), "sha256": sha256(out),
                          "counts": {d["dataset"]: d["counts"] for d in document["datasets"]}}, indent=2))
        return
    datasets = [scan_dataset(Path(args.root), dataset) for dataset in args.datasets]
    document = {"schema": "vlfm.all_completed_failures.v1", "created_client_date": "2026-10-10",
                "root": args.root, "scope": "All completed failures; no failure removed for missing evidence or stairs",
                "dedup_rule": "Latest lexicographic shard/attempt/evidence completed record, never outcome-based",
                "eligibility_note": "Input files/counts only; exact replay validation still required. Stairs remain in cohort.",
                "terminal_note": "Explicit console NF branch plus explore STOP overrides non-STOP budget; historical failure_cause not trusted",
                "no_gpu": True, "read_only_source": True, "datasets": datasets,
                "episode_count": sum(len(d["episodes"]) for d in datasets),
                "episodes": [row for d in datasets for row in d["episodes"]]}
    # Avoid duplicated mode sequences while retaining per-dataset metadata.
    for dataset in document["datasets"]:
        dataset.pop("episodes")
    print(json.dumps(document, ensure_ascii=False, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
