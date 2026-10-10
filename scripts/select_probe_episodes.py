"""Deterministic diagnostic pilot selection from frozen JSON audits.

This is failure-enriched diagnostic sampling, not an unbiased efficacy cohort.
No services are started and no source evaluation data are changed.  Remote reads
recover exact Habitat task row identities because serialized original episode_id
values are not unique in these ObjectNav content shards.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess


DATASETS = ("hm3dv1", "hm3dv2")
REMOTE_RUN = "/home/zyq/vlfm_evidence_runs/original_vlfm_584ed56_avnav_1224d7e_20261003"
TASK_ROOTS = {
    "hm3dv1": "/home/zyq/vlfm/data/datasets/objectnav/hm3d/v1",
    "hm3dv2": "/home/zyq/AV-Nav/runs/strive_resources/data/objectnav_hm3d_v2",
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scene_from_path(value: str) -> str:
    return Path(value).name.split(".basis")[0]


def row_key(row: dict) -> tuple[str, str]:
    return row["scene"], str(row["episode_id"])


def witness_modes(row: dict, flag: dict | None) -> dict:
    if flag is None:
        return {}
    witness = flag["witness"]
    modes = row["evidence"]["modes"]
    start, end = witness["leave_step"], witness["return_step"]
    assert 0 <= start <= end < len(modes), (row_key(row), witness)
    return dict(sorted(Counter(modes[start : end + 1]).items()))


def select_dataset(dataset: str, source: Path) -> list[dict]:
    terminal = read_json(source / f"vlfm_{dataset}_terminal.json")
    exploratory = read_json(source / f"vlfm_{dataset}_exploratory.json")
    flags = {
        (scene_from_path(row["scene_id"]), str(row["episode_id"])): row
        for row in exploratory["flagged_episodes"]
    }
    assert len(flags) == len(exploratory["flagged_episodes"])
    terminals = terminal["episodes"]
    assert len({row_key(row) for row in terminals}) == len(terminals)
    eligible = [
        row for row in terminals
        if row["console_join_valid"]
        and row["evidence"]["contiguous_from_zero"]
        and row["stairs"] == 0
        and len(row["evidence"]["modes"]) == row["steps_recorded"]
    ]
    selected = []
    used_scenes = set()
    used_keys = set()
    groups = (
        "failure_budget500_return",
        "failure_strict_nf_return",
        "failure_budget500_no_return_control",
        "success_return_control",
    )
    for group in groups:
        candidates = []
        for row in eligible:
            flag = flags.get(row_key(row))
            is_nf = row["no_frontier_terminal_confirmed"]
            keep = {
                "failure_budget500_return": (
                    not row["success"] and row["failure_at_500"] and not is_nf and flag is not None
                ),
                "failure_strict_nf_return": (
                    not row["success"] and is_nf and not row["failure_at_500"] and flag is not None
                ),
                "failure_budget500_no_return_control": (
                    not row["success"] and row["failure_at_500"] and not is_nf and flag is None
                ),
                "success_return_control": row["success"] and flag is not None,
            }[group]
            if keep and row_key(row) not in used_keys:
                candidates.append(row)
        if not candidates:
            raise RuntimeError(f"No eligible row for {dataset}/{group}; do not silently relax criteria")

        def rank(row):
            flag = flags.get(row_key(row))
            modes = witness_modes(row, flag)
            pure_explore = bool(modes) and set(modes) == {"explore"}
            # Prefer exploration-specific witnesses, then different scenes,
            # then lexicographic scene / numeric loaded task index. No ranking
            # by rescue outcome or by longest/most severe return.
            return (not pure_explore if flag else False,
                    row["scene"] in used_scenes, row["scene"], int(row["episode_id"]))

        chosen = min(candidates, key=rank)
        key = row_key(chosen)
        flag = flags.get(key)
        evidence = chosen["evidence"]
        modes = witness_modes(chosen, flag)
        trace = chosen["step_trace"]
        assert f"/attempts/" in trace
        attempt = trace.split("/attempts/", 1)[1].split("/", 1)[0]
        record = {
            "dataset": dataset,
            "scene": key[0],
            "episode_id": key[1],
            "episode_key": f"{key[0]}.basis.glb_{key[1]}",
            "pilot_stratum": group,
            "eligible_stratum_size": len(candidates),
            "selection_rationale": (
                "Frozen 2026-10-09 diagnostic screen; failure-enriched, not population random sampling. "
                "Prefer first witness entirely in explore mode, then a scene not already selected in this dataset, "
                "then scene lexicographic and loaded episode index numeric. "
                "No intervention outcome inspected. Budget and NF strata are mutually exclusive."
            ),
            "original_success": chosen["success"],
            "original_failure_at_500": chosen["failure_at_500"],
            "original_strict_nf": chosen["no_frontier_terminal_confirmed"],
            "original_steps_recorded": chosen["steps_recorded"],
            "original_last_step_zero_based": evidence["last_step"],
            "original_last_mode": evidence["last_mode"],
            "original_last_action": evidence["last_action"],
            "original_mode_counts": evidence["mode_counts"],
            "original_mode_sequence": evidence["modes"],
            "reported_stairs": chosen["stairs"],
            "return_screen_hit": flag is not None,
            "first_return_witness": flag["witness"] if flag else None,
            "first_witness_mode_counts": modes,
            "first_witness_pure_explore": bool(modes) and set(modes) == {"explore"},
            "trace_source": trace,
            "attempt": attempt,
            "console_source": chosen["console"],
            "nf_console_line_numbers": evidence["nf_message_lines"],
            "console_episode_end_line": evidence["episode_end_line"],
            "task_content_path": f"{TASK_ROOTS[dataset]}/val/content/{key[0]}.json.gz",
            "task_loaded_index_in_content": int(key[1]),
            "loaded_index_note": (
                "Existing single-scene Habitat ObjectNav loader renumbers each content shard as str(i); "
                "do not filter source JSON by its potentially duplicated original episode_id. "
                "Runner must preserve loaded id, scene, start pose, category and goal set."
            ),
        }
        selected.append(record)
        used_scenes.add(key[0])
        used_keys.add(key)
    return selected


def enrich_remote(records: list[dict], host: str) -> dict:
    """Read task shards and previous run metadata only, through SSH stdin."""
    payload = json.dumps(records, ensure_ascii=True)
    remote_code = r'''
import gzip, hashlib, json, pathlib
records = json.loads(PAYLOAD)
output = []
for record in records:
    path = pathlib.Path(record["task_content_path"])
    raw = path.read_bytes()
    task = json.loads(gzip.decompress(raw))
    index = record["task_loaded_index_in_content"]
    row = task["episodes"][index]
    if pathlib.Path(row["scene_id"]).name.split(".basis")[0] != record["scene"]:
        raise RuntimeError("Selected index and scene disagree")
    folder = pathlib.Path(record["trace_source"]).parent
    episode = json.loads((folder / "episode.json").read_text())
    if str(episode["episode_id"]) != record["episode_id"]:
        raise RuntimeError("Recorded loaded episode id disagrees")
    console_command = pathlib.Path(record["console_source"]).with_name("command.json")
    run_root = pathlib.Path(record["console_source"].split("/shards/", 1)[0])
    manifest = json.loads((run_root / "manifest.json").read_text())
    result = json.loads((folder / "result.json").read_text())
    category = row.get("object_category")
    goals_by_category = task.get("goals_by_category", {})
    goal_key = pathlib.Path(row["scene_id"]).name + "_" + str(category)
    goals = row.get("goals") or goals_by_category.get(goal_key)
    if not goals:
        matching = [v for k, v in goals_by_category.items()
                    if k.endswith("_" + str(category))]
        if len(matching) == 1:
            goals = matching[0]
    identity = {"scene_id": row["scene_id"], "loaded_episode_id": record["episode_id"],
                "object_category": category, "start_position": row["start_position"],
                "start_rotation": row["start_rotation"], "goals": goals}
    if not goals or not category or identity["start_position"] is None:
        raise RuntimeError("Incomplete source task identity; do not guess")
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    output.append({
        "dataset": record["dataset"], "episode_key": record["episode_key"],
        "task_content_sha256": hashlib.sha256(raw).hexdigest(),
        "task_original_episode_id": str(row["episode_id"]),
        "task_object_category": category,
        "task_scene_id": row["scene_id"],
        "task_start_position": row["start_position"],
        "task_start_rotation": row["start_rotation"],
        "task_goal_count": len(goals),
        "task_identity_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "task_full_identity": identity,
        "task_row_sha256": hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "source_command": json.loads(console_command.read_text()),
        "source_run_manifest": manifest,
        "source_result": result,
        "source_trace_sha256": hashlib.sha256(pathlib.Path(record["trace_source"]).read_bytes()).hexdigest(),
        "task_index_verified": True,
    })
print(json.dumps(output, ensure_ascii=True))
'''.replace("PAYLOAD", repr(payload))
    completed = subprocess.run(
        ["ssh", host, "python3", "-"], input=remote_code,
        text=True, encoding="utf-8", capture_output=True, timeout=120, check=True,
    )
    identities = json.loads(completed.stdout)
    return {(row["dataset"], row["episode_key"]): row for row in identities}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path,
                        default=Path("outputs/frontier_backtrack_screen_20261009"))
    parser.add_argument("--output", type=Path,
                        default=Path("work/vlfm_frontier_probe_20261010/probe_episodes.json"))
    parser.add_argument("--ssh-host", default="yingu")
    parser.add_argument("--compact-output", type=Path,
                        default=Path("work/vlfm_frontier_probe_20261010/probe_episodes_compact.json"))
    parser.add_argument("--no-remote", action="store_true",
                        help="Selector test only: mark task identities unverified, never run these episodes")
    args = parser.parse_args()
    records = [row for dataset in DATASETS for row in select_dataset(dataset, args.source)]
    group_order = {name: index for index, name in enumerate((
        "failure_budget500_return", "failure_strict_nf_return",
        "failure_budget500_no_return_control", "success_return_control"))}
    records.sort(key=lambda row: (group_order[row["pilot_stratum"]], DATASETS.index(row["dataset"])))
    assert len(records) == 8
    remote = {} if args.no_remote else enrich_remote(records, args.ssh_host)
    for index, record in enumerate(records, 1):
        record["run_order"] = index
        if remote:
            record.update(remote[(record["dataset"], record["episode_key"])] )
        else:
            record["task_index_verified"] = False
    sources = {
        str(args.source / f"vlfm_{dataset}_{kind}.json"):
        sha256_file(args.source / f"vlfm_{dataset}_{kind}.json")
        for dataset in DATASETS for kind in ("terminal", "exploratory")
    }
    manifest = {
        "schema_version": 1,
        "created_date_local": "2026-10-10",
        "purpose": "Failure-first diagnostic/instrumentation pilot, not efficacy or population prevalence estimate",
        "selection_uses_future_intervention_results": False,
        "baseline_policy_changes_authorized_by_this_manifest": False,
        "frozen_return_screen": {"excursion_m_at_least": 3.0, "return_distance_m_at_most": 0.5,
                                  "action_gap_at_least": 20, "exclude_reported_stairs": True},
        "screen_limitation": "2D return only; not proof of low gain or causal failure. No-stairs is only a floor proxy.",
        "selection_sources_sha256": sources,
        "selector_sha256": sha256_file(Path(__file__)),
        "episode_count": len(records),
        "all_task_indices_verified": all(row["task_index_verified"] for row in records),
        "run_order_note": "Four return-associated failures first, then two non-return failure controls, then two returning success controls.",
        "episodes": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    compact = dict(manifest)
    compact["artifact_kind"] = "Compact diagnostic manifest; no source goals or raw task rows"
    compact["episodes"] = []
    for record in records:
        slim = {key: value for key, value in record.items()
                if key not in {"task_full_identity", "source_result", "source_run_manifest"}}
        source_manifest = record.get("source_run_manifest", {})
        slim["source_run_identity"] = {
            key: source_manifest.get(key)
            for key in ("dataset", "av_nav_commit", "vlfm_commit", "dataset_root", "scenes_dir", "manifest_sha256")
        }
        compact["episodes"].append(slim)
    args.compact_output.parent.mkdir(parents=True, exist_ok=True)
    args.compact_output.write_text(json.dumps(compact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for record in records:
        print(record["run_order"], record["dataset"], record["episode_key"],
              record["pilot_stratum"], record["first_witness_mode_counts"],
              "task_verified=" + str(record["task_index_verified"]))


if __name__ == "__main__":
    main()
