"""CPU-only paired analysis of completed original/anchor/OCS runs; no rollout.

Motion is measured between successive pre-action GPS samples. Its final action
has no following observation, so no displacement is invented for that action.
Planar recurrence is a screen, never a claim of low-information/invalid returns.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from screen_allfail_routes import trajectory_screen


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def movement_metrics(xy, actions):
    xy = np.asarray(xy, dtype=float)
    if xy.shape != (len(actions), 2) or not np.isfinite(xy).all():
        raise ValueError("GPS/action length or shape mismatch")
    moves = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    forward = np.asarray(actions[:-1]) == 1
    return {"preaction_planar_path_m": float(moves.sum()),
            "observed_transitions": len(moves),
            "forward_with_following_observation": int(forward.sum()),
            "forward_below_1cm": int(np.count_nonzero(moves[forward] < 0.01)),
            "action_counts": dict(Counter(str(action) for action in actions)),
            "spatial_return_screen": trajectory_screen(xy)}


def event_metrics(events):
    calls = [e for e in events if e["kind"] == "pointnav_call"]
    resets = [e for e in events if e["kind"] == "pointnav_reset_called"]
    exploration = [e for e in calls if not e["stop_allowed"]]
    target = [e for e in calls if e["stop_allowed"]]
    er = sum(e["actual_resets_this_call"] for e in exploration)
    tr = sum(e["actual_resets_this_call"] for e in target)
    reset_counts = Counter(e["reason"] for e in resets)
    if er != reset_counts["exploration_navigation"] or tr != reset_counts["target_navigation"]:
        raise ValueError("Reset callbacks differ from per-call counts")
    selections = [e for e in events if e["kind"] == "frontier_selection"]
    return {"exploration_calls": len(exploration), "target_calls": len(target),
            "actual_exploration_resets": er, "actual_target_resets": tr,
            "exploration_resets_per_100_calls": 100 * er / len(exploration) if exploration else None,
            "episode_resets": reset_counts["episode_reset"],
            "first_target_navigation_step": target[0]["evidence_step"] if target else None,
            "selection_reasons": dict(Counter(e["diagnostics"]["reason"] for e in selections)),
            "defer_steps": [e["evidence_step"] for e in selections
                            if e["diagnostics"]["reason"] == "no_distance_or_map_progress_defer"],
            "fallback_calls": sum(e["diagnostics"].get("fallback_kind") == "upstream_selection" for e in selections)}


def analyze_condition(run, mode):
    run = Path(run).resolve()
    manifest = read(run / "manifest.json")
    if manifest["mode"] != mode or sha(manifest["dataset_path"]) != manifest["dataset_sha256"]:
        raise ValueError("Mode or frozen data mismatch")
    if mode == "original":
        completion = read(run / "completion.reverified.json")
        if not completion["data_complete"] or not completion["physical_identities_match"]:
            raise ValueError("Historical original was not verified")
        for episode in completion["episodes"]:
            for file, digest in episode["evidence_hashes"].items():
                if sha(file) != digest:
                    raise ValueError("Historical verified evidence changed")
    else:
        completion = read(run / "completion.json")
        if not completion["complete"] or completion["returncode"] != 0:
            raise ValueError("Condition did not complete")
    events = lines(run / "ocs_events.jsonl")
    expected = {str(r["filtered_index"]): r for r in manifest["identities"]}
    result = []
    for file in sorted((run / "evidence/episodes").glob("*/result.json")):
        folder = file.parent
        meta, final = read(folder / "episode.json"), read(file)
        task = expected[str(meta["episode_id"])]
        if Path(meta["scene_id"]).name != Path(task["scene_id"]).name:
            raise ValueError("Loaded scene/ID mismatch")
        if mode != "original" and any(meta[key] != task[key] for key in
                                      ("object_category", "start_position", "start_rotation")):
            raise ValueError("Runtime physical identity mismatch")
        rows = lines(folder / "steps.jsonl")
        if [r["step"] for r in rows] != list(range(len(rows))) or len(rows) != final["steps_recorded"]:
            raise ValueError("Noncontiguous result")
        if not rows[-1]["done"] or any(r["done"] for r in rows[:-1]):
            raise ValueError("Missing or duplicate terminal")
        gps = []
        for step in range(len(rows)):
            with np.load(folder / "sensors" / f"{step:04d}.npz", allow_pickle=False) as sensor:
                gps.append(sensor["gps"].tolist())
                if step == len(rows) - 1:
                    terminal_frontiers = sensor["frontier_sensor"].tolist()
        episode_events = [e for e in events if e["episode_evidence_dir"] == str(folder)]
        if not episode_events:
            raise ValueError("Missing joined runtime events")
        targets = [r["step"] for r in rows if r["policy_info"].get("target_detected")]
        filtered = []
        for path in sorted((folder / "detections").glob("*_filtered_target/detections.json")):
            detection = read(path)
            if detection["boxes_xyxy"]:
                filtered.append(detection["step"])
        robot_xy = np.asarray(gps) * [1, -1]  # Habitat adapter: gps copy; y *= -1.
        record = {"mode": mode, "scene": Path(task["scene_id"]).name,
                  "source_episode_id": task["source_episode_id"], "loaded_id": meta["episode_id"],
                  "target": task["object_category"], "episode_dir": str(folder),
                  "steps": len(rows), "success": final["metrics"]["success"], "spl": final["metrics"]["spl"],
                  "terminal_action": rows[-1]["action"], "terminal_frontiers": terminal_frontiers,
                  "terminal_goal_distance_m": rows[-1]["info"]["distance_to_goal"],
                  "reported_stairs": any(r["info"].get("traveled_stairs") for r in rows),
                  "historical_failure_string_not_ground_truth": final["failure_cause"],
                  "first_filtered_detection_step": min(filtered) if filtered else None,
                  "first_usable_target_step": min(targets) if targets else None,
                  **movement_metrics(robot_xy, [r["action"] for r in rows]), **event_metrics(episode_events),
                  "robot_xy": robot_xy.tolist(),
                  "distance_to_goal": [r["info"]["distance_to_goal"] for r in rows],
                  "source_sha256": {str(p): sha(p) for p in [file, folder / "episode.json", folder / "steps.jsonl"]}}
        result.append(record)
    if len(result) != len(expected) or len({r["loaded_id"] for r in result}) != len(expected):
        raise ValueError("Missing or duplicate episode")
    return {"mode": mode, "run_dir": str(run), "manifest_sha256": sha(run / "manifest.json"),
            "events_sha256": sha(run / "ocs_events.jsonl"), "task_identities": manifest["identities"],
            "dataset_sha256": manifest["dataset_sha256"], "episodes": result,
            "successes": sum(int(r["success"]) for r in result),
            "SR_percent": 100 * np.mean([r["success"] for r in result]),
            "SPL_percent": 100 * np.mean([r["spl"] for r in result])}


def plot(result, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"original": "#747c84", "anchor_only": "#d79528", "ocs": "#158465"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    for row, (scene, title) in enumerate([("5biL7VEkByM", "Toilet: escape from near-stationary loop"),
                                         ("NEVASPhcrxR", "Chair: release a stalled goal")]):
        for condition in result["conditions"]:
            episode = next(r for r in condition["episodes"] if r["scene"].startswith(scene))
            xy = np.asarray(episode["robot_xy"])
            mode, color = condition["mode"], colors[condition["mode"]]
            axes[row, 0].plot(xy[:, 0], xy[:, 1], color=color, lw=1.7, alpha=.85, label=mode)
            axes[row, 0].scatter(*xy[-1], marker="*" if episode["success"] else "x", color=color, s=90, zorder=5)
            axes[row, 1].plot(episode["distance_to_goal"], color=color, lw=1.6, label=mode)
            if mode == "ocs":
                for step in episode["defer_steps"]:
                    axes[row, 1].axvline(step, color=color, lw=.6, alpha=.22)
                first = episode["first_usable_target_step"]
                axes[row, 1].scatter(first, episode["distance_to_goal"][first], color=color, s=40, zorder=5)
        axes[row, 0].scatter(0, 0, color="#23344a", marker="o", s=35, zorder=6)
        axes[row, 0].set(title=title, xlabel="Planner episodic x (m)", ylabel="Planner episodic y (m)")
        axes[row, 0].set_aspect("equal", adjustable="datalim")
        axes[row, 1].set(title="Distance to annotated goal", xlabel="Logged step (0-based)", ylabel="Geodesic distance (m)")
        for ax in axes[row]:
            ax.grid(alpha=.2)
            ax.legend(fontsize=9)
    fig.suptitle("Frozen train-5 paired rollout: recovered cases\nFaint green lines: OCS no-progress deferrals; dot: usable target evidence", fontsize=13)
    fig.savefig(destination / "recovered_cases.png", dpi=180)
    fig.savefig(destination / "recovered_cases.pdf")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original", required=True)
    p.add_argument("--anchor-only", required=True)
    p.add_argument("--ocs", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()
    dest = Path(args.output)
    if dest.exists():
        raise ValueError("Use a new output directory")
    conditions = [analyze_condition(getattr(args, mode), mode) for mode in ["original", "anchor_only", "ocs"]]
    if any(c["task_identities"] != conditions[0]["task_identities"] or
           c["dataset_sha256"] != conditions[0]["dataset_sha256"] for c in conditions[1:]):
        raise ValueError("Conditions use different tasks")
    result = {"schema": "ocs.train5.paired_analysis.v1", "analysis_source_sha256": sha(__file__),
              "step_indexing": "0-based", "no_new_rollout": True,
              "path_unit": "metres over successive pre-action GPS samples; final action has no endpoint",
              "forward_low_motion_diagnostic_threshold_m": 0.01,
              "recurrence_is_not_low_information": True, "conditions": conditions}
    dest.mkdir(parents=True)
    with (dest / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    if args.plot:
        plot(result, dest)
    for condition in conditions:
        print(condition["mode"], "SR", condition["SR_percent"], "SPL", condition["SPL_percent"])
        for r in sorted(condition["episodes"], key=lambda r: int(r["loaded_id"])):
            print(json.dumps({k: r[k] for k in ["target", "steps", "success", "preaction_planar_path_m",
                "actual_exploration_resets", "exploration_calls", "actual_target_resets", "target_calls",
                "forward_below_1cm", "defer_steps", "first_filtered_detection_step", "first_usable_target_step"]}))


if __name__ == "__main__":
    main()
