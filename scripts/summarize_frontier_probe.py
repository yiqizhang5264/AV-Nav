"""Descriptive evidence from matched historical decisions, not causal efficacy."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def interval(rows, start, end):
    """Pre-action pose at start to pre-action pose at end, end-start actions."""
    segment = rows[start:end + 1]
    if len(segment) != end - start + 1:
        raise ValueError("Incomplete interval")
    xy = np.asarray([row["robot_xy"] for row in segment])
    distance = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
    # End's map observes the pose reached by action end-1; exclude start's map.
    area = sum(row["new_cumulative_seen_cells"] / row["normalization_pixels_per_meter"] ** 2
               for row in segment[1:])
    branches = Counter(row["decision"]["branch"] for row in segment[:-1] if row["decision"] is not None)
    return dict(start_pre_action_step=start, end_pre_action_step=end,
                actions=end-start, traveled_xy_m=distance,
                new_processed_map_m2=float(area),
                new_processed_map_m2_per_traveled_m=float(area / distance) if distance > 0 else None,
                mode_counts=dict(Counter(row["mode"] for row in segment[:-1])),
                decision_branch_counts=dict(branches))


def delayed_known_frontiers(rows):
    """Spatial recurrence proxy, not a persistent frontier identity or causal label."""
    history = []
    previous_goal = None
    events = []
    for row in rows:
        decision = row["decision"]
        if decision is None or row["replayed_nav_goal"] is None:
            continue
        goal = np.asarray(row["replayed_nav_goal"])
        step = row["step"]
        robot = np.asarray(row["robot_xy"])
        changed = previous_goal is None or np.linalg.norm(goal - previous_goal) > .5
        matches = [(old_step, points[np.argmin(np.linalg.norm(points - goal, axis=1))].tolist())
                   for old_step, points in history if old_step <= step-20 and len(points)
                   and np.linalg.norm(points - goal, axis=1).min() <= .5]
        if changed and np.linalg.norm(goal-robot) >= 3. and matches:
            events.append(dict(step=step, goal_xy=goal.tolist(),
                               distance_from_robot_xy_m=float(np.linalg.norm(goal-robot)),
                               earliest_matched_step=matches[0][0],
                               earliest_matched_frontier_xy=matches[0][1],
                               branch=decision["branch"],
                               sorted_rank=decision["selected_sorted_index"]+1,
                               current_frontier_count=row["n_frontiers"]))
        history.append((step, np.asarray(decision["input_frontiers_xy"])))
        previous_goal = goal
    return events


def case_summary(case, result, rows):
    if not result["validated_to_end"] or not result["complete_episode_reconstruction"]:
        raise ValueError("Cannot summarize partial/unvalidated reconstruction")
    if result["mismatches"]:
        raise ValueError("Historical goals did not reproduce exactly")
    if [row["step"] for row in rows] != list(range(result["source_steps"])):
        raise ValueError("Missing/duplicate replay rows")
    decisions = [row["decision"] for row in rows if row["decision"] is not None]
    selected = [row for row in rows if row["replayed_nav_goal"] is not None]
    branches = Counter(d["branch"] for d in decisions)
    checks = [check for d in decisions for check in d.get("cyclic_checks", [])]
    lower_rank = [row for row in selected if row["decision"]["selected_sorted_index"] > 0]
    old_goal_selections = []
    xy = np.asarray([row["robot_xy"] for row in rows])
    for row in selected:
        step = row["step"]
        if step < 20:
            continue
        goal = np.asarray(row["replayed_nav_goal"])
        distance = float(np.linalg.norm(goal - xy[step]))
        past = float(np.linalg.norm(xy[:step-19] - goal, axis=1).min())
        if distance >= 3.0 and past <= 0.5:
            old_goal_selections.append(dict(step=step, current_to_goal_xy_m=distance,
                                            minimum_old_pose_to_goal_xy_m=past,
                                            branch=row["decision"]["branch"],
                                            rank=row["decision"]["selected_sorted_index"] + 1))
    output = dict(dataset=case["dataset"], episode_key=case["episode_key"],
                  stratum=case["pilot_stratum"], original_success=case["original_success"],
                  original_steps=len(rows), frontier_decisions=result["frontier_decisions"],
                  exact_frontier_decisions=result["exact_frontier_goals"],
                  empty_frontier_stops=result["empty_frontier_stops"],
                  decision_branch_counts=dict(branches), cyclic_checks=len(checks),
                  cyclic_hits=sum(bool(c["cyclic"]) for c in checks),
                  non_top_rank_selections=len(lower_rank),
                  delayed_known_frontier_switches=delayed_known_frontiers(rows),
                  delayed_known_frontier_note="A destination switch >0.5m; chosen point >=3m away and within0.5m of an available frontier >=20 actions earlier. Spatial recurrence only, not proven identity or low utility.",
                  old_pose_near_goal_selection_steps=old_goal_selections,
                  old_pose_near_goal_note="Per-decision count, not trip count. Goal >=3m away, <=0.5m from a pose >=20 actions old. No path-cost or low-gain inference.",
                  processed_map_area_removed_on_steps=sum(r["removed_explored_cells"] > 0 for r in rows),
                  total_seen_processed_map_m2=rows[-1]["cumulative_seen_cells"] / rows[-1]["normalization_pixels_per_meter"] ** 2,
                  first_return=None)
    witness = case["first_return_witness"]
    if witness:
        start, far, end = (witness[k] for k in ("leave_step", "furthest_step", "return_step"))
        output["first_return"] = dict(screen_witness=witness, full=interval(rows, start, end),
                                       outward=interval(rows, start, far), return_leg=interval(rows, far, end))
        if any(mode != "explore" for mode in output["first_return"]["full"]["mode_counts"]):
            output["first_return"]["pure_explore"] = False
        else:
            output["first_return"]["pure_explore"] = True
        output["first_return"]["decision_examples"] = [rows[step]["decision"] for step in (start, far, end)]
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", required=True)
    parser.add_argument("--replay-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.replay_dir)
    results = json.loads((root / "summary.json").read_text())
    selection = json.loads(Path(args.selection).read_text())
    cases = []
    for result in results["cases"]:
        index = result["case_index"]
        case = selection["episodes"][index]
        if (case["dataset"], case["episode_key"]) != (result["dataset"], result["episode_key"]):
            raise ValueError("Mismatched manifest identity")
        folder = root / f"{index:02d}_{case['dataset']}_{case['episode_key']}"
        cases.append(case_summary(case, result, read_jsonl(folder / "steps.jsonl")))
    report = dict(schema="vlfm.frontier_probe_analysis.v1", source_replay=str(root),
                  total_expected=len(selection["episodes"]), complete_cases=len(cases),
                  population_efficacy_claim=False, counterfactual_recovery_measured=False,
                  no_gpu_inference=True, cases=cases,
                  limitations=["Conditioned on historical mode, RGB-D and scores; no new action rollout.",
                               "Purposefully selected pilot; not overall failure prevalence.",
                               "Processed-map area is a proxy, not GT or true visible volume.",
                               "2D poses, stairs proxy only; no original per-step true height.",
                               "Returning can be useful. Positive or small gain alone does not prove failure causation."])
    path = Path(args.output)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(dict(complete_cases=len(cases), total_frontier_decisions=sum(c["frontier_decisions"] for c in cases),
                          cyclic_checks=sum(c["cyclic_checks"] for c in cases), cyclic_hits=sum(c["cyclic_hits"] for c in cases))))


if __name__ == "__main__":
    main()
