"""CPU-only OCS mechanism probe on exactly reconstructed historical inputs.

The unmodified historical replay remains the acceptance oracle. Counterfactual
queries change only frontier representation; no PointNav action or SR is inferred.
No model endpoint, simulator or CUDA policy is instantiated.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def clone_policy(policy):
    """Clone decision state but share read-only maps; never append to class history."""
    result = copy.copy(policy)
    result.__dict__ = dict(policy.__dict__)
    for key in ("_last_frontier", "_last_goal"):
        if hasattr(policy, key):
            setattr(result, key, np.array(getattr(policy, key), copy=True))
    result._observations_cache = dict(policy._observations_cache)
    result._acyclic_enforcer = copy.copy(policy._acyclic_enforcer)
    result._acyclic_enforcer.history = copy.deepcopy(policy._acyclic_enforcer.history)
    result.__dict__.pop("_vlfm_frontier_probe_pending_20261010", None)
    return result


def goal_reset_condition(goal, previous_goal):
    """Exact numeric predicate from pinned _pointnav; NOT an executed reset."""
    goal, previous_goal = np.asarray(goal), np.asarray(previous_goal)
    return bool(not np.array_equal(goal, previous_goal)
                and np.linalg.norm(goal - previous_goal) > 0.1)


def contour_variants(frontiers, contours_xy, max_shift_m=0.2, reachable=None):
    """Resample the same contour, never translate a waypoint through obstacles.

    IDs denote the original current-step connected frontier curve, not persistent
    semantic rooms. Proposal splitting does not add any new map cells.
    """
    frontiers = np.asarray(frontiers)
    if len(frontiers) != len(contours_xy):
        raise ValueError("Each original waypoint needs its own original boundary")
    ids = np.arange(len(frontiers), dtype=int)
    variants = [("identity", frontiers.copy(), ids.copy()),
                ("reverse_order", frontiers[::-1].copy(), ids[::-1].copy()),
                ("duplicate_each", np.repeat(frontiers, 2, axis=0), np.repeat(ids, 2))]
    minus, plus = frontiers.copy(), frontiers.copy()
    for i, (goal, curve) in enumerate(zip(frontiers, contours_xy)):
        curve = np.asarray(curve).reshape(-1, 2)
        if len(curve) < 2:
            continue
        center = int(np.argmin(np.linalg.norm(curve - goal, axis=1)))
        distances = np.linalg.norm(curve - goal, axis=1)
        allowed = distances <= max_shift_m + 1e-9
        if reachable is not None:
            for j in np.flatnonzero(allowed):
                allowed[j] = reachable(goal, curve[j])
        left = np.flatnonzero((np.arange(len(curve)) <= center) & allowed)
        right = np.flatnonzero((np.arange(len(curve)) >= center) & allowed)
        if len(left):
            minus[i] = curve[left[np.argmax(distances[left])]]
        if len(right):
            plus[i] = curve[right[np.argmax(distances[right])]]
    variants += [("same_boundary_minus_0p20m", minus, ids.copy()),
                 ("same_boundary_plus_0p20m", plus, ids.copy()),
                 ("split_same_boundary_two_proposals", np.stack((minus, plus), axis=1).reshape(-1, 2),
                  np.repeat(ids, 2))]
    return variants


def locally_reachable(goal, candidate, obstacle, expanded_explored):
    """Require a short 4-connected path on the detector's known-side map.

    The 5x5 dilation is exactly the upstream frontier detector's boundary
    tolerance, not an oracle free-space map. This is a local eligibility guard,
    not a claim that PointNav can execute the route.
    """
    px = obstacle._xy_to_px(np.asarray([goal, candidate]))
    x0, y0 = (int(v) for v in px[0])
    x1, y1 = (int(v) for v in px[1])
    height, width = obstacle._navigable_map.shape
    if not (0 <= x0 < width and 0 <= x1 < width and 0 <= y0 < height and 0 <= y1 < height):
        return False
    lo_x, hi_x = max(0, min(x0, x1) - 2), min(width, max(x0, x1) + 3)
    lo_y, hi_y = max(0, min(y0, y1) - 2), min(height, max(y0, y1) + 3)
    allowed = (obstacle._navigable_map[lo_y:hi_y, lo_x:hi_x].astype(bool)
               & expanded_explored[lo_y:hi_y, lo_x:hi_x].astype(bool))
    start, end = (y0 - lo_y, x0 - lo_x), (y1 - lo_y, x1 - lo_x)
    if not allowed[start] or not allowed[end]:
        return False
    queue, seen = deque([start]), {start}
    while queue:
        y, x = queue.popleft()
        if (y, x) == end:
            return True
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            item = (y + dy, x + dx)
            if (0 <= item[0] < allowed.shape[0] and 0 <= item[1] < allowed.shape[1]
                    and allowed[item] and item not in seen):
                seen.add(item)
                queue.append(item)
    return False


def current_boundary_id(goal, points, ids):
    indices = np.flatnonzero(np.all(np.asarray(points) == np.asarray(goal), axis=1))
    return sorted(set(int(ids[i]) for i in indices))


def frozen_state_signature(policy, obstacle):
    h = hashlib.sha256()
    for array in (obstacle._map, obstacle._navigable_map, obstacle.explored_area,
                  policy._value_map._map, policy._value_map._value_map):
        h.update(np.ascontiguousarray(array).view(np.uint8))
    h.update(np.asarray(policy._last_frontier).tobytes())
    h.update(np.asarray(policy._last_goal).tobytes())
    h.update(repr(policy._last_value).encode())
    entries = []
    for entry in policy._acyclic_enforcer.history:
        data = {key: plain(value) for key, value in vars(entry).items()} if hasattr(entry, "__dict__") else plain(entry)
        entries.append(json.dumps(data, sort_keys=True, allow_nan=False))
    h.update(json.dumps(sorted(entries)).encode())
    return h.hexdigest()


class Probe:
    def __init__(self, args, replay):
        self.args, self.replay = args, replay
        self.output = Path(args.output_dir)
        self.output.mkdir(parents=True, exist_ok=False)
        self.records = (self.output / "frozen_probes.jsonl").open("w", encoding="utf-8")
        self.timeline = (self.output / "goal_reset_conditions.jsonl").open("w", encoding="utf-8")
        self.last_obstacle = None
        self.original_methods = {}
        self.states = {}
        self.counts = Counter()
        self.by_variant = {}
        self.by_episode = {}
        self.baseline_acceptance = []
        self.started = time.monotonic()
        manifest = json.loads(Path(args.selection).read_text(encoding="utf-8"))
        self.cases = {(c["dataset"], c["episode_key"]): c for c in manifest["episodes"]}
        self.enable_ocs = args.with_ocs

    def install(self):
        import decision_trace
        from vlfm.mapping.obstacle_map import ObstacleMap
        original_update = ObstacleMap.update_map
        original_load = self.replay.load_replay_class
        original_begin = decision_trace._Recorder.begin
        probe = self

        def update(obstacle, *args, **kwargs):
            result = original_update(obstacle, *args, **kwargs)
            probe.last_obstacle = obstacle
            return result

        def load(*args, **kwargs):
            cls = original_load(*args, **kwargs)
            probe.original_methods[cls] = cls._get_best_frontier
            return cls

        def begin(recorder, policy, observations, frontiers):
            context = recorder._context(policy, observations)
            probe.observe(policy, observations, frontiers, context)
            return original_begin(recorder, policy, observations, frontiers)

        ObstacleMap.update_map = update
        self.replay.load_replay_class = load
        decision_trace._Recorder.begin = begin

    def state(self, context):
        key = (context["dataset"], context["episode_key"])
        if key not in self.states:
            case = self.cases[key]
            rows = [json.loads(line) for line in Path(case["trace_source"]).read_text().splitlines()]
            state = {"rows": rows, "modes": case["original_mode_sequence"], "explore_count": 0,
                     "previous_ocs_goal": np.zeros(2), "previous_ocs_step": None}
            if self.enable_ocs:
                from av_nav.frontier_ocs import FrontierOCS
                state["ocs"] = FrontierOCS()
            self.states[key] = state
        return self.states[key]

    def observe(self, policy, observations, frontiers, context):
        import cv2
        from frontier_exploration.frontier_detection import detect_frontiers, frontier_waypoints

        step = int(context["step"])
        state = self.state(context)
        episode_key = context["dataset"] + "/" + context["episode_key"]
        episode_counts = self.by_episode.setdefault(episode_key, {"counts": Counter(), "variants": {}})
        rows = state["rows"]
        historical_goal = np.asarray(rows[step]["policy_info"]["nav_goal"])
        previous_goal = np.asarray(rows[step - 1]["policy_info"]["nav_goal"]) if step else np.zeros(2)
        previous_mode = state["modes"][step - 1] if step else "episode_start"
        distance = float(np.linalg.norm(historical_goal - previous_goal))
        reset = goal_reset_condition(historical_goal, previous_goal)
        same_near = bool(previous_mode == "explore" and 0.1 < distance <= 0.5)
        trigger = "periodic_20_explore" if state["explore_count"] % self.args.every_explore == 0 else None
        if same_near:
            trigger = "near_goal_change_gt_0p1m" if trigger is None else trigger + "+near_goal_change_gt_0p1m"
        state["explore_count"] += 1
        self.counts["explore_decisions"] += 1
        self.counts["goal_reset_condition_true"] += int(reset)
        self.counts["same_near_explore_goal_reset_condition_true"] += int(same_near)
        episode_counts["counts"]["explore_decisions"] += 1
        episode_counts["counts"]["goal_reset_condition_true"] += int(reset)
        episode_counts["counts"]["same_near_explore_goal_reset_condition_true"] += int(same_near)
        self.timeline.write(json.dumps(dict(context, previous_mode=previous_mode, previous_nav_goal=previous_goal.tolist(),
            current_historical_nav_goal=historical_goal.tolist(), goal_delta_m=distance,
            pointnav_reset_condition=reset, actual_pointnav_executed=False,
            near_goal_change_is_not_yet_same_opportunity=same_near), allow_nan=False) + "\n")
        obstacle = self.last_obstacle
        if obstacle is None:
            raise RuntimeError("No decision-time obstacle map captured")
        original = self.original_methods[type(policy)]
        robot_xy = np.asarray(policy._observations_cache["robot_xy"])
        ocs_before = copy.deepcopy(state.get("ocs")) if self.enable_ocs and trigger else None

        def score_fn(points):
            return policy._sort_frontiers_by_value(observations, points)

        previous_ocs_goal = np.array(state["previous_ocs_goal"], copy=True)
        ocs_consecutive = state["previous_ocs_step"] == step - 1
        if self.enable_ocs:
            ocs_real = state["ocs"].select(frontiers.copy(), robot_xy.copy(), obstacle, score_fn, step)
            state["previous_ocs_goal"] = np.array(ocs_real[0], copy=True)
            state["previous_ocs_step"] = step
        if trigger is None:
            return
        before = frozen_state_signature(policy, obstacle)
        expanded = cv2.dilate(obstacle.explored_area.astype(np.uint8), np.ones((5, 5), np.uint8))
        expanded[obstacle._navigable_map == 0] = 0
        curves = detect_frontiers(obstacle._navigable_map.astype(np.uint8), expanded, obstacle._area_thresh_in_pixels)
        exact_points = obstacle._px_to_xy(frontier_waypoints(curves))
        if not np.array_equal(exact_points, frontiers):
            raise RuntimeError("Frontier curves do not reproduce current original waypoint set")
        contours_xy = [obstacle._px_to_xy(curve.reshape(-1, 2)) for curve in curves]
        variants = contour_variants(frontiers, contours_xy,
                                    reachable=lambda a, b: locally_reachable(a, b, obstacle, expanded))
        records = []
        debug_before = os.environ.get("DEBUG_INFO")
        try:
            for name, points, ids in variants:
                clone = clone_policy(policy)
                capture = io.StringIO()
                with contextlib.redirect_stdout(capture):
                    goal, value = original(clone, observations, points.copy())
                record = dict(variant=name, proposal_count=len(points), goal=plain(goal), value=float(value),
                              boundary_ids=current_boundary_id(goal, points, ids),
                              pointnav_reset_condition=goal_reset_condition(goal, previous_goal),
                              input_changed=not np.array_equal(points, frontiers),
                              different_boundary_cells=False,
                              changed_coordinate_count=int(sum(not np.array_equal(point, frontiers[i])
                                                               for point, i in zip(points, ids))),
                              selector_stdout=capture.getvalue().strip())
                if self.enable_ocs:
                    alternative = copy.deepcopy(ocs_before)
                    alt_goal, alt_score, diagnostics = alternative.select(points.copy(), robot_xy.copy(), obstacle, score_fn, step)
                    record["ocs"] = dict(goal=plain(alt_goal), value=float(alt_score), diagnostics=plain(diagnostics),
                                         previous_shadow_goal=plain(previous_ocs_goal),
                                         consecutive_shadow_explore=ocs_consecutive,
                                         pointnav_reset_condition=(goal_reset_condition(alt_goal, previous_ocs_goal)
                                                                  if ocs_consecutive else None))
                records.append(record)
        finally:
            if debug_before is None:
                os.environ.pop("DEBUG_INFO", None)
            else:
                os.environ["DEBUG_INFO"] = debug_before
        if frozen_state_signature(policy, obstacle) != before:
            raise RuntimeError("Counterfactual probes mutated the live baseline/map state")
        identity = records[0]
        if not np.array_equal(identity["goal"], historical_goal):
            raise RuntimeError("Probe identity selector does not match the historical baseline goal")
        for record in records:
            record["selected_coordinate_changed"] = not np.array_equal(record["goal"], identity["goal"])
            record["selected_boundary_changed"] = record["boundary_ids"] != identity["boundary_ids"]
            record["extra_reset_condition_vs_identity"] = bool(record["pointnav_reset_condition"] and not identity["pointnav_reset_condition"])
            count = self.by_variant.setdefault(record["variant"], Counter())
            local_count = episode_counts["variants"].setdefault(record["variant"], Counter())
            count["n"] += 1
            local_count["n"] += 1
            for field in ("input_changed",):
                count[field] += int(record[field])
                local_count[field] += int(record[field])
            effective = record["input_changed"] or record["variant"] == "identity"
            count["effective_tests"] += int(effective)
            local_count["effective_tests"] += int(effective)
            for field in ("selected_coordinate_changed", "selected_boundary_changed", "extra_reset_condition_vs_identity"):
                count[field] += int(record[field])
                local_count[field] += int(record[field])
            if self.enable_ocs:
                record["ocs"]["coordinate_changed_vs_identity"] = not np.array_equal(record["ocs"]["goal"], identity["ocs"]["goal"])
                count["ocs_coordinate_changed"] += int(record["ocs"]["coordinate_changed_vs_identity"])
                local_count["ocs_coordinate_changed"] += int(record["ocs"]["coordinate_changed_vs_identity"])
        self.records.write(json.dumps(dict(context, trigger=trigger, robot_xy=plain(robot_xy),
            frozen_state_sha256=before, previous_nav_goal=plain(previous_goal),
            raw_frontiers=plain(frontiers), variants=records, observation_and_map_unchanged=True,
            actual_pointnav_executed=False, new_navigation_rollout=False), allow_nan=False) + "\n")
        self.records.flush()
        self.counts["frozen_states_probed"] += 1
        episode_counts["counts"]["frozen_states_probed"] += 1
        self.timeline.flush()
        if self.counts["frozen_states_probed"] % 20 == 0:
            self.write_summary("running")

    def write_summary(self, status, error=None):
        summary = dict(schema="vlfm.ocs_cpu_mechanism_probe.v1", status=status,
            elapsed_seconds=time.monotonic() - self.started, counts=dict(self.counts),
            variants={k: dict(v) for k, v in self.by_variant.items()}, error=error,
            episodes={key: {"counts": dict(val["counts"]),
                            "variants": {k: dict(v) for k, v in val["variants"].items()}}
                      for key, val in self.by_episode.items()},
            with_ocs=self.enable_ocs, no_gpu_inference=True, no_new_navigation_rollout=True,
            actual_pointnav_executed=False, reset_field_is_predicate_not_observed_execution=True,
            sampling_rule=f"every {self.args.every_explore} explore decisions plus every consecutive-explore goal movement in (0.1,0.5] m",
            source_sha256=digest(__file__), selection_sha256=digest(self.args.selection),
            upstream_replay_sha256=digest(self.replay.__file__),
            frontier_perturbation_radius_m=0.2,
            limit_steps=self.args.limit_steps,
            prefix_is_interface_smoke_only=self.args.limit_steps is not None,
            baseline_acceptance=self.baseline_acceptance,
            boundary_identity="Current-step connected frontier curve, not persistent task or semantic room",
            eligibility_guard="Changed points are locally 4-connected on navigable AND upstream-expanded explored map",
            ocs_interpretation="Shadow scheduler follows historical observations; no on-policy trajectory or action is evaluated",
            interpretation="Selection sensitivity only; no success, causal recoverability or action improvement claim")
        (self.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")

    def validate_replay(self, accepted):
        from summarize_frontier_probe import read_jsonl, verify_printed_scores
        if not accepted["cases"]:
            raise RuntimeError("Replay produced no cases")
        for result in accepted["cases"]:
            if not result["all_frontier_decisions_exact"]:
                raise RuntimeError("Historical baseline replay was not exact")
            if self.args.limit_steps is None and not result["complete_episode_reconstruction"]:
                raise RuntimeError("A full run did not reconstruct the complete source episode")
            case = self.cases[(result["dataset"], result["episode_key"])]
            folder = (self.output / "historical_replay"
                      / f"{result['case_index']:02d}_{result['dataset']}_{result['episode_key']}")
            reconstructed = read_jsonl(folder / "steps.jsonl")
            original = read_jsonl(case["trace_source"])[:result["replayed_steps"]]
            score_count = verify_printed_scores(original, reconstructed)
            if score_count != result["frontier_decisions"]:
                raise RuntimeError("Printed score and goal verification denominators disagree")
            self.baseline_acceptance.append({
                "dataset": result["dataset"], "episode_key": result["episode_key"],
                "source_steps": result["source_steps"], "replayed_steps": result["replayed_steps"],
                "frontier_decisions": result["frontier_decisions"],
                "exact_frontier_goals": result["exact_frontier_goals"],
                "original_printed_scores_matched": score_count,
                "score_precision": "Original console's 2 decimal percent, not full float identity",
                "complete_episode_reconstruction": result["complete_episode_reconstruction"],
                "source_steps_sha256": result["source_steps_sha256"],
                "config_sha256": result["config_sha256"],
                "mismatches": result["mismatches"],
            })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vlfm-root", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--case-index", type=int)
    parser.add_argument("--episode-key")
    parser.add_argument("--limit-steps", type=int)
    parser.add_argument("--every-explore", type=int, default=20)
    parser.add_argument("--with-ocs", action="store_true")
    args = parser.parse_args()
    if args.every_explore < 1:
        raise ValueError("every-explore must be positive")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(args.vlfm_root).resolve()))
    import replay_frontier_probe as replay
    probe = Probe(args, replay)
    probe.install()
    argv = [str(Path(replay.__file__)), "--vlfm-root", args.vlfm_root, "--selection", args.selection,
            "--output-dir", str(Path(args.output_dir) / "historical_replay"), "--cpu-threads", "1"]
    for name in ("case_index", "episode_key", "limit_steps"):
        value = getattr(args, name)
        if value is not None:
            argv.extend(["--" + name.replace("_", "-"), str(value)])
    sys.argv = argv
    try:
        replay.main()
        accepted = json.loads((Path(args.output_dir) / "historical_replay/summary.json").read_text())
        probe.validate_replay(accepted)
        probe.write_summary("complete_prefix" if args.limit_steps else "complete")
    except Exception as exc:
        probe.write_summary("failed", f"{type(exc).__name__}: {exc}")
        raise
    finally:
        probe.records.close()
        probe.timeline.close()


if __name__ == "__main__":
    main()
