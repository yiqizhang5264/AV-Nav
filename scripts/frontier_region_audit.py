"""Retrospective, CPU-only audit of spatial returns and repeated transport.

This is an operational *low-yield revisit* screen, not a room recognizer and
not a counterfactual estimate of episodes recoverable by changing the policy.
It never changes a frontier, score, action, or navigation result.

Input rows are the small per-step records from replay_frontier_probe.py.  The
optional final map is used only to veto Euclidean matches through walls; it is
never fed back to the historical policy.  A positive finding additionally
requires destination service, completed follow-up, and no observed target
purpose.  All other returns are retained with an explicit uncertainty reason.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class AuditConfig:
    return_radius_m: float = 0.75
    departure_radius_m: float = 3.0
    min_elapsed_actions: int = 20
    route_proximity_m: float = 0.35
    route_sample_spacing_m: float = 0.10
    min_repeated_fraction: float = 0.70
    max_whole_visit_gain_m2: float = 0.50
    destination_exit_radius_m: float = 3.0
    goal_arrival_radius_m: float = 0.75
    min_return_path_m: float = 2.0
    min_explore_fraction: float = 0.80
    max_local_connectivity_m: float = 1.50
    max_pose_jump_m: float = 1.0


def _truth(value):
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value) if value is not None else False


def _finite_xy(value):
    try:
        array = np.asarray(value, dtype=float)
        return array if array.shape == (2,) and np.isfinite(array).all() else None
    except (TypeError, ValueError):
        return None


def _goal(row):
    decision = row.get("decision") or {}
    return _finite_xy(decision.get("selected_frontier_xy", row.get("replayed_nav_goal")))


def _merge_intervals(intervals):
    """Intervals are pre-action pose indices [start, end), never step counts twice."""
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([int(start), int(end)])
    return merged


def _route_fraction(xy, turn, arrival, config):
    """Length-weighted sampled distance to *segments*, not sparse past poses."""
    if turn < 1 or arrival <= turn:
        return 0.0, 0.0
    old_a, old_b = xy[:turn], xy[1:turn + 1]
    old_d = old_b - old_a
    denom = np.maximum(np.sum(old_d * old_d, axis=1), 1e-20)
    length = repeated = 0.0
    for a, b in zip(xy[turn:arrival], xy[turn + 1:arrival + 1]):
        segment = float(np.linalg.norm(b - a))
        if segment <= 1e-12:
            continue
        count = max(1, int(math.ceil(segment / config.route_sample_spacing_m)))
        points = a + ((np.arange(count) + 0.5) / count)[:, None] * (b - a)
        offsets = points[:, None, :] - old_a[None, :, :]
        projection = np.clip(np.sum(offsets * old_d[None, :, :], axis=2) / denom, 0, 1)
        residual = offsets - projection[:, :, None] * old_d[None, :, :]
        minimum = np.min(np.sum(residual * residual, axis=2), axis=1)
        repeated += segment * float(np.mean(minimum <= config.route_proximity_m ** 2))
        length += segment
    return (repeated / length if length else 0.0), length


def _local_connection(a_xy, b_xy, context, config):
    """4-connected known-free geodesic guard; cannot cut diagonally through walls.

    The conservative Manhattan graph may reject some legitimate nearby pairs;
    those become uncertain, never negative/no-revisit evidence.
    """
    if context is None:
        return {"status": "unavailable", "reason": "final_known_free_map_not_supplied"}
    try:
        navigable = np.asarray(context["navigable"], dtype=bool)
        explored = np.asarray(context["ever_explored"], dtype=bool)
        obstacle = np.asarray(context.get("obstacle", np.zeros_like(navigable)), dtype=bool)
        origin = np.asarray(context["episode_pixel_origin"], dtype=float)
        ppm = float(np.asarray(context["pixels_per_meter"]).item())
        if navigable.ndim != 2 or explored.shape != navigable.shape or obstacle.shape != navigable.shape or origin.shape != (2,) or ppm <= 0:
            raise ValueError("map shape/origin/resolution")
        points = np.rint(np.asarray([a_xy, b_xy])[:, ::-1] * ppm).astype(int) + origin.astype(int)
        points[:, 0] = navigable.shape[0] - points[:, 0]
        # Upstream BaseMap returns (pixel_x, pixel_y), whereas NumPy masks
        # are indexed [row=pixel_y, column=pixel_x]; ObstacleMap swaps here too.
        points = points[:, ::-1]
        start, goal = [tuple(int(v) for v in point) for point in points]
        height, width = navigable.shape
        def free(point):
            r, c = point
            return 0 <= r < height and 0 <= c < width and navigable[r, c] and explored[r, c] and not obstacle[r, c]
        if not free(start) or not free(goal):
            return {"status": "unresolved", "reason": "matched_pose_not_final_known_free", "pixels": [list(start), list(goal)]}
        cap = int(math.floor(config.max_local_connectivity_m * ppm + 1e-9))
        pending, seen = deque([(start, 0)]), {start}
        while pending:
            point, distance = pending.popleft()
            if point == goal:
                return {"status": "connected", "grid_path_m": distance / ppm, "graph": "four_connected_known_free"}
            if distance >= cap:
                continue
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                neighbor = (point[0] + dr, point[1] + dc)
                if neighbor not in seen and free(neighbor):
                    seen.add(neighbor)
                    pending.append((neighbor, distance + 1))
        return {"status": "unresolved", "reason": "no_short_known_free_connection", "max_path_m": config.max_local_connectivity_m}
    except (KeyError, TypeError, ValueError, IndexError):
        return {"status": "unavailable", "reason": "invalid_final_map_context"}


def _semantic_evidence(rows, source_rows, start, end):
    target_steps, navigate_steps, missing_steps = [], [], []
    for i in range(start, end + 1):
        if rows[i].get("mode") == "navigate":
            navigate_steps.append(int(rows[i]["step"]))
        if source_rows is not None:
            source = source_rows[i]
            info, policy = source.get("info", {}), source.get("policy_info", {})
            detected = policy.get("target_detected", info.get("target_detected", source.get("target_detected")))
            if detected is None:
                missing_steps.append(int(rows[i]["step"]))
            if _truth(detected):
                target_steps.append(int(rows[i]["step"]))
    return {"target_detection_steps": target_steps, "target_navigation_steps": navigate_steps,
            "semantic_trace_available": source_rows is not None and not missing_steps,
            "missing_target_detection_field_steps": missing_steps,
            "interpretation": "detection/navigation is evidence of another purpose, not proof of a true target"}


def analyze_episode(rows, source_rows=None, config=None, map_context=None):
    """Return JSON-safe per-return evidence and conservative episode-level counts.

    Every invocation is a fresh episode; map resets or missing pose/step data
    invalidate that episode rather than joining across gaps.  ``source_rows``
    must align one-for-one with replay records if supplied.  ``map_context``
    accepts an np.load mapping with navigable/ever_explored/obstacle, scalar
    pixels_per_meter and episode_pixel_origin in upstream pixel coordinates.
    """
    cfg = AuditConfig(**config) if isinstance(config, dict) else (config or AuditConfig())
    result = {"schema": "vlfm.spatial_revisit_audit.v1", "configuration": asdict(cfg),
              "spatial_region_is_not_semantic_room": True, "causal_recoverability_estimated": False,
              "all_returns_are_not_invalid": True, "coordinate_frame": "episodic_xy",
              "gain_definition": "new lifetime union of policy processed explored masks, not GT visibility or semantic information",
              "events": [], "input_steps": len(rows), "status": "analyzed",
              "episode_category": "no_qualifying_spatial_return", "data_issues": []}
    if not rows:
        result.update(status="insufficient_evidence", episode_category="insufficient_evidence", data_issues=["empty_trace"])
        return result
    if source_rows is not None and (len(source_rows) != len(rows) or any(a.get("step") != b.get("step") for a, b in zip(rows, source_rows))):
        result["data_issues"].append("source_replay_step_join_mismatch")
    steps = [row.get("step") for row in rows]
    if steps != list(range(len(rows))):
        result["data_issues"].append("non_contiguous_or_reset_steps")
    points = [_finite_xy(row.get("robot_xy")) for row in rows]
    if any(point is None for point in points):
        result["data_issues"].append("missing_or_invalid_pose")
    if result["data_issues"]:
        result.update(status="insufficient_evidence", episode_category="insufficient_evidence")
        return result
    xy = np.asarray(points)
    moves = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    jumps = np.flatnonzero(moves > cfg.max_pose_jump_m + 1e-9).tolist()
    if jumps:
        result.update(status="insufficient_evidence", episode_category="insufficient_evidence", data_issues=["pose_jump_or_floor_change"], pose_jump_action_steps=jumps)
        return result
    gain = np.full(len(rows), np.nan)
    for i, row in enumerate(rows):
        try:
            ppm = float(row["normalization_pixels_per_meter"])
            cells = int(row["new_cumulative_seen_cells"])
            if ppm > 0 and cells >= 0:
                gain[i] = cells / ppm ** 2
        except (KeyError, TypeError, ValueError):
            pass
    arrival = cfg.min_elapsed_actions
    while arrival < len(rows):
        prior = np.flatnonzero(np.linalg.norm(xy[:arrival - cfg.min_elapsed_actions + 1] - xy[arrival], axis=1) <= cfg.return_radius_m)
        chosen = None
        for previous in prior[::-1]:
            radial = np.linalg.norm(xy[previous:arrival + 1] - xy[previous], axis=1)
            if float(radial.max()) >= cfg.departure_radius_m:
                chosen = int(previous)
                turn = chosen + int(np.argmax(radial))
                break
        if chosen is None:
            arrival += 1
            continue
        end = arrival
        while end + 1 < len(rows):
            end += 1
            if np.linalg.norm(xy[end] - xy[arrival]) >= cfg.destination_exit_radius_m:
                break
        left_region = bool(np.linalg.norm(xy[end] - xy[arrival]) >= cfg.destination_exit_radius_m)
        terminal_decision = rows[end].get("decision") or {}
        terminal_nf = bool(end == len(rows) - 1 and rows[end].get("mode") == "explore" and
                           terminal_decision.get("branch") == "no_frontiers_stop" and
                           rows[end].get("original_action") == 0)
        followup_complete = left_region or terminal_nf
        repeated_fraction, path_length = _route_fraction(xy, turn, arrival, cfg)
        whole_gain = float(np.sum(gain[turn + 1:end + 1])) if np.isfinite(gain[turn + 1:end + 1]).all() else None
        transport_gain = float(np.sum(gain[turn + 1:arrival + 1])) if np.isfinite(gain[turn + 1:arrival + 1]).all() else None
        service_gain = float(np.sum(gain[arrival + 1:end + 1])) if np.isfinite(gain[arrival + 1:end + 1]).all() else None
        service_steps = []
        for i in range(arrival, end + 1):
            goal = _goal(rows[i])
            if rows[i].get("mode") == "explore" and goal is not None and np.linalg.norm(xy[i] - goal) <= cfg.goal_arrival_radius_m:
                service_steps.append(int(rows[i]["step"]))
        destination_service = bool(service_steps or terminal_nf)
        semantics = _semantic_evidence(rows, source_rows, turn, end)
        modes = [row.get("mode") for row in rows[turn:end + 1]]
        explore_fraction = modes.count("explore") / len(modes)
        connectivity = _local_connection(xy[chosen], xy[arrival], map_context, cfg)
        decision = rows[turn].get("decision") or {}
        alternatives = int(rows[turn].get("n_frontiers", 0))
        old_goal, new_goal = _goal(rows[chosen]), _goal(rows[arrival])
        goal_difference = None if old_goal is None or new_goal is None else float(np.linalg.norm(old_goal - new_goal))
        reasons = []
        if not followup_complete:
            reasons.append("destination_followup_right_censored_at_episode_end")
        if not destination_service:
            reasons.append("transport_only_destination_service_not_observed")
        if connectivity["status"] != "connected":
            reasons.append("same_local_free_space_unverified")
        if not semantics["semantic_trace_available"]:
            reasons.append("semantic_observation_trace_missing")
        if whole_gain is None:
            reasons.append("map_gain_missing")
        if explore_fraction < cfg.min_explore_fraction:
            reasons.append("not_predominantly_frontier_exploration")
        if path_length < cfg.min_return_path_m:
            reasons.append("short_return_transport")
        has_other_purpose = bool(semantics["target_detection_steps"] or semantics["target_navigation_steps"])
        if has_other_purpose:
            category = "target_evidence_or_target_navigation_return"
        elif whole_gain is not None and whole_gain > cfg.max_whole_visit_gain_m2:
            category = "observed_area_gain_return"
        elif alternatives <= 1:
            category = "possibly_necessary_retreat_return"
        elif reasons:
            category = "insufficient_evidence_return"
        elif repeated_fraction < cfg.min_repeated_fraction:
            category = "return_without_high_route_repetition"
        else:
            category = "low_yield_repeated_return"
        event = {"event_index": len(result["events"]), "previous_region_step": chosen,
                 "return_transport_start_step": turn, "arrival_step": arrival, "followup_end_step": end,
                 "previous_region_xy": xy[chosen].tolist(), "arrival_xy": xy[arrival].tolist(),
                 "max_departure_m": float(np.linalg.norm(xy[chosen:arrival + 1] - xy[chosen], axis=1).max()),
                 "return_transport_actions": arrival - turn, "return_transport_path_m": path_length,
                 "repeated_route_fraction": repeated_fraction, "destination_followup_actions": end - arrival,
                 "return_plus_followup_actions": end - turn, "transport_new_area_m2": transport_gain,
                 "destination_followup_new_area_m2": service_gain, "whole_return_visit_new_area_m2": whole_gain,
                 "followup_complete": followup_complete, "followup_left_region": left_region,
                 "followup_terminal_empty_frontier": terminal_nf, "destination_service_observed": destination_service,
                 "goal_arrival_steps": service_steps, "exploration_mode_fraction": explore_fraction,
                 "local_connectivity": connectivity, "semantic_evidence": semantics,
                 "frontier_decision_branch_at_turn": decision.get("branch"),
                 "frontier_count_at_turn": alternatives,
                 "only_frontier_at_turn_may_be_necessary": alternatives <= 1,
                 "multiple_frontier_coordinates_not_reachability_proof": alternatives > 1,
                 "old_vs_return_selected_frontier_distance_m": goal_difference,
                 "same_frontier_identity_required": False, "category": category,
                 "uncertainty_reasons": reasons,
                 "not_a_causal_failure_assignment": True}
        # Descriptive threshold grid only.  It does not replace the frozen rule.
        event["sensitivity"] = [
            {"gain_cap_m2": cap, "repeated_fraction_min": repeat,
             "qualifies": bool(not reasons and alternatives > 1 and not has_other_purpose and whole_gain is not None and
                               whole_gain <= cap and repeated_fraction >= repeat)}
            for cap in (0.0, 0.1, 0.5, 1.0) for repeat in (0.5, 0.7, 0.9)]
        result["events"].append(event)
        # One arrival per continuous local visit; no count inflation while spinning.
        arrival = end + 1
    counts = {}
    for event in result["events"]:
        counts[event["category"]] = counts.get(event["category"], 0) + 1
    low = [event for event in result["events"] if event["category"] == "low_yield_repeated_return"]
    intervals = _merge_intervals([(event["return_transport_start_step"], event["followup_end_step"]) for event in low])
    all_intervals = _merge_intervals([(event["return_transport_start_step"], event["followup_end_step"]) for event in result["events"]])
    result.update(event_category_counts=counts, spatial_return_events=len(result["events"]),
                  low_yield_return_events=len(low), low_yield_union_action_intervals=intervals,
                  all_return_union_action_intervals=all_intervals,
                  low_yield_union_observed_actions=sum(end - start for start, end in intervals),
                  low_yield_union_path_m=float(sum(np.sum(moves[start:end]) for start, end in intervals)),
                  observed_transition_actions=max(0, len(rows) - 1),
                  observed_path_m=float(moves.sum()),
                  terminal_action_has_no_post_pose=True,
                  selection_reason_verified_elsewhere=True,
                  final_map_used_only_as_retrospective_wall_guard=map_context is not None)
    if low:
        result["episode_category"] = "contains_low_yield_repeated_return"
    elif result["events"]:
        result["episode_category"] = "spatial_returns_without_confirmed_low_yield_pattern"
    return result
