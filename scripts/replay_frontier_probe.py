"""CPU-only historical frontier-decision reconstruction, never a new rollout.

Uses the pinned upstream map implementations and AST-extracted, unmodified
selection methods. Cached RGB-D, sensors and ITM responses are the only inputs.
Historical target/explore mode is conditioned on, not independently inferred.
No detector, PointNav, Habitat simulator or model service is instantiated.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import sys
import time


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_method(tree, class_name, method_name):
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    return copy.deepcopy(next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name))


def load_replay_class(source_path, version, namespace):
    """Keep original source locations so the passive observer can hash/inspect it."""
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    methods = [source_method(tree, "BaseITMPolicy", name) for name in ("_get_best_frontier", "_explore")]
    methods.append(source_method(tree, version, "_sort_frontiers_by_value"))
    if version == "ITMPolicyV3":
        methods.append(source_method(tree, version, "_reduce_values"))
    cls = ast.ClassDef(name="ReplaySelection", bases=[], keywords=[], body=methods, decorator_list=[])
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["ReplaySelection"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vlfm-root", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--case-index", type=int)
    parser.add_argument("--episode-key")
    parser.add_argument("--limit-steps", type=int)
    parser.add_argument("--save-raw-maps", action="store_true")
    parser.add_argument("--save-audit-map", action="store_true",
                        help="Save final masks for retrospective local-connectivity checks, not policy input")
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in ("RECORD_VALUE_MAP", "PLAY_VALUE_MAP"):
        if os.environ.get(name, "0") != "0":
            raise RuntimeError(f"Refusing map recording/playback side effect: {name}")
    if os.environ.get("MAP_FUSION_TYPE"):
        raise RuntimeError("Unrecorded MAP_FUSION_TYPE override is not allowed")
    import cv2
    if args.cpu_threads < 1:
        raise ValueError("cpu-threads must be positive")
    cv2.setNumThreads(args.cpu_threads)
    import numpy as np
    import yaml
    sys.path.insert(0, str(Path(args.vlfm_root).resolve()))
    from vlfm.mapping.obstacle_map import ObstacleMap
    from vlfm.mapping.value_map import ValueMap
    from vlfm.utils.geometry_utils import xyz_yaw_to_tf_matrix, closest_point_within_threshold
    from depth_camera_filtering import filter_depth
    from decision_trace import install_decision_trace, EXPECTED_SOURCE_SHA256

    root = Path(args.vlfm_root).resolve()
    source = root / "vlfm/policy/itm_policy.py"
    if hashlib.sha256(source.read_text(encoding="utf-8").encode("utf-8")).hexdigest() != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("Pinned decision source changed")
    # Importing policy package would construct heavyweight optional dependencies.
    # The original acyclic classes need only NumPy and typing names.
    cyclic_path = root / "vlfm/policy/utils/acyclic_enforcer.py"
    cyclic_ns = {}
    exec(compile(cyclic_path.read_text(encoding="utf-8"), str(cyclic_path), "exec"), cyclic_ns)
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    indexed = list(enumerate(selection["episodes"]))
    if args.case_index is not None:
        indexed = [(i, row) for i, row in indexed if i == args.case_index]
    if args.episode_key:
        indexed = [(i, row) for i, row in indexed if row["episode_key"] == args.episode_key]
    if not indexed:
        raise ValueError("No selected episodes")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    summary = {"schema": "vlfm.cpu_frontier_replay.v1", "selection_sha256": sha256(args.selection),
               "script_sha256": sha256(__file__), "policy_source_sha256": sha256(source),
               "acyclic_source_sha256": sha256(cyclic_path), "no_gpu_inference": True,
               "no_new_navigation_rollout": True, "action_recomputed": False,
               "mode_conditioned_on_historical_trace": True, "cases": []}
    summary["map_sources_sha256"] = {str(p.relative_to(root)): sha256(p) for p in (
        root / "vlfm/mapping/obstacle_map.py", root / "vlfm/mapping/value_map.py",
        root / "vlfm/mapping/base_map.py", root / "vlfm/utils/geometry_utils.py")}

    for case_index, case in indexed:
        started = time.monotonic()
        case_out = output / f"{case_index:02d}_{case['dataset']}_{case['episode_key']}"
        case_out.mkdir()
        evidence = Path(case["trace_source"]).parent
        attempt = evidence.parents[2]
        config_path = attempt / "hydra/.hydra/config.yaml"
        config = yaml.safe_load(config_path.read_text())
        policy_cfg = config["habitat_baselines"]["rl"]["policy"]
        version = {"HabitatITMPolicyV2": "ITMPolicyV2", "HabitatITMPolicyV3": "ITMPolicyV3"}.get(policy_cfg["name"])
        if version is None:
            raise ValueError("Replay only audited for map-based ITMPolicyV2/V3")
        channels = len(policy_cfg["text_prompt"].split("|"))
        cameras = config["habitat"]["simulator"]["agents"]["main_agent"]["sim_sensors"]
        dc = cameras["depth_sensor"]
        fov = np.deg2rad(dc["hfov"])
        focal = dc["width"] / (2 * np.tan(fov / 2))
        height = cameras["rgb_sensor"]["position"][1]
        obstacle = ObstacleMap(policy_cfg["min_obstacle_height"], policy_cfg["max_obstacle_height"],
                               policy_cfg["agent_radius"], policy_cfg["obstacle_map_area_threshold"],
                               policy_cfg["hole_area_thresh"])
        value = ValueMap(channels, use_max_confidence=policy_cfg["use_max_confidence"],
                         obstacle_map=obstacle if policy_cfg.get("sync_explored_areas", False) else None)
        Replay = load_replay_class(source, version, {"np": np, "os": os,
                                   "closest_point_within_threshold": closest_point_within_threshold})
        policy = Replay()
        policy._last_frontier = np.zeros(2)
        policy._last_value = float("-inf")
        policy._value_map = value
        policy._exploration_thresh = policy_cfg["exploration_thresh"]
        policy._acyclic_enforcer = cyclic_ns["AcyclicEnforcer"]()
        policy._last_goal = np.zeros(2)
        policy._stop_action = None
        policy._pointnav = lambda goal, stop=False: np.array(goal, copy=True)
        if sha256(case["trace_source"]) != case["source_trace_sha256"]:
            raise ValueError("Historical trace changed since diagnostic selection")
        original_rows = [json.loads(line) for line in Path(case["trace_source"]).read_text().splitlines()]
        if [r["step"] for r in original_rows] != list(range(len(original_rows))):
            raise ValueError("Non-contiguous source step sequence")
        modes = case.get("original_mode_sequence")
        if modes is None or len(modes) != len(original_rows):
            raise ValueError("Exact console-derived original mode sequence is required")
        n_steps = len(original_rows) if args.limit_steps is None else min(args.limit_steps, len(original_rows))
        previous = np.zeros_like(obstacle.explored_area)
        ever = previous.copy()
        map_writer = None
        if args.save_raw_maps:
            from frontier_probe_recorder import MapDeltaWriter
            map_writer = MapDeltaWriter(case_out / "raw_maps")
        event_paths = sorted((evidence / "model_events").glob("*/event.json"))
        itm = []
        for path in event_paths:
            event = json.loads(path.read_text())
            if event["kind"] == "blip2_itm":
                itm.append((event["step"], event["call"], path, event))
        itm.sort(key=lambda item: (item[0], item[1]))
        if len({(s, c) for s, c, _, _ in itm}) != len(itm):
            raise ValueError("Duplicate cached ITM call identity")
        cursor = 0
        decisions = []
        current = {"step": None}
        def emit(record):
            decisions.append(record)
            with (case_out / "decisions.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
        options = {}
        if version == "ITMPolicyV3" and "reduction_class" in inspect.signature(install_decision_trace).parameters:
            options["reduction_class"] = Replay
        handle = install_decision_trace(Replay, emit, lambda p, obs: {
            "dataset": case["dataset"], "episode_key": case["episode_key"], "step": current["step"],
            "mode": modes[current["step"]], "historical_reconstruction": True}, **options)
        result = {"case_index": case_index, "episode_key": case["episode_key"], "dataset": case["dataset"],
                  "pilot_stratum": case["pilot_stratum"], "source_steps": len(original_rows), "replayed_steps": 0,
                  "source_steps_sha256": sha256(case["trace_source"]), "config_sha256": sha256(config_path),
                  "frontier_decisions": 0, "exact_frontier_goals": 0, "empty_frontier_stops": 0,
                  "mismatches": [], "policy_name": policy_cfg["name"], "itm_calls_consumed": 0,
                  "world_height_available": False, "validated_to_end": False,
                  "original_result": json.loads((evidence / "result.json").read_text())}
        try:
            with (case_out / "inputs.jsonl").open("w", encoding="utf-8") as input_file, \
                    (case_out / "steps.jsonl").open("w", encoding="utf-8") as step_file, \
                    (case_out / "selector_stdout.log").open("w", encoding="utf-8") as log:
                for step, row in enumerate(original_rows[:n_steps]):
                    current["step"] = step
                    paths = {name: evidence / folder / f"{step:04d}.{ext}" for name, folder, ext in (
                        ("sensors", "sensors", "npz"), ("depth", "depth", "npy"), ("rgb", "rgb", "png"))}
                    inputs = {name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()}
                    with np.load(paths["sensors"], allow_pickle=False) as sensors:
                        gps = sensors["gps"].copy()
                        yaw = float(sensors["compass"].item())
                    x, y = gps
                    robot_xy = np.array([x, -y])
                    tf = xyz_yaw_to_tf_matrix(np.array([x, -y, height]), yaw)
                    depth = np.load(paths["depth"], allow_pickle=False)
                    depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
                    obstacle.update_map(depth, tf, dc["min_depth"], dc["max_depth"], focal, focal, fov)
                    rgb = cv2.imread(str(paths["rgb"]))
                    if rgb is None:
                        raise ValueError("Missing/invalid raw RGB")
                    responses, consumed = [], []
                    expected_prompts = [p.replace("target_object", row["policy_info"]["target_object"].replace("|", "/"))
                                        for p in policy_cfg["text_prompt"].split("|")]
                    for channel in range(channels):
                        if cursor >= len(itm) or itm[cursor][0] != step:
                            raise ValueError(f"Missing or out-of-order ITM call at step {step}")
                        _, call, path, event = itm[cursor]
                        if event["request"]["text"] != expected_prompts[channel]:
                            raise ValueError(f"Cached ITM prompt mismatch at step {step}, channel {channel}")
                        event_image = path.parent / "image.png"
                        cached_rgb = cv2.imread(str(event_image))
                        if cached_rgb is None or not np.array_equal(rgb, cached_rgb):
                            raise ValueError(f"Cached ITM image differs from raw RGB at step {step}")
                        response = float(event["response"])
                        if not np.isfinite(response):
                            raise ValueError("Non-finite cached ITM response")
                        responses.append(response)
                        consumed.append({"path": str(path), "sha256": sha256(path), "call": call,
                                         "image_sha256": sha256(event_image), "request": event["request"], "response": response})
                        cursor += 1
                    if cursor < len(itm) and itm[cursor][0] == step:
                        raise ValueError(f"Unused ITM call at step {step}")
                    value.update_map(np.asarray(responses), depth, tf, dc["min_depth"], dc["max_depth"], fov)
                    inputs["itm"] = consumed
                    input_file.write(json.dumps({"step": step, "inputs": inputs}, allow_nan=False) + "\n")
                    policy._observations_cache = {"robot_xy": robot_xy, "frontier_sensor": obstacle.frontiers}
                    before = len(decisions)
                    goal = None
                    if modes[step] == "explore":
                        with contextlib.redirect_stdout(log):
                            goal = policy._explore({})
                    observed = decisions[before:]
                    if len(observed) > 1:
                        raise ValueError("More than one selection event in a step")
                    decision = observed[0] if observed else None
                    recorded_goal = np.asarray(row["policy_info"]["nav_goal"])
                    exact = error = None
                    if goal is not None:
                        exact = bool(np.array_equal(goal, recorded_goal))
                        error = float(np.linalg.norm(goal - recorded_goal))
                        result["frontier_decisions"] += 1
                        result["exact_frontier_goals"] += int(exact)
                        if not exact:
                            result["mismatches"].append({"step": step, "error_m": error, "replayed": goal.tolist(), "recorded": recorded_goal.tolist()})
                    elif decision is not None:
                        result["empty_frontier_stops"] += 1
                        if int(row["action"]) != 0:
                            result["mismatches"].append({"step": step, "reason": "empty_frontier_replay_but_original_nonstop"})
                    explored = obstacle.explored_area.astype(bool)
                    new_lifetime = explored & ~ever
                    ever |= explored
                    record = {"step": step, "mode": modes[step], "original_action": row["action"],
                              "robot_xy": robot_xy.tolist(), "yaw_rad": yaw, "n_frontiers": len(obstacle.frontiers),
                              "target_detected": row.get("policy_info", {}).get("target_detected"),
                              "recorded_nav_goal": recorded_goal.tolist(), "replayed_nav_goal": None if goal is None else goal.tolist(),
                              "nav_goal_exact": exact, "nav_goal_error_m": error,
                              "explored_cells": int(explored.sum()), "new_explored_cells": int((explored & ~previous).sum()),
                              "removed_explored_cells": int((previous & ~explored).sum()),
                              "cumulative_seen_cells": int(ever.sum()), "new_cumulative_seen_cells": int(new_lifetime.sum()),
                              "normalization_pixels_per_meter": float(obstacle.pixels_per_meter), "decision": decision}
                    if map_writer is not None:
                        filename, _ = map_writer.write(step, {"obstacle": obstacle._map, "navigable": obstacle._navigable_map,
                            "explored": explored, "ever_explored": ever, "value_channels": value._value_map,
                            "value_confidence": value._map, "frontiers_xy": obstacle.frontiers})
                        record["raw_map_file"] = "raw_maps/" + filename
                    previous = explored.copy()
                    step_file.write(json.dumps(record, allow_nan=False) + "\n")
                    result["replayed_steps"] += 1
                    if step % 50 == 0:
                        print("REPLAY", case_index, case["episode_key"], step, flush=True)
            if n_steps == len(original_rows) and cursor != len(itm):
                raise ValueError("Unused ITM events after complete replay")
            if args.save_audit_map:
                np.savez_compressed(case_out / "final_audit_map.npz",
                                    navigable=obstacle._navigable_map,
                                    ever_explored=ever, obstacle=obstacle._map,
                                    pixels_per_meter=np.asarray(obstacle.pixels_per_meter),
                                    episode_pixel_origin=obstacle._episode_pixel_origin)
                result["final_audit_map_sha256"] = sha256(case_out / "final_audit_map.npz")
                result["final_audit_map_role"] = "Retrospective exclusion of wall/unknown proximity; never a decision-time oracle"
            result["validated_to_end"] = True
        finally:
            handle.uninstall()
            result["itm_calls_consumed"] = cursor
            result["elapsed_seconds"] = time.monotonic() - started
            result["complete_episode_reconstruction"] = result["replayed_steps"] == len(original_rows)
            result["all_frontier_decisions_exact"] = result["validated_to_end"] and not result["mismatches"]
            (case_out / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))
            summary["cases"].append(result)
            (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
        print("DONE", case_index, result["exact_frontier_goals"], "/", result["frontier_decisions"], "exact", flush=True)


if __name__ == "__main__":
    main()
