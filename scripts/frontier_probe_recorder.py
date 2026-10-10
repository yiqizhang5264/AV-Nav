"""Passive, lossless frontier diagnostics. No scores or policy states are changed."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


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


class MapDeltaWriter:
    """Lossless per-step array deltas; exact comparison, never epsilon thresholded."""

    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=False)
        self.previous = {}

    def write(self, step, arrays):
        data, metadata = {}, {}
        for name, array in arrays.items():
            array = np.asarray(array)
            if array.dtype.kind not in "biuf":
                raise TypeError(f"unsupported diagnostic array: {name}, {array.dtype}")
            old = self.previous.get(name)
            full = old is None or old.shape != array.shape or old.dtype != array.dtype
            if full:
                data[name + "__full"] = array
                count = array.size
            else:
                # Compare bytes as well as values: preserve signed zero and NaN payloads.
                width = array.dtype.itemsize
                changed = np.any(array.ravel().view(np.uint8).reshape(-1, width) !=
                                 old.ravel().view(np.uint8).reshape(-1, width), axis=1)
                indices = np.flatnonzero(changed).astype(np.int32)
                data[name + "__indices"] = indices
                data[name + "__values"] = array.ravel()[indices]
                count = len(indices)
            metadata[name] = dict(shape=list(array.shape), dtype=str(array.dtype),
                                  full=full, changed_elements=int(count))
            self.previous[name] = array.copy()
        data["metadata_json"] = np.asarray(json.dumps(metadata))
        path = self.folder / f"{step:04d}.npz"
        if path.exists():
            raise FileExistsError(path)
        np.savez_compressed(path, **data)
        return str(path.name), metadata


class FrontierRecorder:
    def __init__(self):
        self.episode_dir = None
        self.maps = None
        self.ever_explored = None
        self.previous_explored = None
        self.last_map_step = None

    def context(self):
        from vlfm_evidence_recorder import get_evidence_recorder
        evidence = get_evidence_recorder()
        if evidence.episode_dir is None:
            raise RuntimeError("frontier probe called outside an evidence episode")
        if self.episode_dir != evidence.episode_dir:
            self.episode_dir = evidence.episode_dir
            self.maps = MapDeltaWriter(self.episode_dir / "frontier_raw_maps")
            self.ever_explored = self.previous_explored = None
            self.last_map_step = None
        return dict(episode_dir=str(self.episode_dir), step=int(evidence.step),
                    recorded_monotonic_ns=time.monotonic_ns())

    def append(self, filename, record):
        context = self.context()
        with (self.episode_dir / filename).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(plain(dict(context, **record)), allow_nan=False) + "\n")

    def decision(self, record):
        self.append("frontier_decisions.jsonl", record)

    def frame(self, policy, mode):
        context = self.context()
        step = context["step"]
        if step == self.last_map_step:
            raise RuntimeError("multiple map snapshots for the same decision step")
        self.last_map_step = step
        arrays, map_meta = {}, {}
        obstacle = getattr(policy, "_obstacle_map", None)
        if obstacle is not None:
            for name, attr in (("obstacle", "_map"), ("navigable", "_navigable_map"),
                               ("explored", "explored_area"), ("frontiers_xy", "frontiers")):
                arrays[name] = np.asarray(getattr(obstacle, attr))
            explored = arrays["explored"].astype(bool)
            if self.ever_explored is None:
                self.ever_explored = np.zeros_like(explored)
                self.previous_explored = np.zeros_like(explored)
            new = explored & ~self.ever_explored
            removed = self.previous_explored & ~explored
            self.ever_explored |= explored
            arrays["ever_explored"] = self.ever_explored
            ppm = float(obstacle.pixels_per_meter)
            map_meta.update(new_ever_explored_cells=int(new.sum()),
                            removed_explored_cells=int(removed.sum()),
                            new_ever_explored_m2=float(new.sum() / ppm ** 2),
                            pixels_per_meter=ppm,
                            episode_pixel_origin=plain(obstacle._episode_pixel_origin),
                            map_epoch="episode", coordinate_frame="episodic_xy",
                            explored_definition="policy processed mask, not raw visibility rays")
            self.previous_explored = explored.copy()
        value = getattr(policy, "_value_map", None)
        if value is not None:
            arrays["value_channels"] = value._value_map
            arrays["value_confidence"] = value._map
        relative_path, schema = self.maps.write(step, arrays)
        cache = policy._observations_cache
        self.append("frontier_frames.jsonl", dict(
            mode=mode, target_object=policy._target_object,
            robot_xy=plain(cache.get("robot_xy")),
            robot_heading_rad=plain(cache.get("robot_heading")),
            last_nav_goal=plain(getattr(policy, "_last_goal", None)),
            raw_map_file=f"frontier_raw_maps/{relative_path}", map_schema=schema,
            **map_meta))


_RECORDER = None


def get_probe():
    global _RECORDER
    if _RECORDER is None:
        _RECORDER = FrontierRecorder()
    return _RECORDER


def record_policy_frame(policy, mode):
    get_probe().frame(policy, mode)


def record_environment(envs):
    if envs.num_envs != 1:
        raise RuntimeError("frontier probe requires one environment")
    # Read-only worker call; never supplied to the navigation policy.
    get_probe().append("frontier_poses.jsonl", envs.call_at(0, "frontier_probe_state"))


def episode_identity(episode):
    return dict(scene_id=episode.scene_id, episode_id=str(episode.episode_id),
                object_category=episode.object_category,
                start_position=list(episode.start_position),
                start_rotation=list(episode.start_rotation))


def select_episode(envs, observations):
    """Select after Habitat loading, preserving original shard enumeration IDs."""
    request = json.loads(Path(os.environ["VLFM_FRONTIER_CASE"]).read_text(encoding="utf-8"))
    wanted = request["identity"]
    seen = set()
    while True:
        if envs.num_envs != 1:
            raise RuntimeError("single-key probe requires one environment")
        state = envs.call_at(0, "frontier_probe_state")
        actual = state["episode_identity"]
        key = (actual["scene_id"], str(actual["episode_id"]))
        if key in seen:
            raise RuntimeError("requested episode absent from loaded full shard")
        seen.add(key)
        if str(actual["episode_id"]) == str(wanted["episode_id"]) and actual["scene_id"].endswith(wanted["scene_id"]):
            for field in ("object_category", "start_position", "start_rotation"):
                if actual[field] != wanted[field]:
                    raise RuntimeError(f"episode identity mismatch: {field}")
            print("FRONTIER_PROBE_SELECTED", json.dumps(actual), flush=True)
            return observations
        observations = envs.reset()


def install_environment():
    from functools import partial
    from habitat import VectorEnv
    original = VectorEnv.__init__

    def initialize(self, *args, **kwargs):
        kwargs["make_env_fn"] = partial(_make_environment, kwargs["make_env_fn"])
        original(self, *args, **kwargs)

    VectorEnv.__init__ = initialize


def _make_environment(original, *args, **kwargs):
    import gym
    import quaternion

    class ProbeWrapper(gym.Wrapper):
        def frontier_probe_state(self):
            env = self.unwrapped.habitat_env
            state = env.sim.get_agent_state()
            camera = state.sensor_states["depth"]
            return dict(episode_identity=episode_identity(env.current_episode),
                        agent_position_xyz=plain(state.position),
                        agent_rotation_wxyz=plain(quaternion.as_float_array(state.rotation)),
                        camera_position_xyz=plain(camera.position),
                        camera_rotation_wxyz=plain(quaternion.as_float_array(camera.rotation)),
                        scope="audit_only_not_policy_input")

    return ProbeWrapper(original(*args, **kwargs))
