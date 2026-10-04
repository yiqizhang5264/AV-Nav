"""Strictly skip a deterministic prefix already completed by a prior attempt."""

from __future__ import annotations

import json
import os
import pathlib
from typing import Any


def skip_completed_initial_episodes(envs: Any, observations: Any) -> Any:
    plan_path = os.environ.get("VLFM_SKIP_EPISODES_FILE")
    if not plan_path:
        return observations
    plan = json.loads(pathlib.Path(plan_path).read_text(encoding="utf-8"))
    expected = {str(value) for value in plan["episode_ids"]}
    if envs.num_envs != 1:
        raise RuntimeError("episode resume requires exactly one environment")
    skipped = []
    while str(envs.current_episodes()[0].episode_id) in expected:
        episode_id = str(envs.current_episodes()[0].episode_id)
        if episode_id in skipped:
            raise RuntimeError("episode resume cycled before reaching an unfinished episode")
        skipped.append(episode_id)
        observations = envs.reset()
    if set(skipped) != expected:
        missing = sorted(expected.difference(skipped))
        raise RuntimeError(f"completed episodes are not the deterministic initial prefix; missing={missing}")
    print(f"Resumed after strictly skipping {len(skipped)} completed episodes: {skipped}")
    return observations
