"""Read full runtime episode metadata without advancing/resetting an environment."""
from __future__ import annotations

import math


def episode_key(episode):
    return str(episode.scene_id), str(episode.episode_id)


def validate_full_episode(episode):
    category = getattr(episode, "object_category", None)
    if not isinstance(category, str) or not category or category == "unknown":
        raise ValueError("Incomplete episode metadata: missing object_category; request all_info=True")
    for field, size in (("start_position", 3), ("start_rotation", 4)):
        value = getattr(episode, field, None)
        if value is None or len(value) != size or not all(math.isfinite(float(v)) for v in value):
            raise ValueError(f"Incomplete episode metadata: invalid {field}; request all_info=True")
    if sum(float(v) ** 2 for v in episode.start_rotation) <= 0:
        raise ValueError("Invalid zero start_rotation quaternion")


class FullEpisodeCache:
    """One read-only RPC per episode change, never a dataset-inferred identity."""

    def __init__(self):
        self._cache = {}

    def resolve(self, envs, index, brief_episode):
        key = episode_key(brief_episode)
        cached = self._cache.get(index)
        if cached is None or cached[0] != key:
            full = envs.call_at(index, "current_episode", {"all_info": True})
            if episode_key(full) != key:
                raise ValueError("Full episode RPC disagrees with current scene/episode ID")
            validate_full_episode(full)
            self._cache[index] = (key, full)
        return self._cache[index][1]
