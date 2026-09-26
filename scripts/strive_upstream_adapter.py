"""Small runtime guards for deterministic failures in pinned STRIVE code."""

from __future__ import annotations

from typing import Any

import numpy as np


class _EpisodeAlreadyOver(RuntimeError):
    pass


def install_episode_over_guard(agent_class: type[Any]) -> None:
    """End a step cleanly if STRIVE steps after Habitat's 500-step limit."""
    if getattr(agent_class, "_av_nav_episode_over_guard", False):
        return

    original_step_mod = agent_class.step_mod

    def guarded_step_mod(self: Any, *args: Any, **kwargs: Any) -> Any:
        original_env_step = self.env.step

        def guarded_env_step(action: Any, **step_kwargs: Any) -> Any:
            if self.env.episode_over:
                raise _EpisodeAlreadyOver
            return original_env_step(action, **step_kwargs)

        self.env.step = guarded_env_step
        try:
            return original_step_mod(self, *args, **kwargs)
        except _EpisodeAlreadyOver:
            return False
        finally:
            self.env.step = original_env_step

    agent_class.step_mod = guarded_step_mod
    agent_class._av_nav_episode_over_guard = True


def install_interpolation_memory_guard(
    mapper_class: type[Any], max_interpolated_points: int = 2_000_000
) -> None:
    """Voxel-deduplicate pathological ray endpoints before 60x interpolation."""
    if getattr(mapper_class, "_av_nav_interpolation_memory_guard", False):
        return

    original = mapper_class.get_closest_disances_and_points

    def guarded(self: Any, *args: Any, **kwargs: Any) -> Any:
        distances, points = original(self, *args, **kwargs)
        ray_samples = 60
        if len(points) * ray_samples <= max_interpolated_points:
            return distances, points

        voxel_size = float(self.pcd_resolution)
        voxel_keys = np.floor(points / voxel_size).astype(np.int64)
        _, indices = np.unique(voxel_keys, axis=0, return_index=True)
        points = points[np.sort(indices)]
        if len(points) * ray_samples > max_interpolated_points:
            limit = max_interpolated_points // ray_samples
            sample_indices = np.linspace(0, len(points) - 1, limit, dtype=int)
            points = points[sample_indices]
        return distances, points

    mapper_class.get_closest_disances_and_points = guarded
    mapper_class._av_nav_interpolation_memory_guard = True
