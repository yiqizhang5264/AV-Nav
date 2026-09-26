"""Small runtime guards for deterministic failures in pinned STRIVE code."""

from __future__ import annotations

import pathlib
import tempfile
from typing import Any

import numpy as np


class _EpisodeAlreadyOver(RuntimeError):
    pass


def create_runtime_overlay(strive_root: pathlib.Path) -> tempfile.TemporaryDirectory[str]:
    """Load one patched STRIVE module without modifying the pinned submodule."""
    source_path = strive_root / "mapper_with_process_obs.py"
    source = source_path.read_text(encoding="utf-8")
    original = """            self.process_obs_pcd = gpu_merge_pointcloud(
                self.process_obs_pcd,
                self.current_navigable_pcd).voxel_down_sample(self.pcd_resolution)
"""
    replacement = """            self.process_obs_pcd = gpu_merge_pointcloud(
                self.process_obs_pcd,
                self.current_navigable_pcd)
"""
    if source.count(original) != 1:
        raise RuntimeError("pinned STRIVE process_obs merge site changed")
    overlay = tempfile.TemporaryDirectory(prefix="av-nav-strive-overlay-")
    pathlib.Path(overlay.name, source_path.name).write_text(
        source.replace(original, replacement), encoding="utf-8"
    )
    return overlay


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
    mapper_class: type[Any], max_interpolated_points: int = 120_000
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


def _voxel_reduce_arrays(
    positions: np.ndarray, colors: np.ndarray, voxel_size: float
) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(positions).all(axis=1)
    positions = positions[finite]
    colors = colors[finite]
    if len(positions) == 0:
        return positions, colors
    keys = np.floor(positions / voxel_size).astype(np.int64)
    _, indices = np.unique(keys, axis=0, return_index=True)
    indices.sort()
    return positions[indices], colors[indices]


def install_large_merge_memory_guard(
    mapper_module: Any, max_merged_points: int = 1, voxel_size: float = 0.05
) -> None:
    """Pre-voxelize large CUDA merges on CPU before Open3D's GPU hash table."""
    if getattr(mapper_module, "_av_nav_large_merge_memory_guard", False):
        return

    original = mapper_module.gpu_merge_pointcloud

    def guarded(pcd_a: Any, pcd_b: Any, merge_color: bool = True) -> Any:
        if pcd_a is None or pcd_a.is_empty() or pcd_b is None or pcd_b.is_empty():
            return original(pcd_a, pcd_b, merge_color)
        count_a = int(pcd_a.point.positions.shape[0])
        count_b = int(pcd_b.point.positions.shape[0])
        if count_a + count_b <= max_merged_points:
            return original(pcd_a, pcd_b, merge_color)

        positions_a = pcd_a.point.positions.cpu().numpy()
        positions_b = pcd_b.point.positions.cpu().numpy()
        colors_a = pcd_a.point.colors.cpu().numpy()
        colors_b = pcd_b.point.colors.cpu().numpy()
        if merge_color and len(colors_a):
            colors_b = np.repeat(colors_a[:1], len(positions_b), axis=0)
        positions, colors = _voxel_reduce_arrays(
            np.concatenate((positions_a, positions_b), axis=0),
            np.concatenate((colors_a, colors_b), axis=0),
            voxel_size,
        )
        result = mapper_module.o3d.t.geometry.PointCloud(pcd_a.device)
        result.point.positions = mapper_module.o3d.core.Tensor(
            positions,
            dtype=mapper_module.o3d.core.Dtype.Float32,
            device=pcd_a.device,
        )
        result.point.colors = mapper_module.o3d.core.Tensor(
            colors,
            dtype=mapper_module.o3d.core.Dtype.Float32,
            device=pcd_a.device,
        )
        return result

    mapper_module.gpu_merge_pointcloud = guarded
    mapper_module._av_nav_large_merge_memory_guard = True
