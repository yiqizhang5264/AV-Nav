import os
import pathlib
import sys
import tempfile
import unittest

import numpy as np


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from strive_upstream_adapter import (
    install_episode_over_guard,
    install_interpolation_memory_guard,
    _voxel_reduce_arrays,
    create_runtime_overlay,
)


class _Environment:
    def __init__(self):
        self.episode_over = False
        self.actions = []

    def step(self, action, **kwargs):
        self.actions.append(action)
        return action


class StriveUpstreamAdapterTests(unittest.TestCase):
    def test_valid_steps_are_unchanged(self):
        class Agent:
            def __init__(self):
                self.env = _Environment()

            def step_mod(self):
                return self.env.step(1)

        install_episode_over_guard(Agent)
        agent = Agent()
        self.assertEqual(agent.step_mod(), 1)
        self.assertEqual(agent.env.actions, [1])

    def test_post_limit_step_ends_cleanly(self):
        class Agent:
            def __init__(self):
                self.env = _Environment()

            def step_mod(self):
                self.env.episode_over = True
                self.env.step(1)
                return True

        install_episode_over_guard(Agent)
        agent = Agent()
        self.assertFalse(agent.step_mod())
        self.assertEqual(agent.env.actions, [])

    def test_large_interpolation_endpoints_are_voxel_deduplicated(self):
        class Mapper:
            pcd_resolution = 0.1

            def get_closest_disances_and_points(self):
                points = np.repeat(
                    np.array([[0.01, 0.01, 0.0], [0.21, 0.01, 0.0]]),
                    10,
                    axis=0,
                )
                return np.array([1.0]), points

        install_interpolation_memory_guard(Mapper, max_interpolated_points=180)
        distances, points = Mapper().get_closest_disances_and_points()
        self.assertEqual(distances.tolist(), [1.0])
        self.assertEqual(len(points), 2)

    def test_voxel_reduce_removes_nonfinite_and_duplicate_cells(self):
        positions = np.array(
            [[0.01, 0.01, 0.0], [0.02, 0.02, 0.0], [np.nan, 0.0, 0.0], [0.2, 0.0, 0.0]]
        )
        colors = np.arange(12).reshape(4, 3)
        reduced_positions, reduced_colors = _voxel_reduce_arrays(
            positions, colors, 0.1
        )
        self.assertEqual(reduced_positions.shape, (2, 3))
        self.assertEqual(reduced_colors.tolist(), [colors[0].tolist(), colors[3].tolist()])

    def test_runtime_overlay_removes_only_redundant_gpu_downsample(self):
        source = """prefix
            self.process_obs_pcd = gpu_merge_pointcloud(
                self.process_obs_pcd,
                self.current_navigable_pcd).voxel_down_sample(self.pcd_resolution)
suffix
"""
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "mapper_with_process_obs.py").write_text(source)
            overlay = create_runtime_overlay(root)
            patched = pathlib.Path(
                overlay.name, "mapper_with_process_obs.py"
            ).read_text()
            overlay.cleanup()
        self.assertNotIn("current_navigable_pcd).voxel_down_sample", patched)
        self.assertIn("current_navigable_pcd)\nsuffix", patched)


if __name__ == "__main__":
    unittest.main()
