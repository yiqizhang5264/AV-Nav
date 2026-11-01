import importlib.util
from pathlib import Path
import unittest

import numpy as np


spec = importlib.util.spec_from_file_location("cpu_ocs_probe", Path(__file__).parents[1] / "scripts/cpu_ocs_probe.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class ProbeTests(unittest.TestCase):
    def test_requires_exact_curve_count(self):
        with self.assertRaises(ValueError):
            probe.contour_variants(np.array([[1., 0.]]), [])

    def test_variants_remain_on_original_curves(self):
        frontiers = np.array([[1., 0.], [4., 0.]])
        curves = [np.column_stack((np.arange(0.5, 1.51, 0.05), np.zeros(21))),
                  np.column_stack((np.arange(3.5, 4.51, 0.05), np.zeros(21)))]
        for name, points, ids in probe.contour_variants(frontiers, curves):
            self.assertEqual(len(points), len(ids))
            for point, identity in zip(points, ids):
                self.assertLessEqual(np.linalg.norm(point - frontiers[identity]), 0.2 + 1e-9)
                self.assertTrue(np.min(np.linalg.norm(curves[identity] - point, axis=1)) < 1e-8)

    def test_reset_predicate_is_strict(self):
        self.assertFalse(probe.goal_reset_condition([0.1, 0.0], [0., 0.]))
        self.assertTrue(probe.goal_reset_condition([0.10001, 0.0], [0., 0.]))

    def test_geometry_guard_preserves_original_when_no_shift_is_admissible(self):
        frontier = np.array([[1., 0.]])
        curve = np.array([[.8, 0.], [1., 0.], [1.2, 0.]])
        variants = probe.contour_variants(frontier, [curve], reachable=lambda a, b: False)
        for _, points, _ in variants:
            for point in points:
                np.testing.assert_array_equal(point, frontier[0])

    def test_local_reachability_rejects_walls_and_unknown_approach(self):
        class Obstacle:
            _navigable_map = np.ones((12, 12), dtype=bool)

            @staticmethod
            def _xy_to_px(points):
                return np.rint(points).astype(int)
        obstacle = Obstacle()
        explored = np.ones((12, 12), dtype=bool)
        self.assertTrue(probe.locally_reachable([4, 5], [6, 5], obstacle, explored))
        obstacle._navigable_map[:, 5] = False
        self.assertFalse(probe.locally_reachable([4, 5], [6, 5], obstacle, explored))
        obstacle._navigable_map[:, 5] = True
        explored[:, 5] = False
        self.assertFalse(probe.locally_reachable([4, 5], [6, 5], obstacle, explored))
        self.assertFalse(probe.locally_reachable([-1, 5], [2, 5], obstacle, explored))

    def test_duplicate_coordinate_keeps_ambiguous_boundary_ids(self):
        self.assertEqual(probe.current_boundary_id([1., 0.], [[1., 0.], [1., 0.]], [0, 1]), [0, 1])

    def test_clone_history_does_not_mutate_baseline_class_history(self):
        class Cyclic:
            history = {1, 2}
        class Policy:
            pass
        policy = Policy()
        policy._last_frontier = np.array([1., 2.])
        policy._last_goal = np.zeros(2)
        policy._observations_cache = {}
        policy._acyclic_enforcer = Cyclic()
        duplicate = probe.clone_policy(policy)
        duplicate._acyclic_enforcer.history.add(3)
        duplicate._last_frontier[:] = 0
        self.assertEqual(policy._acyclic_enforcer.history, {1, 2})
        np.testing.assert_array_equal(policy._last_frontier, [1., 2.])


if __name__ == "__main__":
    unittest.main()
