"""Wrapper contract tests do not construct Habitat, models, or CUDA tensors."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from av_nav.frontier_ocs_runtime import install_policy_hooks


def classes():
    class Low:
        def __init__(self):
            self.resets = 0
        def reset(self):
            self.resets += 1

    class Object:
        def __init__(self):
            self._pointnav_policy = Low()
            self._last_goal = np.zeros(2)
            self._num_steps = 0
            self._pointnav_stop_radius = .30
            self._observations_cache = {"robot_xy": np.array([2., 2.])}
            self._obstacle_map = SimpleNamespace(
                _navigable_map=np.ones((100, 100), bool), pixels_per_meter=10,
                _xy_to_px=lambda points: np.rint(points * 10).astype(int))
            self.skip_reset = False
        def _reset(self):
            self._pointnav_policy.reset()
            self._last_goal = np.zeros(2)
            self._num_steps = 0
        def _pointnav(self, goal, stop=False):
            if np.linalg.norm(goal - self._last_goal) > .1 and not self.skip_reset:
                self._pointnav_policy.reset()
            self._last_goal = goal
            return ("STOP" if stop and np.linalg.norm(goal - self._observations_cache["robot_xy"]) < .3
                    else "TARGET" if stop else "EXPLORE")

    class Policy(Object):
        def __init__(self):
            super().__init__()
            self._last_frontier = np.zeros(2)
            self._last_value = float("-inf")
            self.selection_calls = 0
        def _reset(self):
            super()._reset()
            self._last_frontier = np.zeros(2)
        def _sort_frontiers_by_value(self, observations, points):
            order = np.argsort(-points[:, 1], kind="stable")
            return points[order], points[order, 1]
        def _get_best_frontier(self, observations, frontiers):
            points, values = self._sort_frontiers_by_value(observations, frontiers)
            self._last_frontier = points[0]
            self._last_value = values[0]
            self.selection_calls += 1
            return points[0], values[0]

    return Object, Policy


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.Object, self.Policy = classes()
        self.events = []
        self.handle = None

    def tearDown(self):
        if self.handle is not None:
            self.handle.restore()

    def install(self, mode, factory=None):
        self.handle = install_policy_hooks(policy_class=self.Policy, object_class=self.Object,
            settings={"mode": mode}, emit=self.events.append, context=lambda: {"episode": "fake"},
            scheduler_factory=factory)

    def test_original_selection_and_native_action_unchanged(self):
        points = np.array([[3., 8.], [6., 4.]])
        baseline = self.Policy()
        expected_goal, expected_score = baseline._get_best_frontier({}, points)
        expected_action = baseline._pointnav(expected_goal)
        self.install("original")
        observed = self.Policy()
        goal, score = observed._get_best_frontier({}, points)
        action = observed._pointnav(goal)
        np.testing.assert_array_equal(goal, expected_goal)
        self.assertEqual(score, expected_score)
        self.assertEqual(action, expected_action)
        self.assertEqual(observed.selection_calls, baseline.selection_calls)
        self.assertEqual(observed._pointnav_policy.resets, baseline._pointnav_policy.resets)
        np.testing.assert_array_equal(points, [[3., 8.], [6., 4.]])

    def test_target_stop_path_never_calls_scheduler(self):
        class Scheduler:
            def select(self, *args):
                raise AssertionError("Target navigation must not select a frontier")
        self.install("ocs", Scheduler)
        policy = self.Policy()
        goal = np.array([2., 2.])
        self.assertEqual(policy._pointnav(goal, stop=True), "STOP")
        self.assertIs(policy._last_goal, goal)
        event = self.events[-1]
        self.assertTrue(event["stop_allowed"])
        self.assertEqual(event["actual_resets_this_call"], 1)

    def test_actual_reset_not_inferred_from_predicate(self):
        self.install("original")
        policy = self.Policy()
        policy.skip_reset = True
        policy._pointnav(np.array([5., 5.]))
        self.assertTrue(self.events[-1]["expected_reset_predicate"])
        self.assertEqual(self.events[-1]["actual_resets_this_call"], 0)
        self.assertFalse(any(row["kind"] == "pointnav_reset_called" for row in self.events))

    def test_episode_reset_recreates_scheduler_and_isolates_counts(self):
        created = []
        def factory():
            obj = object()
            created.append(obj)
            return obj
        self.install("ocs", factory)
        policy = self.Policy()
        policy._reset()
        first = policy._ocs_runtime_state.scheduler
        policy._pointnav(np.array([4., 4.]))
        self.assertEqual(policy._ocs_runtime_state.actual_resets, 2)
        policy._reset()
        self.assertIsNot(first, policy._ocs_runtime_state.scheduler)
        self.assertEqual(policy._ocs_runtime_state.actual_resets, 1)
        self.assertEqual(self.events[-2]["reason"], "episode_reset")

    def test_anchor_holds_then_releases_at_arrival(self):
        self.install("anchor_only")
        policy = self.Policy()
        first, _ = policy._get_best_frontier({}, np.array([[5., 5.]]))
        held, _ = policy._get_best_frontier({}, np.array([[5.2, 5.1]]))
        np.testing.assert_array_equal(first, held)
        policy._observations_cache["robot_xy"] = first.copy()
        released, _ = policy._get_best_frontier({}, np.array([[5.2, 5.1]]))
        np.testing.assert_array_equal(released, [5.2, 5.1])

    def test_anchor_releases_if_obstacle_added(self):
        self.install("anchor_only")
        policy = self.Policy()
        policy._get_best_frontier({}, np.array([[5., 5.]]))
        policy._obstacle_map._navigable_map[50, 50] = False
        released, _ = policy._get_best_frontier({}, np.array([[5.2, 5.1]]))
        np.testing.assert_array_equal(released, [5.2, 5.1])

    def test_hook_restore_restores_class_and_low_instance(self):
        original = self.Policy._get_best_frontier
        self.install("original")
        policy = self.Policy()
        policy._pointnav(np.array([4., 4.]))
        self.handle.restore()
        self.handle = None
        self.assertIs(self.Policy._get_best_frontier, original)
        count = len(self.events)
        policy._pointnav(np.array([5., 5.]))
        self.assertEqual(len(self.events), count)


class CompletionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "scripts/run_vlfm_ocs.py"
        spec = importlib.util.spec_from_file_location("ocs_runner_test", path)
        cls.runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.runner)

    def test_completion_rejects_right_count_wrong_task(self):
        expected = {"scene_id": "hm3d/train/scene.basis.glb", "object_category": "bed",
                    "start_position": [1, 2, 3], "start_rotation": [0, 0, 0, 1]}
        with tempfile.TemporaryDirectory() as td:
            episode = Path(td) / "evidence/episodes/one"
            episode.mkdir(parents=True)
            (episode / "result.json").write_text(json.dumps({"metrics": {"success": 1}, "steps_recorded": 10}))
            wrong = dict(expected, object_category="chair")
            (episode / "episode.json").write_text(json.dumps(wrong))
            self.assertEqual(self.runner.verify_completed_identities(td, [expected]), (1, False))
            actual = dict(expected, scene_id="/assets/" + expected["scene_id"], episode_id="0")
            (episode / "episode.json").write_text(json.dumps(actual))
            self.assertEqual(self.runner.verify_completed_identities(td, [expected]), (1, True))


class GPUGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "scripts/check_gpu_idle.py"
        spec = importlib.util.spec_from_file_location("ocs_idle_test", path)
        cls.guard = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.guard)

    def test_low_utilization_with_compute_process_is_busy(self):
        responses = ["0, GPU-0, 0, 22000, 49000\n", "GPU-0, 123, train.py, 21900\n"]
        with patch.object(self.guard.subprocess, "check_output", side_effect=responses):
            self.assertFalse(self.guard.query_idle("0")["idle"])

    def test_no_compute_but_large_memory_is_not_idle(self):
        with patch.object(self.guard.subprocess, "check_output", side_effect=["0, GPU-0, 0, 22000, 49000\n", ""]):
            self.assertFalse(self.guard.query_idle("0")["idle"])

    def test_empty_low_utilization_gpu_is_idle(self):
        with patch.object(self.guard.subprocess, "check_output", side_effect=["0, GPU-0, 0, 18, 49000\n", ""]):
            self.assertTrue(self.guard.query_idle("0")["idle"])


if __name__ == "__main__":
    unittest.main()
