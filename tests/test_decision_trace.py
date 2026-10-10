"""CPU-only behavioral parity tests using methods extracted from real pinned VLFM.

Run with VLFM_ITM_POLICY_SOURCE=/absolute/path/to/vlfm/policy/itm_policy.py.
No Habitat, Torch, model server, or GPU is imported. The upstream file hash is
verified by the adapter, not replaced with a fake policy implementation.
"""

import ast
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import unittest
from typing import Any, Dict, List, Tuple, Union
from unittest import mock

import numpy as np

import decision_trace


SOURCE = Path(os.environ.get(
    "VLFM_ITM_POLICY_SOURCE",
    "/home/zyq/AV-Nav-worktrees/vlfm-original-5596d7a/external/vlfm/vlfm/policy/itm_policy.py",
))


def _load_real_policy_methods(include_reduce=False):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    original_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BaseITMPolicy")
    methods = [copy.deepcopy(n) for n in original_class.body
               if isinstance(n, ast.FunctionDef) and n.name in ("_get_best_frontier", "_explore")]
    if include_reduce:
        v3 = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ITMPolicyV3")
        methods.extend(copy.deepcopy(n) for n in v3.body
                       if isinstance(n, ast.FunctionDef) and n.name == "_reduce_values")
    test_class = ast.ClassDef(name="Policy", bases=[], keywords=[], body=methods, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[test_class], type_ignores=[]))

    def closest(points, point, threshold):
        distances = np.linalg.norm(points - point, axis=1)
        index = int(np.argmin(distances))
        return index if distances[index] <= threshold else -1

    namespace = dict(np=np, os=os, Any=Any, Dict=Dict, List=List, Tuple=Tuple,
                     Union=Union, Tensor=Any, TensorDict=dict,
                     closest_point_within_threshold=closest)
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace["Policy"]


class CountingEnforcer:
    def __init__(self, checks):
        self.answers = list(checks)
        self.checked = []
        self.added = []

    def check_cyclic(self, robot, frontier, top_two):
        index = len(self.checked)
        self.checked.append((robot.tolist(), frontier.tolist(), list(top_two)))
        return self.answers[index] if index < len(self.answers) else False

    def add_state_action(self, robot, frontier, top_two):
        self.added.append((robot.tolist(), frontier.tolist(), list(top_two)))


def _prepare(cls, inputs, sorted_pts, values, checks, last_frontier, last_value):
    policy = cls()
    policy._observations_cache = {"robot_xy": np.array([0.0, 0.0]),
                                  "frontier_sensor": np.asarray(inputs, dtype=float).reshape(-1, 2)}
    policy._last_frontier = np.asarray(last_frontier, dtype=float)
    policy._last_value = last_value
    policy._acyclic_enforcer = CountingEnforcer(checks)
    policy._stop_action = "STOP_SENTINEL"
    policy.sort_count = 0
    policy.pointnav_calls = []

    def sort(observations, frontiers):
        policy.sort_count += 1
        return np.asarray(sorted_pts, dtype=float), list(values)

    def pointnav(frontier, stop):
        policy.pointnav_calls.append((frontier.tolist(), stop))
        return "POINTNAV_ACTION"

    policy._sort_frontiers_by_value = sort
    policy._pointnav = pointnav
    return policy


class DecisionTraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not SOURCE.exists():
            raise RuntimeError("Set VLFM_ITM_POLICY_SOURCE to the pinned upstream source file")

    def _paired(self, inputs=None, sorted_pts=None, values=None, checks=(),
                last_frontier=(0.0, 0.0), last_value=float("-inf"), explore=False):
        inputs = [[1.0, 1.0], [2.0, 2.0]] if inputs is None else inputs
        sorted_pts = inputs if sorted_pts is None else sorted_pts
        values = [0.9, 0.7] if values is None else values
        baseline_class, traced_class = _load_real_policy_methods(), _load_real_policy_methods()
        baseline = _prepare(baseline_class, inputs, sorted_pts, values, checks, last_frontier, last_value)
        traced = _prepare(traced_class, inputs, sorted_pts, values, checks, last_frontier, last_value)
        records = []
        original_method = traced_class._get_best_frontier
        handle = decision_trace.install_decision_trace(
            traced_class, records.append, lambda policy, observations: {"episode_key": "fixture_1", "step": 7})
        with contextlib.redirect_stdout(io.StringIO()):
            for policy in (baseline, traced):
                if explore:
                    result = policy._explore({})
                else:
                    result = policy._get_best_frontier({}, policy._observations_cache["frontier_sensor"])
                snapshot = {
                    "result": result if explore else (result[0].tolist(), result[1]),
                    "last_frontier": policy._last_frontier.tolist(),
                    "last_value": policy._last_value,
                    "checked": policy._acyclic_enforcer.checked,
                    "added": policy._acyclic_enforcer.added,
                    "sort_count": policy.sort_count,
                    "pointnav": policy.pointnav_calls,
                    "debug_info": os.environ.get("DEBUG_INFO"),
                }
                if policy is baseline:
                    expected = snapshot
                else:
                    self.assertEqual(expected, snapshot)
        self.assertEqual(len(records), 1)
        json.dumps(records, allow_nan=False)
        self.assertFalse(hasattr(traced, decision_trace._PENDING))
        handle.uninstall()
        self.assertIs(traced_class._get_best_frontier, original_method)
        handle.uninstall()  # Idempotent.
        return records[0], expected

    def test_highest_noncyclic(self):
        record, state = self._paired()
        self.assertEqual(record["branch"], "highest_noncyclic")
        self.assertEqual(len(record["cyclic_checks"]), 1)
        self.assertEqual(state["sort_count"], 1)
        self.assertEqual(len(state["added"]), 1)

    def test_equal_scores_preserve_actual_sort_order(self):
        record, _ = self._paired(inputs=[[1, 0], [2, 0]], sorted_pts=[[2, 0], [1, 0]], values=[0.5, 0.5])
        self.assertEqual(record["selected_frontier_xy"], [2.0, 0.0])
        self.assertEqual(record["selected_input_indices"], [1])

    def test_keep_last_exact_skips_all_cyclic_checks(self):
        record, state = self._paired(last_frontier=[2, 2], last_value=0.7, checks=[True, True])
        self.assertEqual(record["branch"], "keep_last")
        self.assertEqual(record["keep_last"]["exact_or_near"], "exact")
        self.assertEqual(record["cyclic_checks"], [])
        self.assertEqual(state["checked"], [])

    def test_keep_last_near(self):
        record, _ = self._paired(last_frontier=[2, 2.3], last_value=0.7)
        self.assertEqual(record["branch"], "keep_last")
        self.assertEqual(record["keep_last"]["exact_or_near"], "within_0.5m")

    def test_last_value_drop_reselects(self):
        record, _ = self._paired(last_frontier=[2, 2], last_value=0.8)
        self.assertEqual(record["branch"], "highest_noncyclic")

    def test_suppressed_first_candidate(self):
        record, _ = self._paired(checks=[True, False])
        self.assertEqual([c["cyclic"] for c in record["cyclic_checks"]], [True, False])
        self.assertEqual(record["selected_sorted_index"], 1)

    def test_all_cyclic_keeps_original_fallback_even_when_index_mapping_is_odd(self):
        record, _ = self._paired(inputs=[[2, 0], [9, 0], [5, 0]],
                                 sorted_pts=[[9, 0], [5, 0], [2, 0]],
                                 values=[0.9, 0.5, 0.2], checks=[True, True, True])
        self.assertEqual(record["branch"], "all_cyclic_fallback")
        self.assertEqual(record["selected_sorted_index"], 1)
        self.assertEqual(record["selected_frontier_xy"], [5.0, 0.0])
        self.assertIn("max original-input", record["fallback_actual_rule"])

    def test_single_candidate(self):
        record, _ = self._paired(inputs=[[3, 4]], values=[0.7])
        self.assertEqual(record["top_two_values"], [0.7])

    def test_duplicate_coordinates_do_not_invent_unique_id(self):
        record, _ = self._paired(inputs=[[1, 1], [1, 1]], values=[0.8, 0.7])
        self.assertEqual(record["selected_input_indices"], [0, 1])

    def test_empty_frontier_stop(self):
        record, state = self._paired(inputs=[], values=[], explore=True)
        self.assertEqual(record["event"], "empty_frontier_stop")
        self.assertEqual(record["sentinel_or_empty"], "empty")
        self.assertEqual(state["sort_count"], 0)

    def test_zero_sentinel_frontier_stop(self):
        record, _ = self._paired(inputs=[[0, 0]], values=[], explore=True)
        self.assertEqual(record["sentinel_or_empty"], "zero_sentinel")

    def test_explore_still_calls_pointnav_once(self):
        record, state = self._paired(explore=True)
        self.assertEqual(len(state["pointnav"]), 1)
        self.assertEqual(record["event"], "frontier_decision")

    def test_wrong_source_hash_refuses_install_without_mutation(self):
        policy_class = _load_real_policy_methods()
        original = policy_class._get_best_frontier
        with mock.patch.object(decision_trace, "EXPECTED_SOURCE_SHA256", "0" * 64):
            with self.assertRaisesRegex(RuntimeError, "source hash mismatch"):
                decision_trace.install_decision_trace(policy_class, lambda record: None)
        self.assertIs(policy_class._get_best_frontier, original)

    def test_double_install_refused(self):
        policy_class = _load_real_policy_methods()
        handle = decision_trace.install_decision_trace(policy_class, lambda record: None)
        try:
            with self.assertRaisesRegex(RuntimeError, "duplicate instrumentation"):
                decision_trace.install_decision_trace(policy_class, lambda record: None)
        finally:
            handle.uninstall()

    def _paired_v3(self, channel_values, threshold):
        records = []
        baseline_class, traced_class = _load_real_policy_methods(True), _load_real_policy_methods(True)
        inputs = [[1, 1], [2, 2], [3, 3]]
        handle = decision_trace.install_decision_trace(traced_class, records.append,
                                                       reduction_class=traced_class)
        snapshots = []
        for cls in (baseline_class, traced_class):
            policy = _prepare(cls, inputs, inputs, [], [], [0, 0], float("-inf"))
            policy._exploration_thresh = threshold
            def actual_sort(observations, frontiers):
                policy.sort_count += 1
                reduced = policy._reduce_values(channel_values)
                indices = np.argsort([-v for v in reduced])
                return frontiers[indices], [reduced[i] for i in indices]
            policy._sort_frontiers_by_value = actual_sort
            with contextlib.redirect_stdout(io.StringIO()):
                result = policy._get_best_frontier({}, np.asarray(inputs, dtype=float))
            snapshots.append((result[0].tolist(), result[1], policy.sort_count,
                              policy._acyclic_enforcer.checked, policy._acyclic_enforcer.added))
        self.assertEqual(snapshots[0], snapshots[1])
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["v3_reduction"]["input_order_target_exploration_values"],
                         [list(pair) for pair in channel_values])
        handle.uninstall()
        return records[0]

    def test_v3_exploration_channel_applies_globally(self):
        record = self._paired_v3([(0.1, 0.8), (0.2, 0.3), (0.3, 0.5)], 0.5)
        self.assertEqual(record["v3_reduction"]["global_mode"], "exploration")
        self.assertEqual(record["selected_frontier_xy"], [1.0, 1.0])

    def test_v3_target_channel_at_exact_threshold_applies_globally(self):
        record = self._paired_v3([(0.1, 0.8), (0.2, 0.3), (0.5, 0.5)], 0.5)
        self.assertEqual(record["v3_reduction"]["global_mode"], "target")
        self.assertEqual(record["selected_frontier_xy"], [3.0, 3.0])

    def test_real_upstream_stateaction_membership_identity_behavior(self):
        # This is a diagnostic of the pinned original, deliberately NOT a fix:
        # the same numeric state/action is newly allocated by check_cyclic.
        path = SOURCE.parent / "utils" / "acyclic_enforcer.py"
        if not path.exists():
            path = Path(os.environ.get("VLFM_ACYCLIC_SOURCE", str(path)))
        if not path.exists():
            raise RuntimeError("Set VLFM_ACYCLIC_SOURCE for a standalone source fixture")
        namespace = {"np": np}
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)
        state_class = namespace["StateAction"]
        enforcer_class = namespace["AcyclicEnforcer"]
        position, action, other = np.array([1., 2.]), np.array([3., 4.]), (0.9, 0.7)
        first = state_class(position, action, other)
        second = state_class(position.copy(), action.copy(), other)
        self.assertEqual(hash(first), hash(second))
        self.assertNotEqual(first, second)
        self.assertIn(first, {first})
        self.assertNotIn(second, {first})
        enforcer = enforcer_class()
        enforcer.history = set()  # Isolate test from class-level shared history.
        enforcer.add_state_action(position, action, other)
        self.assertFalse(enforcer.check_cyclic(position.copy(), action.copy(), other))


if __name__ == "__main__":
    unittest.main(verbosity=2)
