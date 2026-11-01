import copy
import unittest

import numpy as np

from av_nav.frontier_ocs import FrontierOCS, OCSConfig, Opportunity, short_known_path


class Map:
    def __init__(self):
        self.pixels_per_meter = 10
        self.explored_area = np.zeros((100, 100), bool)
        self.explored_area[10:71, 10:71] = True
        self._navigable_map = np.ones((100, 100), bool)
        self.curves = [np.array([[x, 72] for x in range(20, 41)]),
                       np.array([[72, y] for y in range(20, 41)])]

    def _xy_to_px(self, xy):
        return np.rint(xy * 10).astype(int)

    def _px_to_xy(self, px):
        return px / 10


def scores(points):
    values = points[:, 1]
    order = np.argsort(-values, kind="stable")
    return points[order], values[order]


class OCS(unittest.TestCase):
    def setUp(self):
        self.map = Map()
        self.ocs = FrontierOCS(contour_extractor=lambda m: m.curves)
        self.points = np.array([[3., 7.2], [7.2, 3.]])
        self.robot = np.array([4., 4.])

    def test_duplicate_permutation_and_resampling_invariant(self):
        snapshots = (self.points, self.points[::-1], np.repeat(self.points, 2, axis=0),
                     self.points + np.array([[.15, 0], [0, -.15]]))
        outputs = [copy.deepcopy(self.ocs).select(p, self.robot, self.map, scores, 0) for p in snapshots]
        self.assertTrue(all(np.array_equal(o[0], outputs[0][0]) for o in outputs))
        self.assertTrue(all(o[2]["opportunities"] == 2 for o in outputs))

    def test_active_anchor_stays_but_blocked_anchor_releases(self):
        first = self.ocs.select(self.points, self.robot, self.map, scores, 0)
        second = self.ocs.select(self.points + [[.15, 0], [0, 0]], self.robot, self.map, scores, 1)
        np.testing.assert_array_equal(first[0], second[0])
        self.map._navigable_map[70:75, 20:41] = False
        third = self.ocs.select(self.points, self.robot, self.map, scores, 2)
        self.assertFalse(np.array_equal(first[0], third[0]))

    def test_same_room_different_corner_retained(self):
        result = self.ocs.select(self.points, self.robot, self.map, scores, 0)
        self.assertEqual(result[2]["opportunities"], 2)
        self.assertNotEqual(result[2]["task_records"][0]["task_id"], result[2]["task_records"][1]["task_id"])

    def test_arrival_attempt_does_not_delete_all_candidates(self):
        first = self.ocs.select(self.points, self.robot, self.map, scores, 0)
        second = self.ocs.select(self.points, first[0], self.map, scores, 1)
        self.assertFalse(np.array_equal(first[0], second[0]))
        third = self.ocs.select(self.points, second[0], self.map, scores, 2)
        self.assertEqual(len(third[0]), 2)
        self.assertTrue(third[2]["all_deferred_retry"])

    def test_unknown_geometry_falls_back_not_nf(self):
        self.map.curves = []
        result = self.ocs.select(self.points, self.robot, self.map, scores, 0)
        np.testing.assert_array_equal(result[0], self.points[0])
        self.assertEqual(result[2]["fallback_kind"], "score_only")

    def test_unknown_geometry_calls_full_baseline_not_highest_score(self):
        self.map.curves = []
        calls = []
        def native_selector():
            calls.append("native_keep_last_or_cyclic_selection")
            return self.points[1].copy(), 3.
        def forbidden_score_only(points):
            raise AssertionError("Native fallback must not be replaced by score sorting")
        result = self.ocs.select(self.points, self.robot, self.map, forbidden_score_only, 0,
                                 baseline_select=native_selector)
        np.testing.assert_array_equal(result[0], self.points[1])
        self.assertEqual(result[1], 3.)
        self.assertEqual(calls, ["native_keep_last_or_cyclic_selection"])
        self.assertEqual(result[2]["fallback_kind"], "upstream_selection")
        self.assertIsNone(self.ocs.active_id)

    def test_available_geometry_does_not_call_baseline_fallback(self):
        def forbidden_baseline():
            raise AssertionError("Native fallback should only run without opportunities")
        result = self.ocs.select(self.points, self.robot, self.map, scores, 0,
                                 baseline_select=forbidden_baseline)
        self.assertEqual(result[2]["opportunities"], 2)
        self.assertNotIn("fallback_kind", result[2])

    def test_upstream_maps_are_unchanged(self):
        a, b = self.map.explored_area.copy(), self.map._navigable_map.copy()
        self.ocs.select(self.points, self.robot, self.map, scores, 0)
        np.testing.assert_array_equal(a, self.map.explored_area)
        np.testing.assert_array_equal(b, self.map._navigable_map)

    def test_new_unknown_corner_not_inherited_as_complete(self):
        first = self.ocs.select(self.points, self.robot, self.map, scores, 0)
        self.ocs.select(self.points, first[0], self.map, scores, 1)
        self.map.curves.append(np.array([[x, 72] for x in range(50, 61)]))
        out = self.ocs.select(np.vstack([self.points, [5.5, 7.2]]), self.robot, self.map, scores, 2)
        self.assertEqual(out[2]["opportunities"], 3)
        new_task = next(t for t in out[2]["task_records"] if t["anchor_xy"][0] == 5.5)
        self.assertEqual(new_task["attempted_support_cells"], 0)

    def test_wall_prevents_short_known_path(self):
        mask = np.ones((10, 10), bool)
        mask[:, 5] = False
        self.assertFalse(short_known_path(mask, (4, 4), (6, 4), 10))

    def test_disconnected_known_region_not_selected(self):
        self.map._navigable_map[:, 50] = False
        result = self.ocs.select(self.points, self.robot, self.map, scores, 0)
        self.assertEqual(result[2]["opportunities"], 1)
        self.assertLess(result[0][0], 5.)

    def test_anchor_still_free_but_no_longer_sees_support_updates(self):
        old = Opportunity(1, np.array([3., 7.2]), {(1, 1), (2, 1)})
        self.ocs.tasks = {1: old}
        self.ocs.next_uid = 2
        self.ocs._support = lambda obstacle, px, known, nav: set()
        tasks = self.ocs._associate([(np.array([3.5, 7.2]), {(1, 1), (2, 1)})],
                                    self.map, np.ones((100, 100), bool), 1)
        self.assertEqual(tasks[0].uid, 1)
        np.testing.assert_array_equal(tasks[0].anchor, [3.5, 7.2])

    def test_split_inherits_only_intersecting_attempt_cells(self):
        parent = Opportunity(1, np.array([3., 7.2]), {(1, 1), (2, 1), (3, 1)})
        self.ocs._attempt(parent)
        child = Opportunity(2, np.array([3.2, 7.2]), {(2, 1), (4, 1)})
        counts = self.ocs._attempt_counts(child, self.map, np.ones((100, 100), bool))
        self.assertEqual(sorted(counts), [0, 1])

    def test_merge_inherits_attempts_without_completing_new_cells(self):
        self.ocs._attempt(Opportunity(1, np.array([3., 7.2]), {(1, 1)}))
        self.ocs._attempt(Opportunity(2, np.array([3.5, 7.2]), {(2, 1)}))
        merged = Opportunity(3, np.array([3.2, 7.2]), {(1, 1), (2, 1), (3, 1)})
        counts = self.ocs._attempt_counts(merged, self.map, np.ones((100, 100), bool))
        self.assertEqual(sorted(counts), [0, 1, 1])

    def test_same_unknown_cells_from_blocked_approach_do_not_inherit_attempt(self):
        self.ocs._attempt(Opportunity(1, np.array([4.8, 4.]), {(20, 20)}))
        opposite = Opportunity(2, np.array([5.2, 4.]), {(20, 20)})
        known = np.ones((100, 100), bool)
        known[:, 50] = False
        self.assertEqual(self.ocs._attempt_counts(opposite, self.map, known), [0])

    def test_unchanged_geometry_does_not_accumulate_task_aliases(self):
        for step in range(100):
            result = self.ocs.select(self.points, self.robot, self.map, scores, step)
        self.assertEqual(len(self.ocs.tasks), 2)
        self.assertEqual(result[2]["opportunities"], 2)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            OCSConfig(support_overlap_min=1.1)
        with self.assertRaises(ValueError):
            self.ocs.select([], self.robot, self.map, scores, 0)
        with self.assertRaises(ValueError):
            self.ocs.select(self.points, [float("nan"), 0], self.map, scores, 0)
        with self.assertRaises(ValueError):
            OCSConfig(known_side_padding_px=1.5)


if __name__ == "__main__":
    unittest.main()
