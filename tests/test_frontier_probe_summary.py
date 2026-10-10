import unittest

from scripts.summarize_frontier_probe import interval, delayed_known_frontiers, verify_printed_scores


class SummaryUnitsTests(unittest.TestCase):
    def test_original_display_score_verified_at_its_recorded_precision(self):
        old = [dict(policy_info=dict(debug="debug: Best value: 10.63%"))]
        new = [dict(step=0, replayed_nav_goal=[1., 2.], decision=dict(selected_value=.10633738))]
        self.assertEqual(verify_printed_scores(old, new), 1)
        new[0]["decision"]["selected_value"] = .11633738
        with self.assertRaisesRegex(ValueError, "mismatch"):
            verify_printed_scores(old, new)

    def test_known_frontier_is_not_the_same_as_visited_robot_pose(self):
        rows = []
        for step, robot, goal in ((0, [0., 0.], [0., 1.]), (21, [8., 0.], [0., 4.])):
            rows.append(dict(step=step, robot_xy=robot, replayed_nav_goal=goal,
                             n_frontiers=2, decision=dict(input_frontiers_xy=[[0., 1.], [0., 4.]],
                             sorted_frontiers_xy=[goal, [9., 9.]], sorted_values=[.2, .1], selected_value=.2,
                             branch="highest_noncyclic", selected_sorted_index=0)))
        result = delayed_known_frontiers(rows)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["earliest_matched_step"], 0)

    def test_pre_action_interval_distance_and_area_units(self):
        rows = [dict(robot_xy=[x, 0.], new_cumulative_seen_cells=cells,
                     normalization_pixels_per_meter=20., mode="explore", decision=None)
                for x, cells in ((0., 1000), (.25, 4), (.50, 8))]
        result = interval(rows, 0, 2)
        self.assertEqual(result["actions"], 2)
        self.assertEqual(result["traveled_xy_m"], .5)
        self.assertAlmostEqual(result["new_processed_map_m2"], 12/400)
        self.assertEqual(result["mode_counts"], {"explore": 2})
        with self.assertRaises(ValueError):
            interval(rows, 0, 4)


if __name__ == "__main__":
    unittest.main()
