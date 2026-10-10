import unittest

from scripts.summarize_frontier_probe import interval


class SummaryUnitsTests(unittest.TestCase):
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
