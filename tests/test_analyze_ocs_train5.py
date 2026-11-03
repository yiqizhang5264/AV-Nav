import unittest
from scripts.analyze_ocs_train5 import event_metrics, movement_metrics


class AnalysisTests(unittest.TestCase):
    def test_final_action_without_next_observation_is_not_counted_as_stall(self):
        r = movement_metrics([[0, 0], [0, 0], [.25, 0]], [1, 1, 1])
        self.assertEqual(r["forward_with_following_observation"], 2)
        self.assertEqual(r["forward_below_1cm"], 1)
        self.assertEqual(r["preaction_planar_path_m"], .25)

    def test_rotations_are_not_distance_or_failed_forward(self):
        r = movement_metrics([[0, 0], [0, 0], [0, 0]], [2, 3, 0])
        self.assertEqual(r["forward_with_following_observation"], 0)
        self.assertEqual(r["preaction_planar_path_m"], 0)
        self.assertFalse(r["spatial_return_screen"]["spatial_return"])

    def test_bad_gps_fails(self):
        for xy in ([[0, 0]], [[0, 0], [float("nan"), 0]]):
            with self.assertRaises(ValueError):
                movement_metrics(xy, [1, 0])

    def test_reset_kinds_have_separate_denominators(self):
        events = [dict(kind="pointnav_reset_called", reason=r) for r in
                  ["episode_reset", "exploration_navigation", "target_navigation"]]
        events += [dict(kind="pointnav_call", stop_allowed=target, actual_resets_this_call=reset,
                        evidence_step=i) for i, (target, reset) in enumerate([(False, 1), (False, 0), (True, 1)])]
        result = event_metrics(events)
        self.assertEqual(result["actual_exploration_resets"], 1)
        self.assertEqual(result["exploration_resets_per_100_calls"], 50)
        self.assertEqual(result["first_target_navigation_step"], 2)
        self.assertEqual(result["episode_resets"], 1)

    def test_missing_reset_callback_rejected(self):
        with self.assertRaises(ValueError):
            event_metrics([dict(kind="pointnav_call", stop_allowed=False, actual_resets_this_call=1)])


if __name__ == "__main__":
    unittest.main()
