import unittest

import numpy as np

from scripts.screen_allfail_routes import trajectory_screen, xy_from_row


class RouteScreenTests(unittest.TestCase):
    def test_matrix_screen_matches_brute_force(self):
        rng = np.random.RandomState(417)
        for unused in range(12):
            xy = np.cumsum(rng.normal(size=(80, 2)) * .3, axis=0)
            expected = None
            for end in range(20, len(xy)):
                for start in range(end - 20 + 1):
                    if np.linalg.norm(xy[start] - xy[end]) <= .75 and np.linalg.norm(xy[start:end + 1] - xy[start], axis=1).max() >= 3.0:
                        expected = (start, end)
                        break
                if expected:
                    break
            actual = trajectory_screen(xy)
            self.assertEqual(actual["spatial_return"], expected is not None)
            if expected:
                self.assertEqual((actual["witness"]["old_step"], actual["witness"]["returned_step"]), expected)

    def test_return_does_not_require_frontier_identity(self):
        path = [[float(x), 0.] for x in np.linspace(0., 4., 21)]
        path += [[float(x), .1] for x in np.linspace(4., .5, 21)]
        self.assertTrue(trajectory_screen(path)["spatial_return"])

    def test_rotations_alone_are_not_long_returns(self):
        self.assertFalse(trajectory_screen([[0., 0.]] * 500)["spatial_return"])

    def test_straight_exploration_has_no_return(self):
        self.assertFalse(trajectory_screen([[i * .25, 0.] for i in range(60)])["spatial_return"])

    def test_invalid_gps_is_unknown_not_no_return(self):
        with self.assertRaises(ValueError):
            xy_from_row(dict(policy_info=dict(gps="[nan 0]")))
        self.assertEqual(xy_from_row(dict(policy_info=dict(gps="[-0.5  2.25]"))).tolist(), [-.5, 2.25])


if __name__ == "__main__":
    unittest.main()
