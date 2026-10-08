import unittest
from types import SimpleNamespace
import numpy as np
from av_nav.sap_rejections import RejectedCandidates


class RejectionTests(unittest.TestCase):
    def test_rejected_extent_and_goal_outside_center_disk_do_not_reappear(self):
        center = np.array([1.7206743231074992, .07886159911848657])
        goal = np.array([2.4461107254048153, .2772574852521016])
        self.assertGreater(np.linalg.norm(goal-center), .75)
        points = np.array([[*center, .5], [2.8, .3, .7], [3.4, .3, .8]])
        cloud = np.array([[*goal, .5, 1], [2.8, .3, .7, 1], [3.4, .3, .8, 1],
                          [6., 2., .5, 2]])
        original = cloud.copy()
        rejected = RejectedCandidates(.75)
        rejected.add([SimpleNamespace(points=points)], goal, center)
        self.assertTrue(rejected.rejects(SimpleNamespace(center=np.r_[center, .5])))
        self.assertFalse(rejected.rejects(SimpleNamespace(center=np.array([6., 2., .5]))))
        for _ in range(3):  # Detector adds the same candidate again each frame.
            np.testing.assert_array_equal(rejected.filter_cloud(cloud), cloud[-1:])
        np.testing.assert_array_equal(cloud, original)

    def test_all_associated_views_are_excluded_but_gap_is_preserved(self):
        rejected = RejectedCandidates(.75)
        views = [SimpleNamespace(points=np.array([[0., 0., .5]])),
                 SimpleNamespace(points=np.array([[4., 0., .5]]))]
        rejected.add(views, [0., 0.], [0., 0.])
        cloud = np.array([[0., 0., .5, 1], [4., 0., .5, 2], [2., 0., .5, 3]])
        np.testing.assert_array_equal(rejected.filter_cloud(cloud), cloud[-1:])

    def test_stale_anchor_empty_cloud_and_episode_reset(self):
        rejected = RejectedCandidates(.75)
        rejected.add([], [0., 0.], [0., 0.])
        self.assertEqual(rejected.filter_cloud(np.empty((0, 4))).shape, (0, 4))
        cloud = np.array([[0., 0., .5, 1], [2., 0., .5, 2]])
        np.testing.assert_array_equal(rejected.filter_cloud(cloud), cloud[-1:])
        np.testing.assert_array_equal(RejectedCandidates(.75).filter_cloud(cloud), cloud)
