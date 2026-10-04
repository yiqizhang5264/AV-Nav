import unittest
import numpy as np
from av_nav.room_geometry import active_map_pose, ros_sensor_pose


class RoomGeometryTests(unittest.TestCase):
    def test_optical_forward_and_vertical_are_consistent(self):
        p, r = ros_sensor_pose([2., 1.88, -3.], np.eye(3), 1.)
        np.testing.assert_allclose(p, [2., 3., .88])
        np.testing.assert_allclose(r @ [0., 0., 1.], [0., 1., 0.])
        np.testing.assert_allclose(r @ [0., -1., 0.], [0., 0., 1.])
        np.testing.assert_allclose(r.T @ r, np.eye(3))
        self.assertAlmostEqual(np.linalg.det(r), 1.)

    def test_active_motion_heading_and_map_center(self):
        start = np.array([2., 1., 3.])
        np.testing.assert_allclose(active_map_pose(start, np.eye(3), start), [24., 24., 0.])
        np.testing.assert_allclose(active_map_pose(start + [0., 0., -1.], np.eye(3), start), [25., 24., 0.])
        yaw_left = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
        pose = active_map_pose(start + [-1., 0., 0.], yaw_left, start)
        np.testing.assert_allclose(pose, [24., 25., np.pi / 2])
