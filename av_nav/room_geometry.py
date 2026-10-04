"""Explicit coordinate adapters used only by the room comparison."""
import numpy as np

HABITAT_TO_ROS = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
OPTICAL_TO_HABITAT_CAMERA = np.diag([1., -1., -1.])


def ros_sensor_pose(position, rotation, floor_y):
    p = HABITAT_TO_ROS @ np.asarray(position)
    p[2] -= floor_y
    r = HABITAT_TO_ROS @ np.asarray(rotation) @ OPTICAL_TO_HABITAT_CAMERA
    return p, r


def active_map_pose(position, rotation, start_position, map_size_m=48.):
    delta = np.asarray(position) - np.asarray(start_position)
    forward = np.asarray(rotation) @ np.array([0., 0., -1.])
    return np.array([map_size_m / 2 - delta[2], map_size_m / 2 - delta[0],
                     np.arctan2(-forward[0], -forward[2])])
