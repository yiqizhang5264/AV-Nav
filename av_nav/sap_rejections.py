"""Episode-local exclusion of rejected candidates' observed XY footprints."""
import numpy as np
from scipy.spatial import cKDTree


class RejectedCandidates:
    def __init__(self, radius):
        self.radius = radius
        self._footprints = []

    def add(self, observations, goal, center):
        # A large or truncated candidate can extend beyond a center-radius disk.
        # Include every associated view and the selected navigation anchor.
        points = [np.asarray(o.points)[:, :2] for o in observations]
        points.extend([np.asarray(goal)[None, :2], np.asarray(center)[None, :2]])
        footprint = np.unique(np.concatenate(points), axis=0)
        self._footprints.append(cKDTree(footprint.copy()))
        return len(footprint)

    def filter_cloud(self, cloud):
        keep = np.ones(len(cloud), dtype=bool)
        for footprint in self._footprints:
            distances, _ = footprint.query(cloud[:, :2])
            keep &= distances > self.radius
        return cloud[keep]

    def rejects(self, observation):
        return any(tree.query(np.asarray(observation.center)[:2])[0] <= self.radius
                   for tree in self._footprints)
