"""Observed RGB-D geometry for SAP-Nav equations 4--6; no simulator access."""
import math
import numpy as np


def depth_points(depth, transform, minimum, maximum, fx, fy):
    depth = np.asarray(depth)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError('Expected an HxW depth image')
    v, u = np.where(np.isfinite(depth) & (depth > 0) & (depth < .95))
    z = depth[v, u] * (maximum - minimum) + minimum
    h, w = depth.shape
    points = np.column_stack((z, -(u-w//2)*z/fx, -(v-h//2)*z/fy))
    return points @ transform[:3, :3].T + transform[:3, 3]


class HeightMap:
    def __init__(self, shape, xy_to_px):
        self.values = np.full(shape, -np.inf, dtype=np.float32)
        self.xy_to_px = xy_to_px

    def update(self, points, max_height=None):
        if max_height is not None:
            # A ceiling surface is not a solid column from floor to ceiling.
            # Keep the same below-camera obstacle band used for navigation.
            points = points[points[:, 2] <= max_height]
        pixels = self.xy_to_px(points[:, :2])
        x, y = pixels.T
        valid = (x >= 0) & (y >= 0) & (x < self.values.shape[1]) & (y < self.values.shape[0])
        np.maximum.at(self.values, (y[valid], x[valid]), points[valid, 2])

    def visibility(self, xy, footprint, camera_height, grid):
        """Unknown cells are conservatively occluding; target endpoint is excluded."""
        start = np.array(grid.cell(xy))
        visible = 0
        for point in footprint:
            end = np.array(grid.cell(point[:2]))
            n = int(np.max(np.abs(end-start)))
            if n < 1:
                continue
            fraction = np.arange(1, n) / n
            cells = np.rint(start + fraction[:, None]*(end-start)).astype(int)
            if any(not grid.inside(p) for p in cells):
                continue
            heights = self.values[cells[:, 0], cells[:, 1]]
            ray = camera_height + fraction*(point[2]-camera_height)
            # Explored free floor has no measured obstacle height. Unexplored
            # space must never become a transparent shortcut.
            known = np.isfinite(heights) | grid.free[cells[:, 0], cells[:, 1]]
            if np.all(known & (heights < ray)):
                visible += 1
        return visible / max(1, len(footprint))


def select_view(points, robot_xy, grid, heights, camera_height, vertical_fov,
                visited=(), max_path=20.0):
    """Paper rings/angles, reachable explored cells, vertical FOV, visibility."""
    pixels = grid.xy_to_px(points[:, :2])
    unique, inverse = np.unique(pixels, axis=0, return_inverse=True)
    top = np.full(len(unique), -np.inf)
    np.maximum.at(top, inverse, points[:, 2])
    footprint = np.column_stack((grid.px_to_xy(unique), top))
    center = np.median(footprint[:, :2], axis=0)
    minimum = max(0.0, (camera_height - float(top.max())) / math.tan(vertical_fov/2))
    costs, parents = grid.distances(robot_xy, max_path)
    candidates = []
    for radius in (.8, 1.2, 1.6, 2.0, 2.4):
        if radius < minimum:
            continue
        for theta in np.arange(24)*2*np.pi/24:
            xy = center + radius*np.array([np.cos(theta), np.sin(theta)])
            cell = grid.cell(xy)
            if cell not in costs or np.linalg.norm(xy-robot_xy) < .25:
                continue
            if any(np.linalg.norm(xy-p) < .25 for p in visited):
                continue
            score = heights.visibility(xy, footprint, camera_height, grid)
            if score > 0:
                candidates.append(dict(xy=xy, cell=cell, visibility=score, distance=costs[cell]))
    if not candidates:
        return None, []
    # Stable ring/angle order resolves ties; distance is not an extra objective.
    best = max(candidates, key=lambda item: item['visibility']).copy()
    cells, cell = [], best['cell']
    while cell in parents:
        cells.append(cell)
        cell = parents[cell]
    cells.reverse()
    best['path'] = grid.px_to_xy(np.array([[p[1], p[0]] for p in cells]))
    return best, candidates
