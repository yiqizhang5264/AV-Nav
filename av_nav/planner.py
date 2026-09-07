"""Bounded grid search on observed free space; no simulator navmesh/GT access."""
import heapq
import math
import numpy as np
from .geometry import adaptive_range, angle_delta


class Grid:
    def __init__(self, free, obstacles, pixels_per_meter, xy_to_px, px_to_xy):
        self.free = np.asarray(free, dtype=bool)
        self.obstacles = np.asarray(obstacles, dtype=bool)
        self.ppm = pixels_per_meter
        self.xy_to_px = xy_to_px
        self.px_to_xy = px_to_xy

    def cell(self, xy):
        x, y = self.xy_to_px(np.asarray(xy).reshape(1, 2))[0]
        return int(y), int(x)

    def inside(self, p):
        return 0 <= p[0] < self.free.shape[0] and 0 <= p[1] < self.free.shape[1]

    def distances(self, xy, budget):
        start = self.cell(xy)
        if not self.inside(start) or not self.free[start]:
            return {}, {}
        costs, parents, queue = {start: 0.0}, {}, [(0.0, start)]
        while queue:
            cost, p = heapq.heappop(queue)
            if cost != costs[p]:
                continue
            for dy, dx in ((0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)):
                q = p[0] + dy, p[1] + dx
                if not self.inside(q) or not self.free[q]:
                    continue
                if dx and dy and (not self.free[p[0]+dy,p[1]] or not self.free[p[0],p[1]+dx]):
                    continue
                nxt = cost + math.hypot(dx, dy) / self.ppm
                if nxt <= budget and nxt < costs.get(q, math.inf):
                    costs[q], parents[q] = nxt, p
                    heapq.heappush(queue, (nxt, q))
        return costs, parents

    def visible(self, xy, center, object_radius):
        a, b = np.array(self.cell(xy)), np.array(self.cell(center))
        n = int(np.max(np.abs(b-a))) + 1
        cells = np.rint(np.linspace(a, b, max(2,n))).astype(int)
        # Exclude target's own surface, which may be an obstacle in the map.
        trim = max(1, int((object_radius + .15) * self.ppm))
        for p in cells[:-trim]:
            if not self.inside(p) or self.obstacles[tuple(p)]:
                return False
        return True


def select_view(obs, history, robot_xy, grid, cfg, rng):
    center = obs.center[:2]
    costs, parents = grid.distances(robot_xy, cfg.max_path)
    desired = adaptive_range(obs, cfg)
    historical_angles = [math.atan2(*(p - center)[::-1]) for p in history]
    candidates = []
    for theta in np.arange(cfg.directions) * 2 * np.pi / cfg.directions:
        unit = np.array([math.cos(theta), math.sin(theta)])
        # Local visible-surface support, not a claim of complete object geometry.
        support = max(0.0, float(np.quantile((obs.points[:,:2]-center) @ unit, .9))) if cfg.surface_correction else 0.0
        for factor in (.8, 1.0, 1.2):
            surface = float(np.clip(desired * factor, cfg.min_range, cfg.max_range))
            xy = center + (support + surface) * unit
            cell = grid.cell(xy)
            cost = costs.get(cell)
            if cost is None or cost < cfg.independent_distance:
                continue
            if not grid.visible(xy, center, support):
                continue
            delta = min([abs(angle_delta(theta,a)) for a in historical_angles] or [np.pi/2])
            diversity = min(delta / (np.pi/2), 1.0)
            predicted_area = obs.area * (obs.depth / max(surface,.1))**2
            scale = math.exp(-abs(math.log(max(predicted_area,1e-6)/cfg.target_area)))
            utility = cfg.diversity_weight*diversity + cfg.scale_weight*scale - cfg.cost_weight*cost/cfg.max_path
            candidates.append(dict(xy=xy, cost=cost, diversity=diversity, scale=scale, utility=utility, cell=cell))
    if not candidates:
        return None, []
    if cfg.strategy == "random":
        chosen = candidates[int(rng.integers(len(candidates)))]
    elif cfg.strategy == "nearest":
        chosen = min(candidates, key=lambda v:v["cost"])
    else:
        chosen = max(candidates, key=lambda v:v["utility"])
    cells, p = [], chosen["cell"]
    while p in parents:
        cells.append(p)
        p = parents[p]
    cells.reverse()
    path = grid.px_to_xy(np.array([[p[1],p[0]] for p in cells]))
    chosen["path"] = path
    return chosen, candidates
