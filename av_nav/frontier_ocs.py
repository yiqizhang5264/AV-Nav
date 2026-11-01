"""Diagnostic prototype: map-supported opportunity identity and stable anchors.

No detector, model, ground truth, simulator or torch dependency. Upstream maps
are read-only. Defaults are pre-result engineering settings, NOT calibrated
parameters. Coverage below means the union of the upstream processed explored
mask, not verified object visibility or absence evidence.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Callable, Dict, Optional, Set, Tuple

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class OCSConfig:
    support_radius_m: float = 1.0
    history_match_radius_m: float = 1.0
    support_overlap_min: float = 0.35
    support_bin_m: float = 0.25
    arrival_radius_m: float = 0.30
    stall_explore_calls: int = 20
    progress_distance_m: float = 0.25
    progress_area_m2: float = 0.05
    proposal_snap_radius_m: float = 0.35
    known_side_padding_px: int = 2

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.support_overlap_min > 1:
            raise ValueError("support_overlap_min must be in (0,1]")
        if int(self.stall_explore_calls) != self.stall_explore_calls:
            raise ValueError("stall_explore_calls must be integral")
        if int(self.known_side_padding_px) != self.known_side_padding_px:
            raise ValueError("known_side_padding_px must be integral")


@dataclass
class Opportunity:
    uid: int
    anchor: np.ndarray
    support: Set[Tuple[int, int]]
    parent_ids: Tuple[int, ...] = ()
    last_seen: int = 0


def _default_contours(obstacle):
    import cv2
    from frontier_exploration.frontier_detection import detect_frontiers
    explored = cv2.dilate(obstacle.explored_area.astype(np.uint8), np.ones((5, 5), np.uint8))
    explored[obstacle._navigable_map == 0] = 0
    return detect_frontiers(obstacle._navigable_map.astype(np.uint8), explored,
                            obstacle._area_thresh_in_pixels)


def _midpoint(curve):
    curve = np.asarray(curve, dtype=float).reshape(-1, 2)
    if len(curve) == 1:
        return curve[0].copy()
    lengths = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    total = float(lengths.sum())
    if total == 0:
        return curve[0].copy()
    cumulative = np.cumsum(lengths)
    i = min(int(np.searchsorted(cumulative, total / 2, side="right")), len(lengths) - 1)
    before = cumulative[i - 1] if i else 0.0
    return curve[i] + (curve[i + 1] - curve[i]) * ((total / 2 - before) / lengths[i])


def _pixel(obstacle, xy):
    return np.asarray(obstacle._xy_to_px(np.asarray(xy, dtype=float).reshape(1, 2))[0], dtype=int)


def _valid_pixel(mask, xy_px):
    x, y = map(int, xy_px)
    return 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and bool(mask[y, x])


def short_known_path(mask, a, b, max_steps):
    """Bounded 4-connected path, pixel order XY externally / row-column internally."""
    a, b = tuple(map(int, a)), tuple(map(int, b))
    if not _valid_pixel(mask, a) or not _valid_pixel(mask, b):
        return False
    if abs(a[0] - b[0]) + abs(a[1] - b[1]) > max_steps:
        return False
    queue, visited = deque([(a, 0)]), {a}
    while queue:
        p, distance = queue.popleft()
        if p == b:
            return True
        if distance >= max_steps:
            continue
        for q in ((p[0] - 1, p[1]), (p[0] + 1, p[1]), (p[0], p[1] - 1), (p[0], p[1] + 1)):
            if q not in visited and _valid_pixel(mask, q):
                visited.add(q)
                queue.append((q, distance + 1))
    return False


class FrontierOCS:
    """Prototype without permanent deletion or learned semantic re-ranking.

    One canonical proposal per *map-extracted frontier curve*. Only its local
    unknown-side boundary support is used for temporal association. This is not
    a semantic room model or full 3-D predicted visibility implementation.
    """

    def __init__(self, config: Optional[OCSConfig] = None, contour_extractor: Optional[Callable] = None):
        self.config = config or OCSConfig()
        self.contour_extractor = contour_extractor or _default_contours
        self.tasks: Dict[int, Opportunity] = {}
        self.cell_attempts: Dict[Tuple[int, int], int] = {}
        # Attempted cells alone cannot distinguish opposite approaches to the
        # same unknown boundary. Keep the actual attempted anchor as well.
        self.attempt_records = []
        self.next_uid = 1
        self.tick = 0
        self.active_id = None
        self.progress_tick = 0
        self.best_rho = float("inf")
        self.progress_coverage = 0
        self.ever_explored = None

    def _support(self, obstacle, anchor_px, known, navigable):
        ppm = float(obstacle.pixels_per_meter)
        radius = max(1, int(round(self.config.support_radius_m * ppm)))
        x, y = map(int, anchor_px)
        x0, x1 = max(0, x - radius), min(known.shape[1], x + radius + 1)
        y0, y1 = max(0, y - radius), min(known.shape[0], y + radius + 1)
        # Only the first unknown layer adjoining known space, not an assumed
        # unoccluded volume beyond unknown boundaries.
        ys, xs = np.nonzero(self.unknown_rim[y0:y1, x0:x1])
        result = set()
        bin_size = max(1, int(round(self.config.support_bin_m * ppm)))
        for xx, yy in zip(xs + x0, ys + y0):
            if (xx - x) ** 2 + (yy - y) ** 2 > radius ** 2:
                continue
            n = max(abs(int(xx) - x), abs(int(yy) - y)) + 1
            line_x = np.rint(np.linspace(x, xx, n)).astype(int)
            line_y = np.rint(np.linspace(y, yy, n)).astype(int)
            if not np.all(navigable[line_y, line_x]):
                continue
            # The ray may terminate in unknown space but may not pass through it.
            if n > 2 and not np.all(known[line_y[:-2], line_x[:-2]]):
                continue
            result.add((int(xx) // bin_size, int(yy) // bin_size))
        return result

    def _canonical(self, frontiers, obstacle, known, navigable):
        proposal_px = obstacle._xy_to_px(frontiers)
        max_gap = self.config.proposal_snap_radius_m * obstacle.pixels_per_meter
        candidates = []
        legal_mask = known & navigable
        for raw_curve in self.contour_extractor(obstacle):
            curve = np.asarray(raw_curve).reshape(-1, 2)
            if not len(curve):
                continue
            # Proposal multiplicity/order/representative point do not alter the
            # map-extracted canonical curve or its score support.
            gap = np.min(np.linalg.norm(curve[:, None, :] - proposal_px[None, :, :], axis=2))
            if gap > max_gap:
                continue
            px = np.rint(curve).astype(int)
            legal = ((px[:, 0] >= 0) & (px[:, 0] < known.shape[1])
                     & (px[:, 1] >= 0) & (px[:, 1] < known.shape[0]))
            valid_indices = np.flatnonzero(legal)
            legal[valid_indices] &= legal_mask[px[valid_indices, 1], px[valid_indices, 0]]
            if not np.any(legal):
                continue
            midpoint = _midpoint(curve)
            valid_curve = curve[legal]
            anchor_px = valid_curve[np.argmin(np.linalg.norm(valid_curve - midpoint, axis=1))]
            support = self._support(obstacle, anchor_px, known, navigable)
            if not support:
                continue
            anchor = obstacle._px_to_xy(np.asarray(anchor_px).reshape(1, 2))[0]
            candidates.append((np.asarray(anchor, dtype=float), support))
        # Deterministic order, independent of upstream proposal ordering.
        return sorted(candidates, key=lambda c: (float(c[0][0]), float(c[0][1]), tuple(sorted(c[1]))))

    def _associate(self, candidates, obstacle, known, step):
        seen, result = set(), []
        limit = int(np.ceil(2 * self.config.history_match_radius_m * obstacle.pixels_per_meter))
        previous = list(self.tasks.values())
        old_anchors = np.array([old.anchor for old in previous]).reshape(-1, 2)
        for anchor, support in candidates:
            parents = []
            nearby = np.flatnonzero(np.linalg.norm(old_anchors - anchor, axis=1)
                                    <= self.config.history_match_radius_m)
            for index in nearby:
                old = previous[int(index)]
                overlap = len(support & old.support) / max(1, min(len(support), len(old.support)))
                if overlap < self.config.support_overlap_min:
                    continue
                if not short_known_path(known, _pixel(obstacle, anchor), _pixel(obstacle, old.anchor), limit):
                    continue
                parents.append((overlap, old))
            parents.sort(key=lambda p: (-p[0], p[1].uid))
            unused = [old for _, old in parents if old.uid not in seen]
            uid = unused[0].uid if unused else self.next_uid
            if not unused:
                self.next_uid += 1
            stable_anchor = anchor
            if unused and _valid_pixel(known, _pixel(obstacle, unused[0].anchor)):
                # Being in known free space is necessary but not sufficient:
                # the previous anchor must still expose the *current* residual
                # opportunity. Otherwise update it and let PointNav reset.
                current_support = self._support(obstacle, _pixel(obstacle, unused[0].anchor),
                                                known, obstacle._navigable_map)
                overlap = len(current_support & support) / max(1, min(len(current_support), len(support)))
                if current_support and overlap >= self.config.support_overlap_min:
                    stable_anchor = unused[0].anchor.copy()
            task = Opportunity(uid, stable_anchor, support, tuple(old.uid for _, old in parents), int(step))
            self.tasks[uid] = task
            seen.add(uid)
            result.append(task)
        return result

    def _attempt(self, task):
        # Spatial ledger, not a completed boolean copied to all children.
        for cell in task.support:
            self.cell_attempts[cell] = self.cell_attempts.get(cell, 0) + 1
        self.attempt_records.append((task.anchor.copy(), frozenset(task.support)))

    def _attempt_counts(self, task, obstacle, known):
        """Inherit only intersecting spatial attempts from the same local approach."""
        counts = dict.fromkeys(task.support, 0)
        limit = int(np.ceil(2 * self.config.history_match_radius_m * obstacle.pixels_per_meter))
        anchor_px = _pixel(obstacle, task.anchor)
        for attempted_anchor, support in self.attempt_records:
            intersection = task.support & support
            if not intersection or np.linalg.norm(task.anchor - attempted_anchor) > self.config.history_match_radius_m:
                continue
            if not short_known_path(known, anchor_px, _pixel(obstacle, attempted_anchor), limit):
                continue
            for cell in intersection:
                counts[cell] += 1
        return list(counts.values())

    def select(self, frontiers, robot_xy, obstacle_map, score_fn, step,
               baseline_select: Optional[Callable] = None):
        """Select an exploration goal, preserving native selection on fallback.

        ``baseline_select`` is a zero-argument callback running the full native
        selector, including keep-last and cyclic-history behavior. Production
        adapters must provide it. The score-only fallback exists for standalone
        geometry tests and is explicitly distinguished in diagnostics.
        """
        frontiers = np.asarray(frontiers, dtype=float).reshape(-1, 2)
        robot_xy = np.asarray(robot_xy, dtype=float)
        if not len(frontiers) or not np.isfinite(frontiers).all():
            raise ValueError("select requires nonempty finite raw frontiers; native NF stays upstream")
        if robot_xy.shape != (2,) or not np.isfinite(robot_xy).all():
            raise ValueError("robot_xy must contain two finite coordinates")
        self.tick += 1
        obstacle = obstacle_map
        navigable = np.asarray(obstacle._navigable_map, dtype=bool)
        explored = np.asarray(obstacle.explored_area, dtype=bool)
        # Default radius two matches upstream's 5x5 frontier padding; this is
        # processed map geometry, not proof that all pixels were actually seen.
        width = 2 * self.config.known_side_padding_px + 1
        known = ndimage.binary_dilation(explored, structure=np.ones((width, width), bool)) & navigable
        self.unknown_rim = ndimage.binary_dilation(known, structure=np.ones((3, 3), bool)) & ~known & navigable
        if self.ever_explored is None or self.ever_explored.shape != explored.shape:
            self.ever_explored = explored.copy()
        else:
            self.ever_explored |= explored
        coverage = int(np.count_nonzero(self.ever_explored))
        # The native frontier generator is geometric, not a reachability oracle.
        # Avoid adding a canonical anchor in another disconnected known region.
        components, _ = ndimage.label(known)
        robot_px = _pixel(obstacle, robot_xy)
        if _valid_pixel(known, robot_px):
            component = components[robot_px[1], robot_px[0]]
            reachable_known = known & (components == component)
        else:
            reachable_known = np.zeros_like(known)
        previous_anchor = (self.tasks[self.active_id].anchor.copy()
                           if self.active_id in self.tasks else None)
        candidates = self._canonical(frontiers, obstacle, reachable_known, navigable)
        tasks = self._associate(candidates, obstacle, known, step)
        diag = {"schema": "vlfm.ocs.prototype.v1", "step": int(step), "raw_proposals": len(frontiers),
                "opportunities": len(tasks), "coverage_kind": "processed_explored_mask_union_proxy",
                "config": asdict(self.config), "task_records": [], "method_is_calibrated": False}
        if not tasks:
            self.active_id = None
            if baseline_select is not None:
                goal, value = baseline_select()
                fallback_kind = "upstream_selection"
            else:
                points, values = score_fn(frontiers.copy())
                goal, value = points[0], values[0]
                fallback_kind = "score_only"
            goal = np.asarray(goal, dtype=float)
            if goal.shape != (2,) or not np.isfinite(goal).all() or not np.isfinite(value):
                raise ValueError("Fallback selector must return a finite 2-D goal and score")
            diag.update(reason=f"geometry_unavailable_{fallback_kind}_fallback",
                        fallback_kind=fallback_kind, task_id=None)
            return goal.copy(), float(value), diag
        anchors = np.asarray([task.anchor for task in tasks])
        ranked, values = score_fn(anchors.copy())
        scores = {}
        for point, value in zip(ranked, values):
            for task in tasks:
                if np.array_equal(point, task.anchor):
                    scores[task.uid] = float(value)
        if any(task.uid not in scores or not np.isfinite(scores[task.uid]) for task in tasks):
            raise ValueError("score_fn must return a finite value for every anchor")
        active = next((task for task in tasks if task.uid == self.active_id), None)
        reason = "initial_selection" if self.active_id is None else "active_support_or_anchor_invalid"
        if active is not None and previous_anchor is not None and not np.array_equal(active.anchor, previous_anchor):
            # An invalid observation anchor may be replaced within the same
            # task identity. It still starts a fresh physical navigation goal.
            active = None
            reason = "active_anchor_updated_for_current_support"
        if active is not None:
            rho = float(np.linalg.norm(robot_xy - active.anchor))
            area = (coverage - self.progress_coverage) / obstacle.pixels_per_meter ** 2
            if rho < self.best_rho - self.config.progress_distance_m or area >= self.config.progress_area_m2:
                self.best_rho = min(self.best_rho, rho)
                self.progress_tick = self.tick
                self.progress_coverage = coverage
            if rho <= self.config.arrival_radius_m:
                self._attempt(active)
                reason = "anchor_reached_attempt_recorded_not_target_absence"
            elif self.tick - self.progress_tick >= self.config.stall_explore_calls:
                self._attempt(active)
                reason = "no_distance_or_map_progress_defer"
            else:
                diag.update(reason="continue_valid_opportunity_anchor", task_id=active.uid)
                diag["task_records"] = self._records(tasks, scores, obstacle, known)
                return active.anchor.copy(), scores[active.uid], diag
        attempt_counts = {task.uid: self._attempt_counts(task, obstacle, known) for task in tasks}
        def priority(task):
            attempts = attempt_counts[task.uid]
            attempted_fraction = sum(a > 0 for a in attempts) / max(1, len(attempts))
            deferred = attempted_fraction >= self.config.support_overlap_min
            return (int(deferred), sum(attempts) / max(1, len(attempts)), -scores[task.uid], task.uid)
        selected = min(tasks, key=priority)
        self.active_id = selected.uid
        self.best_rho = float(np.linalg.norm(robot_xy - selected.anchor))
        self.progress_tick = self.tick
        self.progress_coverage = coverage
        diag.update(reason=reason, task_id=selected.uid,
                    all_deferred_retry=all(priority(t)[0] for t in tasks))
        diag["task_records"] = self._records(tasks, scores, obstacle, known)
        return selected.anchor.copy(), scores[selected.uid], diag

    def _records(self, tasks, scores, obstacle, known):
        return [{"task_id": t.uid, "parent_ids": list(t.parent_ids), "anchor_xy": t.anchor.tolist(),
                 "support_cells": len(t.support), "attempted_support_cells": sum(a > 0 for a in self._attempt_counts(t, obstacle, known)),
                 "value": scores[t.uid]} for t in tasks]
