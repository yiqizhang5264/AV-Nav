"""Explicit, reversible OCS hooks; pinned VLFM and target/STOP logic stay intact.

This module imports neither Torch nor Habitat at module import time. The original
condition uses the exact upstream selection method, plus observational logging.
Only the exploration frontier selection is replaced for the OCS condition.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np


def plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def anchor_valid(policy: Any, anchor: np.ndarray, proposal: np.ndarray) -> bool:
    """Require the anchor and short connection to its current proposal be free.

    Uses upstream pixel convention: _xy_to_px returns x,y, NumPy uses row=y.
    This is a geometric validity check, not a claim that the room was observed.
    """
    obstacle = getattr(policy, "_obstacle_map", None)
    if obstacle is None or not hasattr(obstacle, "_navigable_map"):
        return False
    count = max(2, int(np.ceil(np.linalg.norm(proposal - anchor)
                               * obstacle.pixels_per_meter * 2)) + 1)
    xy = np.linspace(anchor, proposal, count)
    pixels = obstacle._xy_to_px(xy)
    nav = obstacle._navigable_map
    x, y = pixels[:, 0], pixels[:, 1]
    if np.any(x < 0) or np.any(y < 0) or np.any(x >= nav.shape[1]) or np.any(y >= nav.shape[0]):
        return False
    return bool(np.all(nav[y, x]))


def default_context() -> dict:
    # EvidenceRecorder exists before policy.act in the preserved evidence overlay.
    from vlfm_evidence_recorder import get_evidence_recorder
    recorder = get_evidence_recorder()
    return {"episode_evidence_dir": str(recorder.episode_dir) if recorder.episode_dir else None,
            "evidence_step": recorder.step}


@dataclass
class _RuntimeState:
    scheduler: Any = None
    anchor: Any = None
    actual_resets: int = 0
    reason: str = "outside_pointnav"


class HookHandle:
    def __init__(self, policy_class, object_class, methods, low_hooks):
        self.policy_class, self.object_class = policy_class, object_class
        self.methods, self.low_hooks = methods, low_hooks

    def restore(self):
        self.policy_class._get_best_frontier = self.methods["select"]
        self.policy_class._reset = self.methods["reset"]
        self.object_class._pointnav = self.methods["pointnav"]
        for low, original in self.low_hooks:
            low.reset = original


def install_policy_hooks(*, policy_class=None, object_class=None,
                         settings=None, emit: Callable | None = None,
                         context: Callable | None = None,
                         scheduler_factory: Callable | None = None):
    """Install once per evaluation process. Injectable classes enable CPU tests."""
    if policy_class is None or object_class is None:
        from vlfm.policy.itm_policy import BaseITMPolicy
        from vlfm.policy.base_objectnav_policy import BaseObjectNavPolicy
        policy_class, object_class = BaseITMPolicy, BaseObjectNavPolicy
    if settings is None:
        settings = json.loads(Path(os.environ["VLFM_OCS_SETTINGS"]).read_text(encoding="utf-8"))
    mode = settings["mode"]
    if mode not in {"original", "anchor_only", "ocs"}:
        raise ValueError(f"Unknown OCS mode: {mode}")
    hold_radius = float(settings.get("anchor_match_radius_m", 0.5))
    if not math.isfinite(hold_radius) or hold_radius <= 0:
        raise ValueError("anchor_match_radius_m must be finite and positive")
    if scheduler_factory is None and mode == "ocs":
        from av_nav.frontier_ocs import FrontierOCS, OCSConfig
        cfg = OCSConfig(**settings.get("scheduler", {}))
        scheduler_factory = lambda: FrontierOCS(cfg)
    if context is None:
        context = default_context
    if emit is None:
        output = Path(os.environ["VLFM_OCS_LOG"])
        def emit(row):
            with output.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(plain(row), ensure_ascii=False, allow_nan=False) + "\n")
    methods = {"select": policy_class._get_best_frontier,
               "reset": policy_class._reset, "pointnav": object_class._pointnav}
    low_hooks = []

    def report(policy, kind, **data):
        emit(plain({"schema": "vlfm.ocs.online.v1", "kind": kind, "mode": mode,
                    "policy_step": int(getattr(policy, "_num_steps", -1)),
                    **context(), **data}))

    def ensure(policy):
        if not hasattr(policy, "_ocs_runtime_state"):
            policy._ocs_runtime_state = _RuntimeState(
                scheduler=scheduler_factory() if mode == "ocs" else None)
        low = policy._pointnav_policy
        if not any(instance is low for instance, _ in low_hooks):
            original_low_reset = low.reset
            def actual_reset(*a, **kw):
                # Count only a reset callback that actually ran successfully.
                result = original_low_reset(*a, **kw)
                state = policy._ocs_runtime_state
                state.actual_resets += 1
                report(policy, "pointnav_reset_called", reason=state.reason,
                       actual_resets_episode=state.actual_resets)
                return result
            low.reset = actual_reset
            low_hooks.append((low, original_low_reset))
        return policy._ocs_runtime_state

    def reset(policy):
        ensure(policy)
        policy._ocs_runtime_state = _RuntimeState(
            scheduler=scheduler_factory() if mode == "ocs" else None,
            reason="episode_reset")
        result = methods["reset"](policy)
        report(policy, "episode_policy_reset")
        policy._ocs_runtime_state.reason = "outside_pointnav"
        return result

    def select(policy, observations, frontiers):
        state = ensure(policy)
        robot = np.array(policy._observations_cache["robot_xy"], copy=True)
        prior_goal = np.array(policy._last_goal, copy=True)
        proposal = None
        if mode == "ocs":
            def scorer(points):
                return policy._sort_frontiers_by_value(observations, points)
            goal, value, diag = state.scheduler.select(
                frontiers, robot, policy._obstacle_map, scorer, int(policy._num_steps))
            goal = np.asarray(goal, dtype=float).reshape(2)
            # Display/state bookkeeping only; the OCS condition does not claim to
            # execute the original cyclic selector or a separately fixed version.
            policy._last_frontier = goal.copy()
            policy._last_value = float(value)
        else:
            goal, value = methods["select"](policy, observations, frontiers)
            goal = np.array(goal, copy=True)
            proposal = goal.copy()
            diag = {"reason": "upstream_selection"}
            if mode == "anchor_only":
                old = state.anchor
                keep = (old is not None
                        and np.linalg.norm(old - robot) >= policy._pointnav_stop_radius
                        and np.linalg.norm(old - goal) <= hold_radius
                        and anchor_valid(policy, old, goal))
                if keep:
                    goal = old.copy()
                    diag = {"reason": "keep_valid_anchor", "upstream_score_is_proposal_score": True}
                else:
                    diag = {"reason": "new_or_released_anchor"}
                state.anchor = goal.copy()
        if goal.shape != (2,) or not np.all(np.isfinite(goal)):
            raise ValueError("Exploration scheduler returned invalid goal; refusing false STOP")
        report(policy, "frontier_selection", raw_frontiers=np.asarray(frontiers),
               upstream_proposal=proposal, selected_goal=goal, selected_value=float(value),
               last_pointnav_goal=prior_goal,
               expected_reset_predicate=bool(np.linalg.norm(goal - prior_goal) > 0.1),
               diagnostics=diag)
        return goal, value

    def pointnav(policy, goal, stop=False):
        state = ensure(policy)
        previous = np.array(policy._last_goal, copy=True)
        goal_copy = np.array(goal, copy=True)
        count_before = state.actual_resets
        state.reason = "target_navigation" if stop else "exploration_navigation"
        # Any switch to a real object ends the anchor-only exploration commitment.
        # Target position, stop flag, return action and upstream reset semantics stay intact.
        if stop:
            state.anchor = None
        try:
            result = methods["pointnav"](policy, goal, stop=stop)
        finally:
            report(policy, "pointnav_call", goal=goal_copy, prior_goal=previous,
                   stop_allowed=bool(stop),
                   expected_reset_predicate=bool(np.linalg.norm(goal_copy - previous) > 0.1),
                   actual_resets_this_call=state.actual_resets - count_before)
            state.reason = "outside_pointnav"
        return result

    policy_class._get_best_frontier = select
    policy_class._reset = reset
    object_class._pointnav = pointnav
    return HookHandle(policy_class, object_class, methods, low_hooks)


_HANDLE = None


def install_from_environment():
    global _HANDLE
    if _HANDLE is None:
        _HANDLE = install_policy_hooks()
    return _HANDLE
