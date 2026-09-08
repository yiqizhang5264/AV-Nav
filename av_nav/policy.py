"""Pinned VLFM adapter. Only this file depends on Habitat, torch and VLFM."""
import json
import math
import os
from pathlib import Path
import numpy as np
from habitat_baselines.common.baseline_registry import baseline_registry
from vlfm.policy.habitat_policies import HabitatITMPolicyV2, TorchActionIDs
from vlfm.mapping.obstacle_map import ObstacleMap
from .config import Config
from .geometry import observe, angle_delta
from .planner import Grid, select_view
from .trace import TraceMixin, native


class CapturingSAM:
    def __init__(self, model, owner):
        self.model, self.owner = model, owner

    def segment_bbox(self, rgb, bbox):
        mask = self.model.segment_bbox(rgb, bbox)
        rgbd = self.owner._capture_rgbd
        item = observe(rgb, rgbd[1], mask, *rgbd[2:], self.owner._num_steps)
        if item is not None:
            self.owner._av_observations.append(item)
        return mask


@baseline_registry.register_policy
class AVHabitatPolicy(TraceMixin, HabitatITMPolicyV2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.av = Config.load(os.environ["AV_CONFIG"])
        self._mobile_sam = CapturingSAM(self._mobile_sam, self)
        # A separate observed-space map: preserve VLFM's original exploration path.
        self._av_map = ObstacleMap(min_height=self._camera_height - .65,
                                  max_height=self._camera_height - .38,
                                  agent_radius=.18, hole_area_thresh=-1)
        self._av_episode = -1
        self._av_reset()

    def _av_reset(self):
        self._av_episode += 1
        self._av_observations = []
        self._av_active = None
        self._av_decisions = []
        self._av_total_actions = 0
        self._av_calls = 0
        self._av_path = 0.0
        self._av_triggers = 0
        self._av_map.reset()
        self._av_rng = np.random.default_rng(self.av.seed + self._av_episode)

    def _reset(self):
        super()._reset()
        if hasattr(self, "av"):
            self._av_reset()

    def _pre_step(self, observations, masks):
        super()._pre_step(observations, masks)
        self._av_raw_observations = observations
        self._av_observations = []
        if self.av.strategy != "baseline":
            rgb, depth, tf, lo, hi, fx, fy = self._observations_cache["object_map_rgbd"][0]
            self._av_map.update_map(depth, tf, lo, hi, fx, fy, self._camera_fov)

    def _update_object_map(self, *args):
        self._capture_rgbd = args
        return super()._update_object_map(*args)

    def _get_target_object_location(self, position):
        if self._av_active is not None:
            return self._av_active["goal"]
        goal = super()._get_target_object_location(position)
        if goal is not None:
            for center, status, until in self._av_decisions:
                if np.linalg.norm(goal[:2]-center) < self.av.association_radius and until >= self._num_steps and status != "confirmed":
                    self._event('candidate_suppressed',goal=goal,center=center,status=status,until=until)
                    return None
        return goal

    def _event(self, kind, **values):
        path = Path(os.environ["AV_RUN_DIR"]) / "verification.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(dict(**self._trace_identity(), step=self._num_steps,
                                   candidate_spatial_id=self._trace_candidate,event=kind, **values), default=native) + "\n")

    def _score(self, obs):
        if obs.score is not None:
            return obs.score
        x1,y1,x2,y2 = obs.bbox
        crop = obs.rgb[y1:y2,x1:x2]
        targets = self._target_object.split("|")
        target_text = "/".join(targets)
        negatives = [c for c in self.av.negatives if c not in targets]
        # Fixed contrast vocabulary shared by every verification strategy.
        positive = self._itm.cosine(crop, f"a photo of a {target_text}")
        negative = self._itm.cosine(crop, "a photo of a " + ", ".join(negatives))
        self._av_calls += 2
        obs.score = float(positive-negative)  # ITC margin, NOT a calibrated probability.
        evidence_path = None
        if self.av.record_evidence:
            import cv2
            folder=Path(os.environ['AV_RUN_DIR'])/'evidence'
            folder.mkdir(exist_ok=True)
            evidence_path=f'evidence/ep{self._av_episode}_step{obs.step}_call{self._av_calls}.jpg'
            if not cv2.imwrite(str(Path(os.environ['AV_RUN_DIR'])/evidence_path),cv2.cvtColor(crop,cv2.COLOR_RGB2BGR)):
                raise RuntimeError('Could not save candidate evidence image')
        self._event("evidence", score=obs.score, quality=obs.quality, area=obs.area,
                    center=obs.center, position=obs.position, source_step=obs.step,
                    bbox=obs.bbox,positive=float(positive),negative=float(negative),
                    target=self._target_object, crop=evidence_path)
        return obs.score

    def _matching_observation(self, goal):
        items = [o for o in self._av_observations if np.linalg.norm(o.center[:2]-goal[:2]) <= self.av.association_radius]
        self._event('association',goal=goal,observations=[dict(center=o.center,distance=float(np.linalg.norm(o.center[:2]-goal[:2])),quality=o.quality,bbox=o.bbox) for o in self._av_observations],matched=len(items))
        return max(items, key=lambda o:o.quality) if items else None

    def _resume(self):
        action = super()._explore(self._av_raw_observations)
        self._called_stop = False
        self._event('exploration_resume',action=int(action.item()))
        return action

    def _finish(self, status):
        state = self._av_active
        self._av_decisions.append((state["goal"].copy(), status,
                                  self._num_steps + self.av.cooldown_steps if status != "confirmed" else math.inf))
        self._event("finish", status=status, rounds=state["round"],
                    path=state["travel"], actions=state["actions"], goal=state["goal"])
        self._av_active = None
        if status == "confirmed":
            return super()._pointnav(state["goal"], stop=True)
        return self._resume()

    def _new_round(self, state):
        cfg = self.av
        if state["round"] >= cfg.max_rounds:
            return False
        obs = max(state["evidence"], key=lambda o:o.quality)
        state["round"] += 1
        state["scan"] = 0
        state["samples"] = []
        if cfg.strategy == "passive":
            state["waypoints"] = []
            state["scan_angles"] = [0.0] * len(cfg.angles)
        else:
            grid = Grid(self._av_map.explored_area & self._av_map._navigable_map.astype(bool),
                        self._av_map._map, self._av_map.pixels_per_meter,
                        self._av_map._xy_to_px, self._av_map._px_to_xy)
            view, candidates = select_view(obs, [o.position for o in state["evidence"]],
                                           self._observations_cache["robot_xy"], grid, cfg, self._av_rng)
            if view is None:
                self._event("no_feasible_view")
                return False
            path = list(view["path"])
            state["waypoints"] = path[::max(1,int(grid.ppm*.35))]
            if path:
                state["waypoints"].append(path[-1])
            state["scan_angles"] = list(cfg.angles)
            self._event("view_selected", xy=view["xy"], planned_cost=view["cost"],
                        utility=view["utility"], feasible_count=len(candidates))
        return True

    def _pointnav(self, goal, stop=False):
        if not stop or self.av.strategy == "baseline":
            return super()._pointnav(goal, stop)
        cfg = self.av
        robot = self._observations_cache["robot_xy"]
        if self._av_active is None:
            confirmed = any(s == "confirmed" and np.linalg.norm(goal[:2]-p)<cfg.association_radius
                            for p,s,_ in self._av_decisions)
            if confirmed:
                self._event('decision',status='confirmed',reason='reuse_confirmation',goal=goal)
                return super()._pointnav(goal, stop=True)
            obs = self._matching_observation(goal)
            if obs is None:
                self._event('decision',status='approach_without_match',reason='no_matching_observation',goal=goal)
                # Allow approach, but do not let a stale map candidate directly STOP.
                if np.linalg.norm(robot-goal[:2]) < self._pointnav_stop_radius:
                    self._av_decisions.append((goal[:2].copy(), "unobserved", self._num_steps+cfg.cooldown_steps))
                    self._event('decision',status='unobserved',reason='prevent_stale_target_stop',goal=goal)
                    return self._resume()
                action = super()._pointnav(goal, stop=False)
                return TorchActionIDs.TURN_LEFT if int(action.item()) == 0 else action
            score = self._score(obs)
            if obs.quality >= cfg.min_quality and score >= cfg.high:
                self._event('decision',status='confirmed',reason='high_quality_high_margin',goal=goal,score=score,quality=obs.quality)
                self._av_decisions.append((goal[:2].copy(), "confirmed", math.inf))
                return super()._pointnav(goal, stop=True)
            if (obs.quality >= cfg.min_quality and score < cfg.low) or cfg.strategy == "single":
                self._event('decision',status='rejected',reason='low_margin_or_single',goal=goal,score=score,quality=obs.quality)
                self._av_decisions.append((goal[:2].copy(), "rejected", self._num_steps+cfg.cooldown_steps))
                return self._resume()
            if self._av_total_actions >= cfg.max_episode_actions:
                self._event('decision',status='budget',reason='episode_verification_budget',goal=goal)
                self._av_decisions.append((goal[:2].copy(), "budget", self._num_steps+cfg.cooldown_steps))
                return self._resume()
            self._av_triggers += 1
            self._av_active = dict(goal=goal[:2].copy(), evidence=[obs], round=0, actions=0,
                                   travel=0.0, last_position=robot.copy())
            self._event("trigger", goal=goal[:2], quality=obs.quality, score=score,
                        reason='low_quality' if obs.quality<cfg.min_quality else 'uncertain_margin')
            if not self._new_round(self._av_active):
                return self._finish("unresolved")
        state = self._av_active
        distance = float(np.linalg.norm(robot-state["last_position"]))
        state["travel"] += distance
        self._av_path += distance
        state["last_position"] = robot.copy()
        if state["actions"] >= cfg.max_actions or state["travel"] >= cfg.max_path or self._av_total_actions >= cfg.max_episode_actions:
            return self._finish("budget")
        state["actions"] += 1
        self._av_total_actions += 1
        while state["waypoints"] and np.linalg.norm(robot-state["waypoints"][0]) < cfg.arrival_radius:
            state["waypoints"].pop(0)
        if state["waypoints"]:
            action = super()._pointnav(state["waypoints"][0], stop=False)
            return TorchActionIDs.TURN_LEFT if int(action.item()) == 0 else action
        bearing = math.atan2(*(state["goal"]-robot)[::-1])
        offset = math.radians(state["scan_angles"][state["scan"]])
        error = angle_delta(bearing+offset, self._observations_cache["robot_heading"])
        if abs(error) > math.radians(16):  # Habitat default discrete turn is 30 degrees.
            return TorchActionIDs.TURN_LEFT if error > 0 else TorchActionIDs.TURN_RIGHT
        obs = self._matching_observation(state["goal"])
        if obs is not None:
            state["samples"].append(obs)
        state["scan"] += 1
        if state["scan"] < len(state["scan_angles"]):
            # No Habitat WAIT action: a bounded turn sequence supplies passive frames.
            return TorchActionIDs.TURN_LEFT
        if state["samples"]:
            obs = max(state["samples"], key=lambda o:o.quality)
            self._score(obs)
            state["evidence"].append(obs)
            reliable = [o for o in state["evidence"] if o.quality >= cfg.min_quality]
            if reliable:
                fused = float(np.average([o.score for o in reliable], weights=[o.quality for o in reliable]))
                self._event("fusion", margin=fused, observations=len(reliable))
                if fused >= cfg.high:
                    return self._finish("confirmed")
                if len(reliable) >= 2 and fused < cfg.low:
                    return self._finish("rejected")
        else:
            self._event("target_missing")  # Missing view is not a strong negative.
        if not self._new_round(state):
            return self._finish("unresolved")
        return TorchActionIDs.TURN_LEFT

    def _get_policy_info(self, detections):
        info = super()._get_policy_info(detections)
        info.update(av_actions=self._av_total_actions, av_path=self._av_path,
                    av_model_calls=self._av_calls, av_triggers=self._av_triggers,
                    av_episode_index=self._av_episode)
        info['av_active_at_step']=self._av_active is not None
        return info
