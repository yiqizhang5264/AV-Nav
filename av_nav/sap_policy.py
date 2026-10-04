"""Category-only SAP-Nav adapter over the untouched, pinned VLFM policy."""
import json
import math
import os
from pathlib import Path
import numpy as np
from habitat_baselines.common.baseline_registry import baseline_registry
from vlfm.policy.habitat_policies import HabitatITMPolicyV2, TorchActionIDs
from vlfm.mapping.obstacle_map import ObstacleMap
from .geometry import observe, angle_delta
from .planner import Grid
from .sap_geometry import HeightMap, depth_points, select_view
from .sap_vlm import Verifier
from .trace import native


class CandidateSAM:
    def __init__(self, model, owner):
        self.model, self.owner = model, owner

    def segment_bbox(self, rgb, bbox):
        mask = self.model.segment_bbox(rgb, bbox)
        owner = self.owner
        data = owner._sap_rgbd
        item = observe(rgb, data[1], mask, *data[2:], owner._num_steps)
        if item is not None:
            owner._sap_observations.append(item)
        return mask


@baseline_registry.register_policy
class SAPCategoryPolicy(HabitatITMPolicyV2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sap = json.loads(Path(os.environ['SAP_CONFIG']).read_text())
        self.verifier = Verifier(self.sap['vlm'])
        self._mobile_sam = CandidateSAM(self._mobile_sam, self)
        self._sap_map = ObstacleMap(min_height=.15, max_height=.88, agent_radius=.18, hole_area_thresh=-1)
        self._sap_episode = -2
        self._sap_reset()

    def _sap_reset(self):
        self._sap_episode += 1
        self._sap_observations = []
        self._sap_decisions = []
        self._sap_active = None
        self._sap_calls = self._sap_moves = self._sap_triggers = 0
        self._sap_map.reset()
        self._sap_heights = HeightMap(self._sap_map._map.shape, self._sap_map._xy_to_px)

    def _reset(self):
        super()._reset()
        if hasattr(self, 'sap'):
            self._sap_reset()

    def _event(self, event, **data):
        with (Path(os.environ['AV_RUN_DIR'])/'verification.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(episode_index=self._sap_episode, step=self._num_steps,
                                         target=self._target_object, event=event, **data), default=native)+'\n')

    def _pre_step(self, observations, masks):
        super()._pre_step(observations, masks)
        self._sap_raw = observations
        self._sap_observations = []
        _, depth, tf, lo, hi, fx, fy = self._observations_cache['object_map_rgbd'][0]
        self._sap_map.update_map(depth, tf, lo, hi, fx, fy, self._camera_fov)
        self._sap_heights.update(depth_points(depth, tf, lo, hi, fx, fy))

    def _update_object_map(self, *args):
        self._sap_rgbd = args
        return super()._update_object_map(*args)

    def _distance(self, observation, goal):
        return float(np.linalg.norm(observation.points[:, :2]-goal[:2], axis=1).min())

    def _matching(self, goal):
        items = [o for o in self._sap_observations if self._distance(o, goal) < self.sap['association_radius']]
        return max(items, key=lambda o: o.quality) if items else None

    def _get_target_object_location(self, position):
        if self._sap_active is not None:
            return self._sap_active['goal']
        # Remove rejected local point clusters before upstream selects its nearest
        # candidate, so a blacklisted nearest object cannot hide all other objects.
        cloud = self._object_map.clouds.get(self._target_object)
        if cloud is not None:
            for center, accepted in self._sap_decisions:
                if not accepted:
                    cloud = cloud[np.linalg.norm(cloud[:, :2]-center, axis=1) >= self.sap['association_radius']]
            self._object_map.clouds[self._target_object] = cloud
            self._object_map.last_target_coord = None
        return super()._get_target_object_location(position)

    def _ask(self, obs, kind):
        result, record = self.verifier.ask(obs, self._target_object.split('|')[0], kind)
        self._sap_calls += 1
        folder = Path(os.environ['AV_RUN_DIR'])/'evidence'
        folder.mkdir(exist_ok=True)
        name = f'ep{self._sap_episode}_step{self._num_steps}_{self._sap_calls}.png'
        from PIL import Image
        Image.fromarray(obs.rgb).save(folder/name)
        self._event(kind, result=result, source_step=obs.step, bbox=obs.bbox,
                    evidence='evidence/'+name, **record)
        return result

    def _resume(self):
        self._called_stop = False
        return super()._explore(self._sap_raw)

    def _verify(self):
        state = self._sap_active
        accepted = self._ask(state['best'], 'category')['matches']
        self._sap_decisions.append((state['center'].copy(), accepted))
        self._event('decision', accepted=accepted, goal=state['goal'], center=state['center'],
                    attempts=state['attempts'], best_score=state['score'])
        self._sap_active = None
        if accepted:
            return super()._pointnav(state['goal'], stop=True)
        return self._resume()

    def _next_view(self):
        state = self._sap_active
        if state['attempts'] >= self.sap['max_repositions']:
            return False
        obstacle = self._sap_map
        grid = Grid(obstacle.explored_area & obstacle._navigable_map.astype(bool),
                    obstacle._map, obstacle.pixels_per_meter, obstacle._xy_to_px, obstacle._px_to_xy)
        rgbd = self._observations_cache['object_map_rgbd'][0]
        vfov = 2*math.atan(rgbd[1].shape[0]/(2*rgbd[6]))
        best, candidates = select_view(state['best'].points, self._observations_cache['robot_xy'],
            grid, self._sap_heights, self._camera_height, vfov, state['visited'], self.sap['max_path_m'])
        self._event('view_selection', feasible=len(candidates), selected=best)
        if best is None:
            return False
        state['attempts'] += 1
        state['visited'].append(best['xy'])
        state['waypoint'] = best['xy']
        state['path'] = list(best['path'][::max(1, int(grid.ppm*.3))]) + [best['xy']]
        state['start_step'] = self._num_steps
        self._sap_moves += 1
        return True

    def _score(self, obs):
        value = self._ask(obs, 'sufficiency')
        score = value['visibility'] + value['perspective']
        state = self._sap_active
        if score > state['score']:
            state['best'], state['score'] = obs, score
        return score

    def _pointnav(self, goal, stop=False):
        if not stop:
            return super()._pointnav(goal, stop=False)
        robot = self._observations_cache['robot_xy']
        if self._sap_active is None:
            obs = self._matching(goal)
            if any(accepted and np.linalg.norm(goal-center) < self.sap['association_radius']
                   for center, accepted in self._sap_decisions):
                return super()._pointnav(goal, stop=True)
            if obs is None:
                if np.linalg.norm(robot-goal) <= self._pointnav_stop_radius:
                    self._event('stale_candidate', goal=goal)
                    self._sap_decisions.append((goal.copy(), False))
                    return self._resume()
                action = super()._pointnav(goal, stop=False)
                return TorchActionIDs.TURN_LEFT if int(action.item()) == 0 else action
            self._sap_active = dict(goal=goal.copy(), center=obs.center[:2].copy(), best=obs,
                                   score=-1, attempts=0, visited=[robot.copy()])
            if self._score(obs) >= self.sap['sufficiency_threshold']:
                return self._verify()
            self._sap_triggers += 1
            if not self._next_view():
                return self._verify()
        state = self._sap_active
        if self._num_steps-state['start_step'] >= self.sap['max_reposition_steps']:
            self._event('reposition_timeout')
            if not self._next_view():
                return self._verify()
        while state['path'] and np.linalg.norm(robot-state['path'][0]) <= self.sap['arrival_radius']:
            state['path'].pop(0)
        if state['path']:
            action = super()._pointnav(state['path'][0], stop=False)
            return TorchActionIDs.TURN_LEFT if int(action.item()) == 0 else action
        bearing = math.atan2(*(state['center']-robot)[::-1])
        error = angle_delta(bearing, self._observations_cache['robot_heading'])
        if abs(error) > math.radians(16):
            return TorchActionIDs.TURN_LEFT if error > 0 else TorchActionIDs.TURN_RIGHT
        obs = self._matching(state['goal'])
        if obs is not None and self._score(obs) >= self.sap['sufficiency_threshold']:
            return self._verify()
        if obs is None:
            self._event('candidate_not_visible')
        if not self._next_view():
            return self._verify()
        return TorchActionIDs.TURN_LEFT

    def _get_policy_info(self, detections):
        info = super()._get_policy_info(detections)
        info.update(sap_vlm_calls=self._sap_calls, sap_repositions=self._sap_moves,
                    sap_triggers=self._sap_triggers)
        return info
