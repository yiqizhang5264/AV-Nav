"""Create a temporary VLFM package with evidence hooks, leaving upstream pinned."""

from __future__ import annotations

import pathlib
import shutil
import tempfile


def create_vlfm_evidence_overlay(vlfm_root: pathlib.Path, room_online: bool = False) -> tempfile.TemporaryDirectory[str]:
    overlay = tempfile.TemporaryDirectory(prefix="av-nav-vlfm-evidence-")
    package_dst = pathlib.Path(overlay.name) / "vlfm"
    shutil.copytree(vlfm_root / "vlfm", package_dst)
    trainer_path = package_dst / "utils" / "vlfm_trainer.py"
    source = trainer_path.read_text(encoding="utf-8")
    import_anchor = "from omegaconf import OmegaConf\n"
    source = source.replace(import_anchor, import_anchor + "from vlfm_evidence_recorder import get_evidence_recorder\n", 1)
    loop_anchor = "            current_episodes_info = self.envs.current_episodes()\n\n            with inference_mode():"
    loop_patch = "            current_episodes_info = self.envs.current_episodes()\n            evidence_recorder = get_evidence_recorder()\n            for evidence_i, evidence_episode in enumerate(current_episodes_info):\n                evidence_recorder.record_observation(evidence_episode, observations[evidence_i])\n\n            with inference_mode():"
    if source.count(loop_anchor) != 1:
        raise RuntimeError("pinned VLFM evaluation loop changed")
    source = source.replace(loop_anchor, loop_patch, 1)
    info_anchor = "            for i in range(len(policy_infos)):\n                infos[i].update(policy_infos[i])\n            batch = batch_obs(  # type: ignore"
    info_patch = "            for i in range(len(policy_infos)):\n                infos[i].update(policy_infos[i])\n                evidence_recorder.record_transition(step_data[i], rewards_l[i], dones[i], infos[i], policy_infos[i])\n            batch = batch_obs(  # type: ignore"
    if source.count(info_anchor) != 1:
        raise RuntimeError("pinned VLFM policy info site changed")
    source = source.replace(info_anchor, info_patch, 1)
    finish_anchor = "                    except Exception:\n                        failure_cause = \"Unknown\"\n\n                    if len(self.config.habitat_baselines.eval.video_option) > 0:"
    finish_patch = "                    except Exception:\n                        failure_cause = \"Unknown\"\n\n                    evidence_recorder.finish_episode(episode_stats, failure_cause)\n\n                    if len(self.config.habitat_baselines.eval.video_option) > 0:"
    if source.count(finish_anchor) != 1:
        raise RuntimeError("pinned VLFM episode finish site changed")
    source = source.replace(finish_anchor, finish_patch, 1)
    if room_online:
        source = source.replace(import_anchor, import_anchor + 'from room_online_observer import get_online_rooms\n', 1)
        action_anchor = '            outputs = self.envs.step(step_data)\n'
        if source.count(action_anchor) != 1:
            raise RuntimeError('Pinned VLFM step site changed')
        source = source.replace(action_anchor,
                                '            if self.envs.num_envs != 1:\n'
                                '                raise ValueError("Online room comparison requires one environment")\n'
                                '            get_online_rooms(self.config).observe(self.envs, current_episodes_info[0], observations[0], step_data[0])\n'
                                + action_anchor, 1)
        source = source.replace(action_anchor, action_anchor + '            get_online_rooms(self.config).transition(outputs[0][2])\n', 1)
        anchor = '                    evidence_recorder.finish_episode(episode_stats, failure_cause)\n'
        source = source.replace(anchor, anchor + '                    get_online_rooms(self.config).finish(episode_stats, failure_cause)\n', 1)
    trainer_path.write_text(source, encoding="utf-8")

    # The upstream video renderer can round an edge point to exactly the map
    # size. Guard only this display operation; policy state and actions are
    # already computed before the renderer is called.
    visualizer_path = package_dst / "utils" / "habitat_visualizer.py"
    visualizer = visualizer_path.read_text(encoding="utf-8")
    bounds_anchor = '''    new_map = infos[0]["top_down_map"]["map"].copy()
    new_map[grid_xy[:, 0], grid_xy[:, 1]] = MAP_TARGET_POINT_INDICATOR
'''
    bounds_patch = '''    new_map = infos[0]["top_down_map"]["map"].copy()
    valid_grid_xy = grid_xy[
        (grid_xy[:, 0] >= 0)
        & (grid_xy[:, 0] < new_map.shape[0])
        & (grid_xy[:, 1] >= 0)
        & (grid_xy[:, 1] < new_map.shape[1])
    ]
    new_map[valid_grid_xy[:, 0], valid_grid_xy[:, 1]] = MAP_TARGET_POINT_INDICATOR
'''
    if visualizer.count(bounds_anchor) != 1:
        raise RuntimeError("pinned VLFM point-cloud visualization site changed")
    visualizer_path.write_text(visualizer.replace(bounds_anchor, bounds_patch, 1), encoding="utf-8")
    return overlay
