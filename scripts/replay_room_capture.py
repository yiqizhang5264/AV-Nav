"""Replay captured baseline actions into raw metric RGB-D and world poses."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--gpu', default='3')
    p.add_argument('--limit-frames', type=int)
    p.add_argument('--case-index', type=int, help='Single selected case for server smoke')
    args = p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import cv2
    import habitat_sim
    import numpy as np
    import quaternion
    import yaml
    capture, output = Path(args.capture_dir), Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((capture / 'manifest.json').read_text())
    config = yaml.safe_load((capture / 'hydra/.hydra/config.yaml').read_text())
    hc = config['habitat']['simulator']
    assert len(hc['agents']) == 1
    agent_config = next(iter(hc['agents'].values()))
    cameras = agent_config['sim_sensors']
    rgb, depth = cameras['rgb_sensor'], cameras['depth_sensor']
    sample = json.loads(gzip.decompress((capture / 'hm3dv1_selected_episodes.json.gz').read_bytes()))
    episodes = sample['episodes']
    summary = dict(capture_manifest=manifest, simulator_config=hc, cases=[],
                   frame_convention='Habitat world x-right,y-up,z-back; sensor quaternion stored wxyz',
                   diagnostic=args.limit_frames is not None)
    for case_index, episode in enumerate(episodes):
        if args.case_index is not None and case_index != args.case_index:
            continue
        candidates = []
        for meta_path in sorted((capture / 'evidence').rglob('episode.json')):
            folder = meta_path.parent
            if not (folder / 'result.json').exists():
                continue
            meta = json.loads((folder / 'episode.json').read_text())
            if (meta['scene_id'].endswith(episode['scene_id'])
                    and np.allclose(meta['start_position'], episode['start_position'], atol=1e-6, rtol=0)
                    and np.allclose(meta['start_rotation'], episode['start_rotation'], atol=1e-6, rtol=0)
                    and meta['object_category'] == episode['object_category']):
                candidates.append(folder)
        if len(candidates) != 1:
            raise ValueError(f'Expected exactly one finished matching trajectory: {episode["scene_id"]}')
        evidence = candidates[0]
        rows = [json.loads(line) for line in (evidence / 'steps.jsonl').read_text().splitlines()]
        assert [r['step'] for r in rows] == list(range(len(rows)))
        scene = Path(hc['scene_dataset'])
        if str(scene) != 'default' and not scene.is_absolute():
            scene = Path('/home/zyq/AV-Nav-worktrees/vlfm-original-1224d7e/external/vlfm') / scene
        cfg = habitat_sim.SimulatorConfiguration()
        cfg.scene_id = str(Path('/home/zyq/vlfm/data/scene_datasets') / episode['scene_id'])
        cfg.scene_dataset_config_file = str(scene)
        cfg.gpu_device_id = 0
        cfg.enable_physics = False
        cfg.allow_sliding = hc['habitat_sim_v0']['allow_sliding']
        specs = []
        for name, sensor_cfg, kind in [('rgb', rgb, habitat_sim.SensorType.COLOR), ('depth', depth, habitat_sim.SensorType.DEPTH)]:
            spec = habitat_sim.CameraSensorSpec()
            spec.uuid, spec.sensor_type = name, kind
            spec.resolution = [sensor_cfg['height'], sensor_cfg['width']]
            spec.position = sensor_cfg['position']
            spec.hfov = sensor_cfg['hfov']
            specs.append(spec)
        ac = habitat_sim.agent.AgentConfiguration()
        ac.height, ac.radius = agent_config['height'], agent_config['radius']
        ac.sensor_specifications = specs
        names = {1: 'move_forward', 2: 'turn_left', 3: 'turn_right', 4: 'look_up', 5: 'look_down'}
        ac.action_space = {key: habitat_sim.agent.ActionSpec(name, habitat_sim.agent.ActuationSpec(
            amount=hc['forward_step_size'] if key == 1 else hc['turn_angle'])) for key, name in names.items()}
        sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [ac]))
        state = habitat_sim.AgentState()
        state.position = episode['start_position']
        q = episode['start_rotation']
        state.rotation = quaternion.quaternion(q[3], q[0], q[1], q[2])
        agent = sim.initialize_agent(0, state)
        case_dir = output / f'{case_index:02d}'
        case_dir.mkdir()
        count = min(len(rows), args.limit_frames) if args.limit_frames else len(rows)
        poses, rgb_errors = [], []
        for step in range(count):
            if step:
                previous = rows[step - 1]['action']
                while isinstance(previous, list):
                    previous = previous[0]
                if isinstance(previous, dict):
                    previous = previous['action']
                if int(previous) == 0:
                    raise ValueError('Unexpected frames after STOP')
                sim.step(int(previous))
            obs = sim.get_sensor_observations()
            current = agent.get_state()
            camera = current.sensor_states['depth']
            color = obs['rgb'][:, :, :3]
            reference = cv2.cvtColor(cv2.imread(str(evidence / 'rgb' / f'{step:04d}.png')), cv2.COLOR_BGR2RGB)
            error = float(np.abs(color.astype(float) - reference.astype(float)).mean())
            rgb_errors.append(error)
            if error > 0.05:
                raise ValueError(f'Replay RGB differs from captured input: step {step}, MAE {error}')
            np.savez_compressed(case_dir / f'{step:04d}.npz', rgb=color, depth=obs['depth'].astype(np.float32),
                                agent_position=current.position, sensor_position=camera.position,
                                sensor_rotation=quaternion.as_float_array(camera.rotation),
                                agent_rotation=quaternion.as_float_array(current.rotation))
            poses.append(dict(step=step, position=current.position.tolist()))
        sim.close()
        record = dict(selection=manifest['selection']['cases'][case_index], episode=episode, frames=count,
                      evidence_dir=str(evidence), steps_sha256=hashlib.sha256((evidence / 'steps.jsonl').read_bytes()).hexdigest(),
                      rgb_mae_max=max(rgb_errors), poses=poses, hfov=depth['hfov'], sensor_height=depth['position'][1],
                      depth_units='meters', result=json.loads((evidence / 'result.json').read_text()))
        (case_dir / 'manifest.json').write_text(json.dumps(record, indent=2))
        summary['cases'].append(record)
        (output / 'manifest.json').write_text(json.dumps(summary, indent=2))
        print('REPLAY',case_index,episode['scene_id'],count,'RGB MAE',max(rgb_errors),flush=True)
    (output / 'exit.json').write_text(json.dumps(dict(returncode=0, completed=len(summary['cases']))))


if __name__ == '__main__':
    main()
