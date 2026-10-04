"""Offline shared-trajectory adapter retaining upstream Active room components."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import types


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream', required=True)
    p.add_argument('--detr-source', required=True)
    p.add_argument('--weights', required=True)
    p.add_argument('--case-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--update-every', type=int, default=10)
    args = p.parse_args()
    os.environ['MPLBACKEND'] = 'Agg'
    import cv2
    import numpy as np
    import quaternion
    import torch
    import torchvision
    import igraph
    from matplotlib import pyplot as plt
    plt.show = lambda *a, **k: None
    plt.pause = lambda *a, **k: None
    igraph.plot = lambda *a, **k: None
    torch.set_num_threads(4)
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    from av_nav.room_geometry import active_map_pose
    upstream, output, case = Path(args.upstream), Path(args.output_dir), Path(args.case_dir)
    pin = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if pin != 'a941e3a5a1c8e16920a158fa0a9198c95be9c978':
        raise ValueError('Unexpected Active upstream commit')
    detr_pin = subprocess.check_output(['git', '-C', args.detr_source, 'rev-parse', 'HEAD'], text=True).strip()
    if detr_pin != '29901c51d7fe8712168b8d0d64351170bc0f83e0':
        raise ValueError('Unexpected DETR architecture source commit')
    weight_sha = hashlib.sha256(Path(args.weights).read_bytes()).hexdigest()
    if weight_sha != 'd971e3b760421eb29665c1ca986854ce8ec57ddcc50209fcc77125c5e3cef7ec':
        raise ValueError('Door weights differ from official description=4 model')
    output.mkdir(parents=True, exist_ok=False)
    overlay = output / 'active_overlay'
    overlay.mkdir()
    for name in ['arguments.py', 'frontier_detection.py', 'door_detection.py', 'topomap_construction.py']:
        shutil.copy2(upstream / name, overlay / name)
    shutil.copytree(upstream / 'detr_door_detection', overlay / 'detr_door_detection',
                    ignore=shutil.ignore_patterns('train_params', '__pycache__'))
    (overlay / 'env/habitat').mkdir(parents=True)
    shutil.copytree(upstream / 'env/utils', overlay / 'env/utils')
    shutil.copy2(upstream / 'env/habitat/hough_door_detection.py', overlay / 'env/habitat/hough_door_detection.py')
    weight_dir = overlay / 'detr_door_detection/train_params/detr_resnet50_4/final_doors_dataset'
    weight_dir.mkdir(parents=True)
    (weight_dir / 'model.pth').symlink_to(Path(args.weights).resolve())
    for name in ['env', 'env.utils', 'env.habitat']:
        namespace = types.ModuleType(name)
        namespace.__path__ = [str(overlay / name.replace('.', '/'))]
        sys.modules[name] = namespace
    sys.path.insert(0, str(overlay))
    sys.argv = [sys.argv[0], '--no_cuda', '--auto_gpu_config', '0', '--num_mini_batch', '1']
    original_hub_load = torch.hub.load
    def load_full_architecture(repository, model, *unused, **kwargs):
        if repository != 'facebookresearch/detr':
            raise ValueError('Unexpected torch.hub dependency')
        # Full official door state_dict replaces every model parameter. Avoid
        # downloading redundant generic COCO/ResNet pretrained parameters.
        original_resnet = torchvision.models.resnet50
        def no_redundant_weights(*a, **k):
            k['pretrained'] = False
            return original_resnet(*a, **k)
        torchvision.models.resnet50 = no_redundant_weights
        try:
            return original_hub_load(args.detr_source, model, source='local', pretrained=False)
        finally:
            torchvision.models.resnet50 = original_resnet
    torch.hub.load = load_full_architecture
    from env.utils.map_builder import MapBuilder
    from env.habitat.hough_door_detection import convert_2_laser
    from detr_door_detection.run_detr import run_detr
    from door_detection import Door_detection
    from frontier_detection import Frontier_detection
    from topomap_construction import Topomap_construction
    source = json.loads((case / 'manifest.json').read_text())
    frames = sorted(case.glob('[0-9][0-9][0-9][0-9].npz'))
    first = np.load(frames[0])
    h, w = first['depth'].shape
    mapper = MapBuilder(dict(frame_width=w, frame_height=h, fov=source['hfov'], resolution=5,
                             map_size_cm=4800, agent_min_z=25, agent_max_z=150, agent_height=source['sensor_height'] * 100,
                             agent_view_angle=0, du_scale=2, vision_range=60, visualize=0, obs_threshold=1))
    start = np.asarray(source['episode']['start_position'])
    frontier, topo = Frontier_detection(960), Topomap_construction()
    detector = Door_detection([480, 480], [480, 480])
    detected, records, trigger_frames = [], [], 0
    last_bot = []
    lmb = np.array([0, 960, 0, 960])
    labels = np.zeros((960, 960), np.int32)
    for index, frame in enumerate(frames):
        began = time.perf_counter()
        data = np.load(frame)
        q = quaternion.from_float_array(data['agent_rotation'])
        pose = active_map_pose(data['agent_position'], quaternion.as_rotation_matrix(q), start)
        if np.any(pose[:2] < 2) or np.any(pose[:2] > 46):
            raise ValueError('Trajectory exceeds native 48m map bounds')
        if abs(float(data['agent_position'][1] - start[1])) > .3:
            raise ValueError('Multi-floor trajectory requires separate floor maps')
        square = cv2.resize(data['rgb'], (256, 256)).astype(np.float32) / 255
        with torch.inference_mode():
            mask, _, full = run_detr(square)
        # The official full mask is a fixed top-image ROI, not a detection.
        trigger_frames += bool(np.any(mask))
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        full = cv2.resize(full, (w, h), interpolation=cv2.INTER_NEAREST)
        depth_cm = data['depth'].copy() * 100
        sensor_r = quaternion.as_rotation_matrix(quaternion.from_float_array(data['sensor_rotation']))
        sensor_forward = sensor_r @ np.array([0., 0., -1.])
        mapper.agent_view_angle = np.degrees(np.arcsin(np.clip(sensor_forward[1], -1, 1)))
        native_pose = (pose[0] * 100, pose[1] * 100, pose[2])
        _, occupied, _, explored, pano, pano_exp = mapper.update_map(depth_cm.copy(), native_pose)
        door_map, _ = mapper.get_door_map(depth_cm * mask, native_pose)
        door_full, _ = mapper.get_door_map_full(depth_cm * full, native_pose)
        last_bot.append((pose[:2] * 20).tolist())
        if (index + 1) % args.update_every == 0 or index == len(frames) - 1:
            # Follow official entry point's transpose/coordinate conventions.
            obs, exp = occupied.T.copy(), explored.T.copy()
            points, lasers = convert_2_laser(pano.T.copy(), pano_exp.T.copy(), pose)
            candidates = [pt for pt in points if door_map.T[pt[1]-2:pt[1]+3, pt[0]-2:pt[0]+3].sum() > 0]
            new, _ = detector.door_filter(door_full.T, obs, exp, (pose[:2] * 20).tolist(), last_bot,
                                         detected, use_12point=False, external_door_point=candidates)
            last_bot = []
            detected.extend(new)
            current = pose[:2] * 20
            _, _, _, room_exp, door_grid = frontier.frontier_detection(current.copy(), current, obs, exp, lmb, detected, lasers)
            detected = topo.same_node_check(room_exp, detected)
            new, removed = topo.check_topomap(new, detected, room_exp, current, obs, exp, lmb, door_grid,
                                             0, Path(source['episode']['scene_id']).stem, lasers)
            topo.add_room(new, [current[1], current[0]], obs, exp, lmb)
            plt.close('all')
            for door in removed:
                detected.remove(door)
            labels.fill(0)
            for vertex in topo.g.vs:
                cells = np.asarray(vertex['room_exp'], dtype=int)
                if cells.size:
                    labels[cells[:, 0], cells[:, 1]] = vertex.index + 1
            mapper.map_copy.fill(0)
            mapper.map_door_copy.fill(0)
            mapper.map_door_copy_full.fill(0)
        duration = time.perf_counter() - began
        records.append(dict(frame=index, seconds=duration, room_count=int(len(np.unique(labels[labels > 0]))),
                            topology_nodes=topo.g.vcount(), detected_doors=len(detected), door_trigger=bool(mask.any())))
        if index + 1 in [100, 250, 500] or index == len(frames) - 1:
            np.savez_compressed(output / f'checkpoint_{index + 1:04d}.npz', labels=labels.T,
                                occupied=occupied, explored=explored, pose=pose)
        if index % 20 == 0:
            print('ACTIVE', index, records[-1], flush=True)
    summary = dict(upstream_commit=pin, detr_commit=detr_pin, door_weights_sha256=weight_sha,
                   source=source, frames=len(frames), door_trigger_frames=trigger_frames,
                   final_room_count=records[-1]['room_count'], final_detected_doors=len(detected),
                   accuracy=None, accuracy_reason='No verified room footprint GT',
                   adapter=dict(update_every=args.update_every, detector_resize=[256,256], pose_source='simulator',
                                grid_resolution_m=.05, map_size_m=48, native_x='24-(world_z-start_z)',
                                native_y='24-(world_x-start_x)', labels_index_order='native_y,native_x',
                                exploration_policy='disabled shared trajectory', upstream_visualization_disabled=True,
                                floor_policy='reject transitions >0.3m'),
                   records=records)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    (output / 'exit.json').write_text(json.dumps(dict(returncode=0)))


if __name__ == '__main__':
    main()
