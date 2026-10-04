"""Verify all replay depth frames against the recorded normalized inputs."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay-dir', required=True)
    args = parser.parse_args()
    import numpy as np
    root = Path(args.replay_dir)
    manifest = json.loads((root / 'manifest.json').read_text())
    cameras = next(iter(manifest['simulator_config']['agents'].values()))['sim_sensors']
    sensor = cameras['depth_sensor']
    results = []
    for index, case in enumerate(manifest['cases']):
        errors = []
        for step in range(case['frames']):
            raw = np.load(root / f'{index:02d}' / f'{step:04d}.npz')['depth']
            expected = np.load(Path(case['evidence_dir']) / 'depth' / f'{step:04d}.npy').squeeze()
            actual = np.clip(raw, sensor['min_depth'], sensor['max_depth'])
            if sensor['normalize_depth']:
                actual = (actual - sensor['min_depth']) / (sensor['max_depth'] - sensor['min_depth'])
            error = float(np.max(np.abs(actual - expected)))
            if error > 1e-6:
                raise ValueError(f'Depth mismatch: case {index}, step {step}, error {error}')
            errors.append(error)
        floor = case['episode']['start_position'][1]
        delta = max(abs(p['position'][1] - floor) for p in case['poses'])
        results.append(dict(scene_id=case['episode']['scene_id'], frames=case['frames'],
                            depth_max_abs_error=max(errors), max_floor_displacement_m=delta))
    output = root / 'depth_correspondence.json'
    with output.open('x') as stream:
        json.dump(results, stream, indent=2, allow_nan=False)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
