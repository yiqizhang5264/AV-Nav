"""Audit completed online evaluations and render their final room maps."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', required=True)
    p.add_argument('--resource-audit', required=True)
    p.add_argument('--output-dir', required=True)
    args = p.parse_args()
    import cv2
    import numpy as np
    import yaml
    run, output = Path(args.run_root), Path(args.output_dir)
    audit = json.loads(Path(args.resource_audit).read_text())
    sources, events, exits, hashes, native = [], [], [], {}, []
    for i in range(5):
        job = run / f'online_full5/case_{i:02d}'
        case = job / 'inputs/00'
        source = json.loads((case / 'manifest.json').read_text())
        if source['selection'] != audit['selection']['cases'][i]:
            raise ValueError('Online sample differs from fixed validation selection')
        sources.append(source)
        capture = run / f'online_capture_{i:02d}'
        capture_exit = json.loads((capture / 'exit.json').read_text())
        if capture_exit['returncode'] != 0:
            raise ValueError('Online evaluation failed')
        evidence = list((capture / 'evidence').rglob('result.json'))
        if len(evidence) != 1:
            raise ValueError('Expected exactly one completed evidence episode')
        evidence = evidence[0].parent
        sensor = next(iter(yaml.safe_load((capture / 'hydra/.hydra/config.yaml').read_text())['habitat']['simulator']['agents'].values()))['sim_sensors']['depth_sensor']
        rows = [json.loads(row) for row in (case / 'online_events.jsonl').read_text().splitlines()]
        if len(rows) != source['frames'] or [r['frame'] for r in rows] != list(range(source['frames'])):
            raise ValueError('Incomplete online event sequence')
        method_summaries = []
        for name in ['active', 'occusg']:
            method = job / name / '00'
            state = json.loads((method / 'exit.json').read_text())
            if state['returncode'] != 0:
                raise ValueError('Online segmentation failed')
            summary = json.loads((method / 'summary.json').read_text())
            if summary['source'] != source or summary['frames'] != source['frames'] or not summary['adapter']['online_stream']:
                raise ValueError('Method was not an exact completed online pair')
            expected_sha = sha(Path(__file__).resolve().parent / f'run_{name if name == "active" else "occusg"}_room_replay.py')
            if summary['adapter']['source_sha256'] != expected_sha:
                raise ValueError('Collector and runtime method adapter differ')
            method_summaries.append(dict(method=name, exit=state))
            summary.pop('records')
            summary['source'] = {k:v for k,v in source.items() if k != 'poses'}
            native.append(dict(method=name, case=i, **summary))
        hashes[f'{i:02d}'] = {}
        compact, previous = [], 0
        rgb_max, depth_max = 0., 0.
        for row in rows:
            index = row['frame']
            transition = json.loads((case / f'transition_{index:04d}.json').read_text())
            if not previous < row['published_ns'] < row['both_methods_completed_ns'] < transition['env_step_finished_ns']:
                raise ValueError('Online processing did not precede navigation step')
            if len(row['methods']) != 2 or any(a['frame'] != index or not row['published_ns'] < a['completed_ns'] <= row['both_methods_completed_ns'] for a in row['methods']):
                raise ValueError('Missing exact-frame method acknowledgment')
            terminal = index == source['frames'] - 1
            if row['last'] != terminal or transition['done'] != terminal:
                raise ValueError('Online terminal marker mismatch')
            previous = transition['env_step_finished_ns']
            frame = case / f'{index:04d}.npz'
            hashes[f'{i:02d}'][frame.name] = sha(frame)
            with np.load(frame) as data:
                ref_rgb = cv2.cvtColor(cv2.imread(str(evidence / 'rgb' / f'{index:04d}.png')), cv2.COLOR_BGR2RGB)
                rgb_error = float(np.abs(data['rgb'].astype(float) - ref_rgb.astype(float)).max())
                expected = np.clip(data['depth'],sensor['min_depth'],sensor['max_depth'])
                if sensor['normalize_depth']:
                    expected = (expected-sensor['min_depth'])/(sensor['max_depth']-sensor['min_depth'])
                depth_error = float(np.abs(expected-np.load(evidence/'depth'/f'{index:04d}.npy').squeeze()).max())
            if not np.isfinite(depth_error) or rgb_error != 0 or depth_error > 1e-6:
                raise ValueError('Online input differs from current policy observation')
            rgb_max,depth_max=max(rgb_max,rgb_error),max(depth_max,depth_error)
            compact.append(dict(frame=index,published_ns=row['published_ns'],methods_completed_ns=row['both_methods_completed_ns'],
                                env_step_finished_ns=transition['env_step_finished_ns'],action=row['action'],
                                active_regions=row['methods'][0]['record']['room_count'],
                                occusg_regions=row['methods'][1]['record']['region_count']))
        events.append(dict(case=i,scene=source['episode']['scene_id'],frames=len(rows),
                           rgb_max_abs_error=rgb_max,normalized_depth_max_abs_error=depth_max,steps=compact))
        exits.append(dict(case=i,capture=capture_exit,methods=method_summaries))
    stage = output.with_name(output.name + '_stage')
    stage.mkdir(parents=True, exist_ok=False)
    for name in ['inputs', 'active', 'occusg']:
        (stage/name).mkdir()
        for i in range(5):
            (stage/name/f'{i:02d}').symlink_to((run/f'online_full5/case_{i:02d}'/name/'00').resolve())
    (stage/'inputs/manifest.json').write_text(json.dumps(dict(cases=sources,online=True),indent=2))
    subprocess.run([sys.executable,str(Path(__file__).with_name('collect_room_comparison.py')),
                    '--replay-dir',str(stage/'inputs'),'--active-dir',str(stage/'active'),
                    '--occusg-dir',str(stage/'occusg'),'--output-dir',str(output)],check=True)
    provenance = output/'provenance'
    provenance.mkdir()
    for name, value in [('online_step_audit.json',events),('run_exits.json',exits),('input_frame_sha256.json',hashes),('method_summaries.json',native)]:
        (provenance/name).write_text(json.dumps(value,indent=2,allow_nan=False))
    def finite(value):
        if isinstance(value,float) and not np.isfinite(value):return None
        if isinstance(value,dict):return {k:finite(v) for k,v in value.items()}
        if isinstance(value,list):return [finite(v) for v in value]
        return value
    (provenance/'resource_audit.json').write_text(json.dumps(finite(audit),indent=2,allow_nan=False))
    for i in range(5):
        dest=provenance/f'case_{i:02d}';dest.mkdir()
        capture=run/f'online_capture_{i:02d}'
        shutil.copy2(capture/'manifest.json',dest/'capture_manifest.json')
        shutil.copy2(capture/'hydra/.hydra/config.yaml',dest/'capture_config.yaml')
        shutil.copy2(run/f'settings_{i:02d}.json',dest/'online_settings.json')
        shutil.copy2(run/f'online_full5/case_{i:02d}/inputs/00/worker_commands.json',dest/'worker_commands.json')
        for filename in ['pipeline_params.yaml','dude_params.yaml','commands.json']:
            shutil.copy2(run/f'online_full5/case_{i:02d}/occusg/00'/filename,dest/filename)
    repo=Path(__file__).resolve().parents[1]
    identity=dict(collector_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
                  evaluation_commits=[json.loads((provenance/f'case_{i:02d}/capture_manifest.json').read_text())['av_nav_commit'] for i in range(5)],
                  runtime_scripts_sha256={p.name:sha(p) for p in (repo/'scripts').glob('*room*.py')},
                  scope='Online incremental segmentation during VLFM evaluation; no segmentation action feedback; five independently seeded processes',
                  run_root=str(run),shared_frames=sum(s['frames'] for s in sources),scene_assets='HM3D-0.2 val; ObjectNav episode split HM3D v1 val')
    (provenance/'run_identity.json').write_text(json.dumps(identity,indent=2))
    print('ONLINE_FINAL_AUDIT_COMPLETE',identity['shared_frames'],flush=True)


if __name__ == '__main__':
    main()
