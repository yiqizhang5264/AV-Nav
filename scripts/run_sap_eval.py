"""Immutable HM3Dv1 category evaluation with exact dataset and episode identity."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def episode_keys(rows, scenes_dir=None):
    """Canonical source identity; Habitat expands scene paths to absolute paths."""
    result = []
    for row in rows:
        scene = Path(row['scene_id'])
        if scene.is_absolute():
            if scenes_dir is None:
                raise ValueError('Absolute scene requires the configured scene root')
            scene = scene.relative_to(Path(scenes_dir).resolve())
        result.append((scene.as_posix(), str(row.get('source_uid', row['episode_id']))))
    return result


def read_dataset(root, split, limit=None, scene=None):
    main = root/split/(split+'.json.gz')
    with gzip.open(main, 'rt') as stream:
        merged = json.load(stream)
    sources = [main]
    episodes = list(merged.get('episodes', []))
    def identities(path, rows):
        source_hash = digest(path)
        return [dict(scene_id=e['scene_id'], episode_id=str(e['episode_id']),
                     source_uid=source_hash+':'+str(i), source_file=path.name, source_row=i,
                     source_sha256=source_hash,
                     episode_sha256=hashlib.sha256(json.dumps(e, sort_keys=True).encode()).hexdigest())
                for i,e in enumerate(rows)]
    identity = identities(main, episodes)
    if scene:
        selected = [i for i,e in enumerate(episodes) if scene in Path(e['scene_id']).parts[-1]]
        episodes = [episodes[i] for i in selected]
        identity = [identity[i] for i in selected]
    goals = dict(merged.get('goals_by_category', {}))
    for path in sorted((root/split/'content').glob('*.json.gz')):
        if scene and path.name != scene+'.json.gz':
            continue
        if limit and len(episodes) >= limit:
            break
        with gzip.open(path, 'rt') as stream:
            data = json.load(stream)
        episodes.extend(data['episodes'])
        identity.extend(identities(path, data['episodes']))
        goals.update(data.get('goals_by_category', {}))
        sources.append(path)
    if limit:
        episodes = episodes[:limit]
        identity = identity[:limit]
    keys = [e['source_uid'] for e in identity]
    if not keys or len(keys) != len(set(keys)):
        raise ValueError('Empty or duplicate source episode identities')
    merged.update(episodes=episodes, goals_by_category=goals,
                  sap_source_identities=identity,
                  content_scenes_path='__no_external_content__/{scene}.json.gz')
    return merged, {str(p): digest(p) for p in sources}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(ROOT/'configs/sap_category.json'))
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--scenes-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--split', choices=['train', 'val'], default='val')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--scene', help='Exact content-file stem, for scene sharding')
    parser.add_argument('--max-steps', type=int, default=500)
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--variant', choices=['baseline', 'sap'], default='sap')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('limit must be positive')
    if args.max_steps < 1:
        parser.error('max-steps must be positive')
    root = Path(args.dataset_root).resolve()
    if root.name != 'v1' or root.parent.name != 'hm3d':
        parser.error('Expected the HM3Dv1 ObjectNav dataset root .../hm3d/v1')
    config = json.loads(Path(args.config).read_text())
    data, hashes = read_dataset(root, args.split, args.limit, args.scene)
    categories = set(e['object_category'] for e in data['episodes'])
    allowed = {'chair', 'bed', 'plant', 'toilet', 'tv_monitor', 'sofa'}
    if not categories <= allowed:
        parser.error('Unexpected ObjectNav categories: '+str(categories))
    run = Path(args.output).resolve()
    run.mkdir(parents=True, exist_ok=False)
    dataset = run/'hm3dv1_episodes.json.gz'
    with gzip.open(dataset, 'wt') as stream:
        json.dump(data, stream)
    frozen_config = run/'sap_config.json'
    frozen_config.write_text(json.dumps(config, indent=2))
    upstream = ROOT/'external/vlfm'
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, AV_RUN_DIR=str(run), AV_DATASET_FILE=str(dataset),
               SAP_CONFIG=str(frozen_config), SAP_VARIANT=args.variant, PYTHONHASHSEED=str(config['seed']),
               PYTHONPATH=os.pathsep.join([str(ROOT), str(upstream), env.get('PYTHONPATH', '')]))
    command = [sys.executable, '-u', str(ROOT/'scripts/run_sap_runtime.py'),
        'habitat_baselines.evaluate=true', 'habitat_baselines.num_environments=1',
        'habitat_baselines.torch_gpu_id=0', 'habitat_baselines.test_episode_count=-1',
        'habitat_baselines.eval.video_option=[]', f'habitat_baselines.eval.split={args.split}',
        f'habitat.seed={config["seed"]}', f'habitat.environment.max_episode_steps={args.max_steps}',
        'habitat.environment.iterator_options.shuffle=False',
        'habitat.environment.iterator_options.group_by_scene=False',
        f'habitat.dataset.data_path={dataset}', f'habitat.dataset.scenes_dir={Path(args.scenes_dir).resolve()}',
        f'habitat_baselines.tensorboard_dir={run/"tb"}', f'hydra.run.dir={run/"hydra"}']
    if args.variant == 'sap':
        command.append('habitat_baselines.rl.policy.name=SAPCategoryPolicy')
    manifest = dict(started=datetime.now(timezone.utc).isoformat(), benchmark='HM3Dv1 ObjectNav',
        split=args.split, scene=args.scene, variant=args.variant, expected_episodes=len(data['episodes']),
        diagnostic=bool(config.get('diagnostic')) or args.split != 'val' or args.limit is not None or args.max_steps != 500,
        config=config, source_sha256=hashes, dataset_sha256=digest(dataset), command=command,
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        upstream_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=upstream, text=True).strip(),
        dirty=subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True),
        services={k: env.get(k, str(v)) for k,v in dict(GROUNDING_DINO_PORT=12181,
            BLIP2ITM_PORT=12182, SAM_PORT=12183, YOLOV7_PORT=12184).items()})
    (run/'manifest.json').write_text(json.dumps(manifest, indent=2))
    freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True)
    (run/'environment.txt').write_text(freeze.stdout)
    with (run/'console.log').open('w') as stream:
        result = subprocess.run(command, cwd=upstream, env=env, stdout=stream, stderr=subprocess.STDOUT)
    rows = []
    if (run/'episodes.jsonl').exists():
        rows = [json.loads(line) for line in (run/'episodes.jsonl').read_text().splitlines() if line]
    expected = set(episode_keys(data['sap_source_identities'], args.scenes_dir))
    actual = set(episode_keys(rows, args.scenes_dir))
    complete = result.returncode == 0 and actual == expected and len(rows) == len(expected)
    summary = dict(complete=complete, returncode=result.returncode, expected=len(expected),
                   completed=len(rows), diagnostic=manifest['diagnostic'])
    if complete:
        summary['metrics'] = {key: sum(float(r['metrics'][key]) for r in rows)/len(rows)
                              for key in ('success', 'spl', 'soft_spl') if all(key in r['metrics'] for r in rows)}
    (run/'summary.json').write_text(json.dumps(summary, indent=2))
    raise SystemExit(0 if complete else 2)


if __name__ == '__main__':
    main()
