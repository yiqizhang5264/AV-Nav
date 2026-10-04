"""Sample one episode per distinct scene, preserving Habitat episode identity."""
import argparse
import gzip
import hashlib
import json
import random
from pathlib import Path


def select(source, count, seed, partition):
    source = Path(source)
    if count < 1:
        raise ValueError('Count must be positive')
    if source.name != 'train':
        raise ValueError('Initial room comparison must use the training split')
    scenes = {}
    metadata = None
    for path in sorted((source / 'content').glob('*.json.gz')):
        raw = path.read_bytes()
        data = json.loads(gzip.decompress(raw))
        shared = {k: v for k, v in data.items()
                  if k not in {'episodes', 'goals_by_category', 'content_scenes_path'}}
        if metadata is None:
            metadata = shared
        elif shared != metadata:
            raise ValueError(f'Inconsistent dataset metadata: {path}')
        seen = set()
        for episode in data.get('episodes', []):
            scene = episode['scene_id']
            # Match the existing make_split.py scene partition exactly.
            bucket = int(hashlib.sha256(('scene-v1|' + path.stem).encode()).hexdigest()[:8], 16) % 4
            if (partition == 'calibration' and bucket == 0) or (partition == 'development' and bucket != 0):
                continue
            key = (scene, str(episode['episode_id']))
            if key in seen:
                raise ValueError(f'Duplicate episode identity: {key}')
            seen.add(key)
            record = scenes.setdefault(scene, {'path': path, 'sha256': hashlib.sha256(raw).hexdigest(),
                                               'data': data, 'episodes': []})
            if record['path'] != path:
                raise ValueError(f'Scene appears in multiple source files: {scene}')
            record['episodes'].append(episode)
    if len(scenes) < count:
        raise ValueError(f'Only {len(scenes)} distinct eligible scenes; requested {count}')
    rng = random.Random(seed)
    chosen, goals, cases = [], {}, []
    for scene in sorted(rng.sample(sorted(scenes), count)):
        record = scenes[scene]
        episode = rng.choice(sorted(record['episodes'], key=lambda e: str(e['episode_id'])))
        goal_key = Path(scene).name + '_' + episode['object_category']
        source_goals = record['data'].get('goals_by_category', {})
        if not source_goals.get(goal_key):
            raise ValueError(f'Missing task goals: {goal_key}')
        chosen.append(episode)
        goals[goal_key] = source_goals[goal_key]
        cases.append({'scene_id': scene, 'episode_id': str(episode['episode_id']),
                      'object_category': episode['object_category'], 'source': str(record['path']),
                      'source_sha256': record['sha256']})
    dataset = dict(metadata or {}, episodes=chosen, goals_by_category=goals,
                   content_scenes_path='{data_path}/__no_external_content__/{scene}.json.gz')
    manifest = {'source': str(source), 'seed': seed, 'partition': partition, 'count': count,
                'selection_algorithm': 'uniform-distinct-scenes-then-uniform-episode-v1',
                'dataset_version_claim': 'HM3D v1 episode data; scene asset version must be verified separately',
                'cases': cases}
    return dataset, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output-dir', required=True, help='New immutable directory under runs/')
    parser.add_argument('--count', type=int, default=5)
    parser.add_argument('--seed', type=int, default=20261004)
    parser.add_argument('--partition', choices=['calibration', 'development'], default='calibration')
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        parser.error('Output directory already exists; do not overwrite a sample')
    try:
        dataset, manifest = select(args.source, args.count, args.seed, args.partition)
    except ValueError as exc:
        parser.error(str(exc))
    payload = gzip.compress(json.dumps(dataset, sort_keys=True).encode(), mtime=0)
    manifest['sha256'] = hashlib.sha256(payload).hexdigest()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'episodes.json.gz').write_bytes(payload)
    (output / 'selection.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
