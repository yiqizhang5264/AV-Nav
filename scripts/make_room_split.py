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
    # HM3Dv1 training IDs repeat even within a category. Identity therefore
    # includes the immutable source hash and source row, not just episode_id.
    paths = []
    for path in sorted((source / 'content').glob('*.json.gz')):
        bucket = int(hashlib.sha256(('scene-v1|' + path.stem).encode()).hexdigest()[:8], 16) % 4
        if (partition == 'calibration' and bucket != 0) or (partition == 'development' and bucket == 0):
            paths.append(path)
    if len(paths) < count:
        raise ValueError(f'Only {len(paths)} distinct eligible scenes; requested {count}')
    rng = random.Random(seed)
    chosen, goals, cases = [], {}, []
    scenes = set()
    metadata = None
    # One scene per Habitat content shard; validate that contract on sampled
    # shards and keep only one full shard in memory at a time.
    for path in sorted(rng.sample(paths, count)):
        raw = path.read_bytes()
        data = json.loads(gzip.decompress(raw))
        shared = {k: v for k, v in data.items()
                  if k not in {'episodes', 'goals_by_category', 'content_scenes_path'}}
        if metadata is None:
            metadata = shared
        elif shared != metadata:
            raise ValueError(f'Inconsistent dataset metadata: {path}')
        episodes = data.get('episodes', [])
        shard_scenes = {e['scene_id'] for e in episodes}
        if len(shard_scenes) != 1:
            raise ValueError(f'Expected one nonempty scene per content shard: {path}')
        scene = next(iter(shard_scenes))
        if scene in scenes:
            raise ValueError(f'Scene appears in multiple source files: {scene}')
        scenes.add(scene)
        source_index = rng.randrange(len(episodes))
        episode = episodes[source_index]
        goal_key = Path(scene).name + '_' + episode['object_category']
        source_goals = data.get('goals_by_category', {})
        if not source_goals.get(goal_key):
            raise ValueError(f'Missing task goals: {goal_key}')
        chosen.append(episode)
        goals[goal_key] = source_goals[goal_key]
        cases.append({'scene_id': scene, 'episode_id': str(episode['episode_id']),
                      'object_category': episode['object_category'], 'source': str(path),
                      'source_index': source_index,
                      'episode_sha256': hashlib.sha256(json.dumps(episode, sort_keys=True).encode()).hexdigest(),
                      'source_sha256': hashlib.sha256(raw).hexdigest()})
    dataset = dict(metadata or {}, episodes=chosen, goals_by_category=goals,
                   content_scenes_path='{data_path}/__no_external_content__/{scene}.json.gz')
    manifest = {'source': str(source), 'seed': seed, 'partition': partition, 'count': count,
                'selection_algorithm': 'uniform-scene-shards-then-uniform-source-row-v2',
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
