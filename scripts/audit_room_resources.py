"""Server-only Habitat resource/region audit for an immutable room sample."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample-dir', required=True)
    parser.add_argument('--scenes-dir', default='/home/zyq/vlfm/data/scene_datasets')
    parser.add_argument('--scene-config', default='/home/zyq/vlfm/data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    import habitat_sim
    sample = Path(args.sample_dir)
    selection = json.loads((sample / 'selection.json').read_text())
    assert sha(sample / 'episodes.json.gz') == selection['sha256']
    dataset = json.loads(gzip.decompress((sample / 'episodes.json.gz').read_bytes()))
    summary = dict(python=sys.executable, habitat_sim=habitat_sim.__version__,
                   av_nav_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                   selection=selection, scene_config=str(Path(args.scene_config).resolve()),
                   scene_config_sha256=sha(args.scene_config), cases=[],
                   ground_truth_note='Region IDs/AABBs alone are not precise room-footprint ground truth.')
    (output / 'environment.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    for episode, case in zip(dataset['episodes'], selection['cases']):
        scene = Path(args.scenes_dir) / episode['scene_id']
        record = dict(case, scene_path=str(scene), resolved_scene_path=str(scene.resolve()),
                      scene_files={str(p.resolve()): sha(p) for p in sorted(scene.parent.iterdir()) if p.is_file()})
        cfg = habitat_sim.SimulatorConfiguration()
        cfg.scene_id = str(scene)
        cfg.scene_dataset_config_file = args.scene_config
        cfg.create_renderer = False
        cfg.enable_physics = False
        sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [habitat_sim.agent.AgentConfiguration()]))
        semantic = sim.semantic_scene
        record['semantic_object_count'] = len([o for o in semantic.objects if o is not None])
        record['regions'] = [dict(id=str(r.id), category=r.category.name() if r.category is not None else None,
                                  object_count=len(r.objects), aabb_center=r.aabb.center.tolist(),
                                  aabb_sizes=r.aabb.sizes.tolist())
                             for r in semantic.regions if r is not None]
        record['navmesh_loaded'] = sim.pathfinder.is_loaded
        record['start_navigable'] = sim.pathfinder.is_navigable(episode['start_position'])
        sim.close()
        summary['cases'].append(record)
        (output / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(episode['scene_id'], 'objects', record['semantic_object_count'], 'regions', len(record['regions']),
              'start_navigable', record['start_navigable'], flush=True)
    (output / 'exit.json').write_text(json.dumps(dict(returncode=0, completed=len(summary['cases']))))


if __name__ == '__main__':
    main()
