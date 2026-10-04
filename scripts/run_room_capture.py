"""Record pinned VLFM trajectories for a selected room-comparison sample."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--vlfm-root', required=True)
    parser.add_argument('--gpu', default='3')
    parser.add_argument('--max-steps', type=int, default=500)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    sample = Path(args.sample_dir).resolve()
    output = Path(args.output_dir).resolve()
    vlfm = Path(args.vlfm_root).resolve()
    selection = json.loads((sample / 'selection.json').read_text())
    split = Path(selection['source']).name
    if split not in {'train', 'val'}:
        raise ValueError('Unsupported dataset split')
    if hashlib.sha256((sample / 'episodes.json.gz').read_bytes()).hexdigest() != selection['sha256']:
        raise ValueError('Sample hash mismatch')
    pin = subprocess.check_output(['git', '-C', str(vlfm), 'rev-parse', 'HEAD'], text=True).strip()
    if pin != '584ed56008754fde7997d904983607def8328322':
        raise ValueError('VLFM must match the fixed baseline commit')
    if subprocess.check_output(['git', '-C', str(vlfm), 'diff', 'HEAD', '--', 'vlfm'], text=True):
        raise ValueError('Tracked VLFM code modifications detected')
    output.mkdir(parents=True, exist_ok=False)
    dataset = output / 'hm3dv1_selected_episodes.json.gz'
    dataset.symlink_to(sample / 'episodes.json.gz')
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, VLFM_EVIDENCE_DIR=str(output / 'evidence'),
               ZSOS_LOG_DIR=str(output / 'episode_logs'), ZSOS_DONE_PATH=str(output / 'DONE'),
               PYTHONPATH=os.pathsep.join([str(repo / 'scripts'), str(vlfm), env.get('PYTHONPATH', '')]))
    command = [sys.executable, '-u', str(repo / 'scripts/run_vlfm_evidence.py'), '--vlfm-root', str(vlfm),
               'habitat_baselines.evaluate=true', 'habitat_baselines.eval.video_option=[]',
               f'habitat_baselines.test_episode_count={selection["count"]}',
               'habitat_baselines.num_environments=1', 'habitat_baselines.torch_gpu_id=0',
               f'habitat_baselines.eval.split={split}', f'habitat.dataset.split={split}',
               'habitat.environment.iterator_options.shuffle=False',
               f'habitat.environment.max_episode_steps={args.max_steps}',
               f'habitat.dataset.data_path={dataset}',
               'habitat.dataset.scenes_dir=/home/zyq/vlfm/data/scene_datasets',
               f'habitat_baselines.tensorboard_dir={output / "tb"}', f'hydra.run.dir={output / "hydra"}']
    manifest = dict(command=command, av_nav_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
                    vlfm_commit=pin, selection=selection, gpu=args.gpu, max_steps=args.max_steps,
                    purpose='Shared trajectory input for room segmentation; not autonomous exploration comparison',
                    runtime_id_note='Habitat renumbers loaded episodes; match scene/start pose/category and source row, not runtime ID.')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    (output / 'environment.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    with (output / 'console.log').open('w') as log:
        result = subprocess.run(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
    (output / 'exit.json').write_text(json.dumps(dict(returncode=result.returncode)))
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
