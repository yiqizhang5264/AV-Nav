"""Scene-sharded full HM3Dv1 val evaluation; resumable only at fixed identity."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from run_sap_eval import ROOT, read_dataset, digest, episode_keys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=str(ROOT/'configs/sap_category.json'))
    p.add_argument('--dataset-root', required=True)
    p.add_argument('--scenes-dir', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--gpu', default='1')
    p.add_argument('--variant', choices=['baseline','sap'], default='sap')
    args = p.parse_args()
    if json.loads(Path(args.config).read_text()).get('diagnostic'):
        p.error('Diagnostic configurations cannot run as full efficacy suites')
    data, hashes = read_dataset(Path(args.dataset_root), 'val')
    expected = set(episode_keys(data['sap_source_identities'], args.scenes_dir))
    scenes = sorted(p.name[:-8] for p in (Path(args.dataset_root)/'val/content').glob('*.json.gz'))
    output = Path(args.output).resolve()
    identity = dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                    config_sha256=digest(Path(args.config)), source_sha256=hashes,
                    scenes_dir=str(Path(args.scenes_dir).resolve()), variant=args.variant)
    manifest = output/'suite.json'
    if output.exists():
        if not manifest.exists() or json.loads(manifest.read_text()) != identity:
            raise SystemExit('Refusing to resume a different suite identity')
    else:
        output.mkdir(parents=True)
        manifest.write_text(json.dumps(identity, indent=2))
    all_rows = []
    for scene in scenes:
        scene_rows = None
        for attempt in range(1,4):
            run = output/scene/f'attempt_{attempt:02}'
            if not run.exists():
                command = [sys.executable,str(ROOT/'scripts/run_sap_eval.py'),
                    '--config',args.config,'--dataset-root',args.dataset_root,'--scenes-dir',args.scenes_dir,
                    '--output',str(run),'--gpu',args.gpu,'--variant',args.variant,'--scene',scene]
                subprocess.run(command, check=False)
            summary = run/'summary.json'
            if summary.exists() and json.loads(summary.read_text()).get('complete'):
                scene_rows = [json.loads(line) for line in (run/'episodes.jsonl').read_text().splitlines()]
                break
        if scene_rows is None:
            raise SystemExit('Scene failed three attempts: '+scene)
        all_rows.extend(scene_rows)
    actual = set(episode_keys(all_rows, args.scenes_dir))
    if actual != expected or len(all_rows) != len(expected):
        raise SystemExit('Full validation episode set mismatch')
    summary = dict(complete=True, benchmark='HM3Dv1 ObjectNav val', episodes=len(all_rows),
        scenes=len(scenes), variant=args.variant,
        metrics={k:sum(float(r['metrics'][k]) for r in all_rows)/len(all_rows) for k in ('success','spl')})
    for name, value in [('summary.json',json.dumps(summary,indent=2)),
                        ('episodes.jsonl',''.join(json.dumps(r)+'\n' for r in all_rows))]:
        path=output/name
        if path.exists():
            if path.read_text() != value:
                raise SystemExit('Existing aggregate differs: '+str(path))
        else:
            path.write_text(value)
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
