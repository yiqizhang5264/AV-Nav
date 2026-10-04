"""Archive small reproducibility metadata alongside a completed comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['comparison-dir', 'replay-dir', 'audit-dir', 'capture-dir', 'build-dir', 'active-dir', 'occusg-dir']:
        p.add_argument('--'+name, required=True)
    args = p.parse_args()
    comparison, replay, audit, capture, build, active, occusg = map(Path, [
        args.comparison_dir, args.replay_dir, args.audit_dir, args.capture_dir,
        args.build_dir, args.active_dir, args.occusg_dir])
    summary = json.loads((comparison/'summary.json').read_text())
    if summary['paired_complete'] != 5:
        raise ValueError('Only archive final comparison after all five pairs complete')
    output = comparison/'provenance'
    output.mkdir(exist_ok=False)
    shutil.copy2(replay/'depth_correspondence.json', output/'depth_correspondence.json')
    shutil.copy2(audit/'environment.txt', output/'environment.txt')
    shutil.copy2(capture/'hydra/.hydra/config.yaml', output/'capture_config.yaml')
    shutil.copy2(build/'build_adapter.json', output/'occusg_build_adapter.json')
    audit_data = json.loads((audit/'summary.json').read_text())
    # Disabled-renderer semantic AABBs can contain Infinity; do not create
    # non-standard JSON in a tracked summary. These are not room footprints.
    def finite(value):
        if isinstance(value, float):
            import math
            return value if math.isfinite(value) else None
        if isinstance(value, dict):
            return {k: finite(v) for k,v in value.items()}
        if isinstance(value, list):
            return [finite(v) for v in value]
        return value
    (output/'resource_audit.json').write_text(json.dumps(finite(audit_data), indent=2, allow_nan=False))
    frame_digests, methods = {}, []
    for idx in range(5):
        case_id = f'{idx:02d}'
        frame_digests[case_id] = {path.name: digest(path) for path in sorted((replay/case_id).glob('[0-9][0-9][0-9][0-9].npz'))}
        for name, root in [('Active',active), ('OccuSG',occusg)]:
            case_dir = root/case_id
            native = json.loads((case_dir/'summary.json').read_text())
            source = native.pop('source')
            native.pop('records')
            native['source'] = {k:v for k,v in source.items() if k not in ['poses','episode']}
            native['method'] = name
            methods.append(native)
            if name == 'OccuSG':
                dest = output/f'occusg_{case_id}'
                dest.mkdir()
                for filename in ['pipeline_params.yaml','dude_params.yaml','commands.json']:
                    shutil.copy2(case_dir/filename, dest/filename)
    (output/'method_summaries.json').write_text(json.dumps(methods,indent=2))
    (output/'input_frame_sha256.json').write_text(json.dumps(frame_digests,indent=2))
    code_root = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(['git','-C',str(code_root),'rev-parse','HEAD'],text=True).strip()
    script_hashes = {p.name:digest(p) for p in (code_root/'scripts').glob('*room*.py')}
    metadata = dict(collector_commit=commit, collector_scripts_sha256=script_hashes,
                    active_adapter_commit=subprocess.check_output(['git','-C',str(active),'rev-parse','HEAD'],text=True).strip(),
                    occusg_adapter_commit=subprocess.check_output(['git','-C',str(occusg),'rev-parse','HEAD'],text=True).strip(),
                    capture_commit=json.loads((capture/'manifest.json').read_text())['av_nav_commit'],
                    replay_commit=json.loads((replay/'manifest.json').read_text()).get('av_nav_commit', '0f28e75'),
                    vlfm_commit='584ed56008754fde7997d904983607def8328322',
                    paths=dict(replay=str(replay),active=str(active),occusg=str(occusg),capture=str(capture),build=str(build)),
                    scope='HM3D v1 ObjectNav episodes / HM3D-0.2 scene assets; shared observations; qualitative only')
    (output/'run_identity.json').write_text(json.dumps(metadata,indent=2))


if __name__ == '__main__':
    main()
