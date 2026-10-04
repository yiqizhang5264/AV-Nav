"""Make an isolated build tree while preserving the pinned OccuSG sources."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source, output = Path(args.upstream), Path(args.output)
    if output.exists():
        raise ValueError('Build overlay must be a new directory')
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != '2bca2fa06af87fd9dd0542957039be8f580f0dca':
        raise ValueError('Unexpected OccuSG commit')
    shutil.copytree(source / 'src', output / 'src', ignore=shutil.ignore_patterns('.git', 'build', 'install', 'log'))
    manifest = dict(upstream_commit=commit,
                    submodules=subprocess.check_output(['git', '-C', str(source), 'submodule', 'status'], text=True),
                    build_changes=[],
                    note='Humble pcl_ros supplies transforms.hpp after installing ros-humble-pcl-ros; no source patch needed.')
    (output / 'build_adapter.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
