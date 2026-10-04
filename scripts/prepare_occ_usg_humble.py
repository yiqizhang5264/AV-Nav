"""Make an explicit build overlay for OccuSG's PCL header compatibility."""
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
    header = output / 'src/octomap_mapping/octomap_server/include/octomap_server/octomap_server.hpp'
    original = header.read_bytes()
    old, new = b'"pcl_ros/transforms.hpp"', b'"pcl_ros/transforms.h"'
    if original.count(old) != 1:
        raise ValueError('Unexpected upstream header layout')
    header.write_bytes(original.replace(old, new))
    manifest = dict(upstream_commit=commit,
                    submodules=subprocess.check_output(['git', '-C', str(source), 'submodule', 'status'], text=True),
                    build_changes=[dict(path=str(header.relative_to(output)), reason='ROS2 Humble pcl_ros exports transforms.h',
                                        before_sha256=hashlib.sha256(original).hexdigest(),
                                        after_sha256=hashlib.sha256(header.read_bytes()).hexdigest())])
    (output / 'build_adapter.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
