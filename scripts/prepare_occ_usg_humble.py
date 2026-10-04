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
    # Preserve the native default. Offline replay can explicitly request
    # reliable delivery so a large point cloud cannot disappear mid-sequence.
    octomap = output / 'src/octomap_mapping/octomap_server/src/octomap_server.cpp'
    original = octomap.read_text()
    old = 'point_cloud_sub_.subscribe(this, "cloud_in", rmw_qos_profile_sensor_data);'
    new = '''auto cloud_qos = rmw_qos_profile_sensor_data;
  if (declare_parameter("cloud_sub_qos_reliable", false)) {
    cloud_qos.reliability = RMW_QOS_POLICY_RELIABILITY_RELIABLE;
  }
  point_cloud_sub_.subscribe(this, "cloud_in", cloud_qos);'''
    if original.count(old) != 1:
        raise ValueError('Pinned OctoMap subscription does not match transport adapter')
    octomap.write_text(original.replace(old, new))
    manifest = dict(upstream_commit=commit,
                    submodules=subprocess.check_output(['git', '-C', str(source), 'submodule', 'status'], text=True),
                    build_changes=[dict(file=str(octomap.relative_to(output)),
                                        original_sha256=hashlib.sha256(original.encode()).hexdigest(),
                                        adapted_sha256=hashlib.sha256(octomap.read_bytes()).hexdigest(),
                                        change='Optional reliable cloud subscription; default false preserves native transport')],
                    note='Humble pcl_ros supplies transforms.hpp; replay transport only, no geometry or segmentation changes.')
    (output / 'build_adapter.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
