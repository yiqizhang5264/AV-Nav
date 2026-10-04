import gzip
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from make_room_split import select


class RoomSplitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'train'
        (self.source / 'content').mkdir(parents=True)
        for i in range(40):
            scene = f'hm3d/train/{i:05d}-scene{i}/scene{i}.basis.glb'
            data = dict(episodes=[dict(scene_id=scene, episode_id=f'{j:04d}', object_category='chair',
                                       start_position=[i, 0, j], start_rotation=[0, 0, 0, 1])
                                  for j in range(4)],
                        goals_by_category={Path(scene).name + '_chair': [dict(object_id=i)]})
            (self.source / 'content' / f'scene{i}.json.gz').write_bytes(
                gzip.compress(json.dumps(data).encode()))

    def test_unique_scenes_reproducible_preserved_identity_and_goals(self):
        a, ma = select(self.source, 5, 20261004, 'calibration')
        b, mb = select(self.source, 5, 20261004, 'calibration')
        self.assertEqual((a, ma), (b, mb))
        self.assertEqual(len({e['scene_id'] for e in a['episodes']}), 5)
        for episode, case in zip(a['episodes'], ma['cases']):
            original = json.loads(gzip.decompress(Path(case['source']).read_bytes()))
            self.assertIn(episode, original['episodes'])
            key = Path(episode['scene_id']).name + '_chair'
            self.assertEqual(a['goals_by_category'][key], original['goals_by_category'][key])
            self.assertEqual(len(case['source_sha256']), 64)

    def test_training_partitions_disjoint_and_insufficient_scene_count_rejected(self):
        a, _ = select(self.source, 5, 17, 'calibration')
        b, _ = select(self.source, 5, 17, 'development')
        self.assertFalse({e['scene_id'] for e in a['episodes']} & {e['scene_id'] for e in b['episodes']})
        with self.assertRaisesRegex(ValueError, 'distinct eligible scenes'):
            select(self.source, 41, 17, 'calibration')
        with self.assertRaisesRegex(ValueError, 'training split'):
            select(self.root / 'val', 5, 17, 'calibration')

    def test_cli_deterministic_gzip_and_no_overwrite(self):
        command = [sys.executable, str(ROOT / 'scripts/make_room_split.py'), '--source', str(self.source)]
        for name in ['a', 'b']:
            subprocess.run(command + ['--output-dir', str(self.root / name)], check=True, capture_output=True)
        self.assertEqual((self.root / 'a/episodes.json.gz').read_bytes(),
                         (self.root / 'b/episodes.json.gz').read_bytes())
        result = subprocess.run(command + ['--output-dir', str(self.root / 'a')], capture_output=True)
        self.assertNotEqual(result.returncode, 0)


if __name__ == '__main__':
    unittest.main()
