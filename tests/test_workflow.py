import gzip
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


class SplitTests(unittest.TestCase):
    def test_disjoint_scenes_deterministic_ids_and_goals(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            content=root/'train/content'
            content.mkdir(parents=True)
            for i in range(20):
                scene=f'scene{i}.glb'
                data=dict(episodes=[dict(scene_id=scene,episode_id=str(j),object_category='chair') for j in range(3)],
                          goals_by_category={scene+'_chair':[dict(object_id=i)]})
                (content/f'scene{i}.json.gz').write_bytes(gzip.compress(json.dumps(data).encode()))
            paths=[]
            for name,partition in [('cal1','calibration'),('cal2','calibration'),('dev','development')]:
                dest=root/(name+'.json.gz')
                subprocess.run([sys.executable,str(ROOT/'scripts/make_split.py'),'--source',str(root/'train'),
                                '--output',str(dest),'--count','5','--partition',partition],check=True,capture_output=True)
                paths.append(dest)
            self.assertEqual(paths[0].read_bytes(),paths[1].read_bytes())
            a=json.loads(gzip.decompress(paths[0].read_bytes()))
            b=json.loads(gzip.decompress(paths[2].read_bytes()))
            self.assertFalse({e['scene_id'] for e in a['episodes']} & {e['scene_id'] for e in b['episodes']})
            self.assertEqual(len(a['episodes']),5)
            self.assertTrue(all(a['goals_by_category'][e['scene_id']+'_chair'] for e in a['episodes']))


class MetricsTests(unittest.TestCase):
    def test_mismatched_episode_sets_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            for name,episode in [('a','1'),('b','2')]:
                (root/name).write_text(json.dumps(dict(scene_id='scene',episode_id=episode,metrics=dict(success=1)))+'\n')
            result=subprocess.run([sys.executable,str(ROOT/'scripts/summarize.py'),str(root/'a'),
                                   '--baseline',str(root/'b')],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('identical scene/episode sets',result.stderr)

    def test_paired_recovery_and_regression(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            for name,success in [('a',[1,0,1]),('b',[0,1,0])]:
                rows=[dict(scene_id='scene',episode_id=str(i),metrics=dict(success=s,spl=s*.5)) for i,s in enumerate(success)]
                (root/name).write_text(''.join(json.dumps(r)+'\n' for r in rows))
            result=subprocess.run([sys.executable,str(ROOT/'scripts/summarize.py'),str(root/'a'),
                                   '--baseline',str(root/'b')],capture_output=True,text=True,check=True)
            paired=json.loads(result.stdout)['paired']
            self.assertEqual(paired['recovered'],2)
            self.assertEqual(paired['regressed'],1)


if __name__=='__main__':
    unittest.main()
