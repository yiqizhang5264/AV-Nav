import importlib.util
from pathlib import Path
import tempfile
import gzip
import json
import unittest
import numpy as np
from av_nav.planner import Grid
from av_nav.sap_geometry import HeightMap, depth_points, select_view
from av_nav.sap_vlm import parse_reply


class SAPTests(unittest.TestCase):
    def test_runtime_import_does_not_start_habitat(self):
        path = Path(__file__).resolve().parents[1]/'scripts/run_sap_runtime.py'
        spec = importlib.util.spec_from_file_location('sap_runtime_import_test', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertTrue(callable(module.main))

    def grid(self):
        return Grid(np.ones((100,100), bool), np.zeros((100,100), bool), 10,
                    lambda xy: np.rint(xy*10+50).astype(int), lambda px: (px-50)/10)

    def test_height_blocks_but_low_obstacle_does_not(self):
        grid = self.grid()
        height = HeightMap(grid.free.shape, grid.xy_to_px)
        target = np.array([[2., 0., 1.]])
        height.values[50,60] = .1
        self.assertEqual(height.visibility([0,0], target, 1.5, grid), 1)
        height.values[50,60] = 2
        self.assertEqual(height.visibility([0,0], target, 1.5, grid), 0)

    def test_unknown_not_transparent(self):
        grid = self.grid()
        grid.free[50,60] = False
        height = HeightMap(grid.free.shape, grid.xy_to_px)
        self.assertEqual(height.visibility([0,0], np.array([[2,0,1]]), 1.5, grid), 0)

    def test_online_height_maximum(self):
        grid = self.grid()
        height = HeightMap(grid.free.shape, grid.xy_to_px)
        height.update(np.array([[0,0,.2],[0,0,1.5],[0,0,.5]]))
        self.assertEqual(height.values[50,50], 1.5)

    def test_ceiling_does_not_become_a_solid_column(self):
        grid = self.grid()
        height = HeightMap(grid.free.shape, grid.xy_to_px)
        height.update(np.array([[0,0,.2],[0,0,2.5],[0,0,.5]]), max_height=.88)
        self.assertEqual(height.values[50,50], .5)

    def test_depth_units_and_camera_frame(self):
        points = depth_points(np.array([[.5]]), np.eye(4), 0, 10, 1, 1)
        np.testing.assert_allclose(points, [[5,0,0]])

    def test_view_rings_fov_and_reachability(self):
        grid = self.grid()
        height = HeightMap(grid.free.shape, grid.xy_to_px)
        best, candidates = select_view(np.array([[0.,0.,.5]]), np.array([3.,0.]), grid, height,
                                       1.5, np.pi/2)
        self.assertIsNotNone(best)
        for item in candidates:
            self.assertGreaterEqual(np.linalg.norm(item['xy']), 1.)
            self.assertTrue(any(abs(np.linalg.norm(item['xy'])-r)<1e-6 for r in [1.2,1.6,2.,2.4]))
        grid.free[:,65] = False
        _, candidates = select_view(np.array([[0.,0.,.5]]), np.array([3.,0.]), grid, height, 1.5, np.pi/2)
        self.assertTrue(all(item['cell'][1]>65 for item in candidates))

    def test_strict_vlm_schema(self):
        self.assertEqual(parse_reply('{"matches":false}', 'category'), {'matches':False})
        for text in ['{"matches":"false"}', '{"matches":1}', '{}']:
            with self.assertRaises(ValueError):
                parse_reply(text, 'category')
        for text in ['{"visibility":true,"perspective":5}', '{"visibility":6,"perspective":2}']:
            with self.assertRaises(ValueError):
                parse_reply(text, 'sufficiency')

    def test_dataset_preserves_repeated_ids_with_source_row_identity(self):
        path = Path(__file__).resolve().parents[1]/'scripts/run_sap_eval.py'
        spec = importlib.util.spec_from_file_location('sap_eval', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        scene_root = Path(tempfile.gettempdir()).resolve()
        self.assertEqual(module.episode_keys([{'scene_id':str(scene_root/'hm3d/s.glb'),
                                               'episode_id':'0057'}], scene_root),
                         [('hm3d/s.glb','0057')])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'val/content').mkdir(parents=True)
            with gzip.open(root/'val/val.json.gz','wt') as stream:
                json.dump({'episodes':[]},stream)
            episode = {'scene_id':'scene','episode_id':'0057','object_category':'bed'}
            content = root/'val/content/scene.json.gz'
            with gzip.open(content,'wt') as stream:
                json.dump({'episodes':[episode]},stream)
            data, hashes = module.read_dataset(root, 'val')
            self.assertEqual(data['episodes'][0]['episode_id'], '0057')
            self.assertEqual(len(hashes), 2)
            with gzip.open(content,'wt') as stream:
                json.dump({'episodes':[episode,episode]},stream)
            data, _ = module.read_dataset(root, 'val')
            self.assertEqual([e['episode_id'] for e in data['episodes']], ['0057','0057'])
            self.assertNotEqual(data['sap_source_identities'][0]['source_uid'],
                                data['sap_source_identities'][1]['source_uid'])
            self.assertEqual(len(set(module.episode_keys(data['sap_source_identities']))),2)
