import importlib.util
from pathlib import Path
import tempfile
import gzip
import json
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from av_nav.planner import Grid
from av_nav.sap_geometry import HeightMap, depth_points, select_view
from av_nav.sap_vlm import parse_reply, Verifier


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

    def test_final_json_with_explanation_and_fence(self):
        self.assertEqual(parse_reply('The object is visible.\n```json\n'
            '{"visibility":5,"perspective":5}\n```', 'sufficiency'),
            {'visibility':5,'perspective':5})
        self.assertEqual(parse_reply('It is a sofa.\n{"matches":false}', 'category'),
                         {'matches':False})
        for text in ['{"matches":false} then {"matches":true}',
                     'Explanation {"matches":"false"}', '[]', 'No matching object.']:
            with self.assertRaises(ValueError):
                parse_reply(text, 'category')

    def test_thinking_exhaustion_retries_without_accepting_empty_content(self):
        import io
        observation = SimpleNamespace(rgb=np.zeros((8,8,3),dtype=np.uint8), bbox=(0,0,4,4))
        replies = [dict(choices=[dict(message=dict(content=None,reasoning_content='unfinished'),
                                     finish_reason='length')]),
                   dict(choices=[dict(message=dict(content='{"matches":false}'))])]
        budgets = []
        def respond(request, **kwargs):
            body = json.loads(request.data)
            budgets.append(body['max_tokens'])
            self.assertTrue(body['chat_template_kwargs']['enable_thinking'])
            return io.BytesIO(json.dumps(replies.pop(0)).encode())
        with patch('av_nav.sap_vlm.urllib.request.urlopen', side_effect=respond):
            result, record = Verifier(dict(base_url='http://local/v1',model='qwen',max_tokens=8192)).ask(
                observation,'bed','category')
        self.assertEqual(result, {'matches':False})
        self.assertEqual(budgets, [8192,512])
        self.assertEqual(len(record['retry_failures']),1)

    def test_exhausted_retries_fail_instead_of_approving_candidate(self):
        import io
        observation = SimpleNamespace(rgb=np.zeros((8,8,3),dtype=np.uint8), bbox=(0,0,4,4))
        def respond(*args, **kwargs):
            return io.BytesIO(b'{"choices":[{"message":{"content":""}}]}')
        with patch('av_nav.sap_vlm.urllib.request.urlopen', side_effect=respond):
            with self.assertRaises(RuntimeError):
                Verifier(dict(base_url='http://local/v1',model='qwen',max_tokens=8192)).ask(
                    observation,'bed','category')

    def test_thinking_continuation_keeps_images_and_does_not_disable_thinking(self):
        import io
        observation = SimpleNamespace(rgb=np.zeros((8,8,3),dtype=np.uint8),bbox=(0,0,4,4))
        replies=[dict(choices=[dict(message=dict(content=None,reasoning='visual assessment'),finish_reason='length')]),
                 dict(choices=[dict(message=dict(content=None,reasoning='"matches":false}'),finish_reason='stop')])]
        requests=[]
        def respond(request, **kwargs):
            requests.append(json.loads(request.data))
            return io.BytesIO(json.dumps(replies.pop(0)).encode())
        with patch('av_nav.sap_vlm.urllib.request.urlopen',side_effect=respond):
            result, record=Verifier(dict(base_url='http://local/v1',model='qwen',max_tokens=8192)).ask(
                observation,'bed','category')
        self.assertEqual(result,{'matches':False})
        self.assertEqual(record['reasoning'],'visual assessment')
        self.assertEqual(record['api_requests'],2)
        self.assertEqual(requests[1]['messages'][0],requests[0]['messages'][0])
        self.assertTrue(requests[1]['chat_template_kwargs']['enable_thinking'])
        self.assertTrue(requests[1]['continue_final_message'])
        self.assertFalse(requests[1]['add_generation_prompt'])
        self.assertIn('visual assessment\n',requests[1]['messages'][-1]['content'])
        self.assertTrue(requests[1]['messages'][-1]['content'].endswith('</think>\n\n{'))

    def test_continuation_never_accepts_invalid_final_text_or_uses_initial_cot_as_answer(self):
        import io
        observation=SimpleNamespace(rgb=np.zeros((8,8,3),dtype=np.uint8),bbox=(0,0,4,4))
        calls=[]
        def respond(request, **kwargs):
            body=json.loads(request.data);calls.append(body)
            if body.get('continue_final_message'):
                message=dict(content=None,reasoning='not a JSON answer')
                finish='stop'
            else:
                message=dict(content=None,reasoning='I guess {"matches":true} but need to think more')
                finish='length'
            return io.BytesIO(json.dumps(dict(choices=[dict(message=message,finish_reason=finish)])).encode())
        with patch('av_nav.sap_vlm.urllib.request.urlopen',side_effect=respond):
            with self.assertRaises(RuntimeError):
                Verifier(dict(base_url='http://local/v1',model='qwen',max_tokens=8192)).ask(
                    observation,'bed','category')
        self.assertEqual(len(calls),6)

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
            row = dict(data['sap_source_identities'][0], metrics={'success':1,'spl':.5})
            resume = root/'episodes.jsonl'
            resume.write_text(json.dumps(row)+'\n')
            pending, rows = module.resume_dataset(data, resume)
            self.assertEqual(len(pending['episodes']),1)
            self.assertEqual(pending['sap_source_identities'][0]['source_row'],1)
            self.assertEqual(len(rows),1)
            row['episode_sha256'] = 'changed'
            resume.write_text(json.dumps(row)+'\n')
            with self.assertRaises(ValueError):
                module.resume_dataset(data, resume)
