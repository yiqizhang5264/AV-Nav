import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from av_nav.config import Config
from av_nav.geometry import observe, adaptive_range, angle_delta
from av_nav.planner import Grid, select_view


class GeometryTests(unittest.TestCase):
    def observation(self):
        rgb = np.zeros((100,100,3),dtype=np.uint8)
        depth = np.ones((100,100)) * .4
        mask = np.zeros((100,100),bool)
        mask[35:65,35:65] = True
        return observe(rgb,depth,mask,np.eye(4),0,5,100,100,0)

    def test_metric_depth_and_axis(self):
        obs = self.observation()
        self.assertAlmostEqual(obs.depth,2)
        self.assertAlmostEqual(obs.center[0],2)
        self.assertLess(abs(obs.center[1]),.02)
        self.assertAlmostEqual(obs.area,.09)

    def test_invalid_depth_rejected(self):
        for value in [0,1,np.nan,np.inf]:
            self.assertIsNone(observe(np.zeros((10,10,3)),np.full((10,10),value),
                                     np.ones((10,10)),np.eye(4),0,5,10,10,0))

    def test_surface_distance_clamp_and_truncation(self):
        obs, cfg = self.observation(), Config()
        obs.area = 1e-6
        self.assertEqual(adaptive_range(obs,cfg),cfg.min_range)
        obs.boundary = .5
        self.assertEqual(adaptive_range(obs,cfg),cfg.fixed_range)

    def test_angle_wrap(self):
        self.assertAlmostEqual(angle_delta(np.deg2rad(-179),np.deg2rad(179)),np.deg2rad(2))


class PlannerTests(unittest.TestCase):
    def grid(self,free):
        return Grid(free,~free,10,lambda p:np.rint(p*10).astype(int),lambda p:p/10)

    def test_wall_cannot_be_crossed(self):
        free=np.ones((60,60),bool)
        free[:,30]=False
        costs,_=self.grid(free).distances([1,1],10)
        self.assertNotIn((10,40),costs)

    def test_corner_cutting_forbidden(self):
        free=np.array([[True,False],[False,True]])
        costs,_=self.grid(free).distances([0,0],1)
        self.assertNotIn((1,1),costs)

    def test_budget_uses_path_not_euclidean(self):
        free=np.ones((60,60),bool)
        free[:45,30]=False
        costs,_=self.grid(free).distances([2.5,2],1)
        self.assertNotIn((20,35),costs)

    def test_no_candidate_is_safe(self):
        obs=GeometryTests().observation()
        view,points=select_view(obs,[obs.position],obs.position,
                                self.grid(np.zeros((60,60),bool)),Config(),np.random.default_rng(0))
        self.assertIsNone(view)
        self.assertEqual(points,[])

    def test_seeded_view_path_and_budget(self):
        obs=GeometryTests().observation()
        obs.center[:2]=[3,3]
        obs.points[:,:2]+=[1,3]
        cfg=Config(strategy="random")
        grid=self.grid(np.ones((70,70),bool))
        first,_=select_view(obs,[np.array([1,3])],np.array([1,3]),grid,cfg,np.random.default_rng(17))
        second,_=select_view(obs,[np.array([1,3])],np.array([1,3]),grid,cfg,np.random.default_rng(17))
        self.assertIsNotNone(first)
        np.testing.assert_equal(first['xy'],second['xy'])
        self.assertLessEqual(first['cost'],cfg.max_path)
        self.assertTrue(len(first['path'])>0)


class ConfigTests(unittest.TestCase):
    def test_unknown_key_fails(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'config.json'
            p.write_text(json.dumps(dict(stratgey='active')))
            with self.assertRaises(ValueError):
                Config.load(p)


if __name__ == '__main__':
    unittest.main()
