import unittest
import numpy as np
from av_nav.diagnostic import episode_actions, camera_point


class DiagnosticTests(unittest.TestCase):
    def test_identity_and_sequence(self):
        s='=== Episode 3 (scene a/sceneA.basis.glb) - num 1 ===\nStep: 0 | Mode: initialize | Action: 2\n'
        s+='=== Episode 3 (scene a/sceneB.basis.glb) - num 2 ===\nStep: 0 | Mode: explore | Action: 1\n'
        self.assertEqual(episode_actions(s,'sceneB','3'),[(0,'explore',1)])
        with self.assertRaises(ValueError):
            episode_actions(s.replace('Step: 0 | Mode: explore','Step: 2 | Mode: explore'),'sceneB','3')
        with self.assertRaises(ValueError):
            episode_actions(s+s,'sceneA','3')

    def test_camera_axes(self):
        np.testing.assert_allclose(camera_point(320,240,2),[0,0,-2])
        self.assertGreater(camera_point(400,240,2)[0],0)
        self.assertLess(camera_point(320,300,2)[1],0)
        with self.assertRaises(ValueError): camera_point(320,240,float('nan'))
