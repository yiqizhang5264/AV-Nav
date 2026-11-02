import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.reverify_ocs_identity import check_episode
from scripts.run_vlfm_ocs import completion_record


class IdentityReverificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.folder = self.root / "evidence/episodes/scene"
        self.folder.mkdir(parents=True)
        self.expected = dict(scene_id="data/s.glb", loaded_episode_id="0", source_episode_id="5",
                             object_category="sofa", start_position=[1, 2, 3], start_rotation=[0, 0, 0, 1])
        self.transform = np.eye(4)
        self.transform[:3, 3] = [-3, -1, 2]
        self.metrics = dict(success=1., spl=.4, distance_to_goal=.01, stop_called=1.)
        self.trace = [dict(step=0, done=True, policy_info=dict(target_object="couch"),
                           info=dict(self.metrics, top_down_map=dict(tf_episodic_to_global=self.transform.tolist())))]
        self.meta = dict(scene_id="/data/s.glb", episode_id="0", object_category="unknown",
                         start_position=None, start_rotation=None)
        self.write()

    def tearDown(self):
        self.temp.cleanup()

    def write(self):
        (self.folder / "episode.json").write_text(json.dumps(self.meta))
        (self.folder / "result.json").write_text(json.dumps(dict(steps_recorded=1, metrics=self.metrics)))
        (self.folder / "steps.jsonl").write_text("\n".join(map(json.dumps, self.trace)))

    def test_runtime_transform_recovers_identity_without_editing_metadata(self):
        before = (self.folder / "episode.json").read_bytes()
        result = check_episode(self.folder, self.expected, self.transform)
        self.assertTrue(result["identity_verified"])
        self.assertEqual(before, (self.folder / "episode.json").read_bytes())
        self.assertEqual(result["max_transform_abs_error"], 0.)

    def test_expected_pose_is_not_silently_copied_when_runtime_differs(self):
        changed = self.transform.copy()
        changed[0, 3] += .1
        with self.assertRaisesRegex(ValueError, "origin transform differs"):
            check_episode(self.folder, self.expected, changed)

    def test_target_mismatch_rejected(self):
        self.trace[0]["policy_info"]["target_object"] = "bed"
        self.write()
        with self.assertRaisesRegex(ValueError, "target mismatch"):
            check_episode(self.folder, self.expected, self.transform)

    def test_missing_terminal_rejected(self):
        self.trace[0]["done"] = False
        self.write()
        with self.assertRaisesRegex(ValueError, "terminal"):
            check_episode(self.folder, self.expected, self.transform)

    def test_noncontiguous_steps_rejected(self):
        self.trace[0]["step"] = 2
        self.write()
        with self.assertRaisesRegex(ValueError, "Noncontiguous"):
            check_episode(self.folder, self.expected, self.transform)

    def test_tilt_not_claimed_verified_by_heading_only(self):
        self.expected["start_rotation"] = [.1, 0, 0, .995]
        with self.assertRaisesRegex(ValueError, "Non-upright"):
            check_episode(self.folder, self.expected, self.transform)

    def test_new_completion_records_bad_metadata_instead_of_crashing(self):
        result = completion_record(self.root, [self.expected], 0)
        self.assertFalse(result["complete"])
        self.assertEqual(result["returncode"], 0)
        self.assertEqual(result["completed_result_files"], 1)
        self.assertIn("Incomplete recorded identity", result["verification_error"])

    def test_good_metadata_allows_normal_completion(self):
        self.meta.update({k: self.expected[k] for k in ("object_category", "start_position", "start_rotation")})
        self.write()
        result = completion_record(self.root, [self.expected], 0)
        self.assertTrue(result["complete"])
        self.assertIsNone(result["verification_error"])
