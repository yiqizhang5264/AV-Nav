import pathlib
import tempfile
import unittest

import numpy as np

from scripts.vlfm_evidence_adapter import create_vlfm_evidence_overlay

try:
    from scripts.vlfm_evidence_recorder import EvidenceRecorder
except ModuleNotFoundError as error:
    if error.name != "cv2":
        raise
    EvidenceRecorder = None


class VLFMEvidenceAdapterTests(unittest.TestCase):
    def test_overlay_injects_capture_hooks(self):
        source = '''from omegaconf import OmegaConf
            current_episodes_info = self.envs.current_episodes()

            with inference_mode():
                pass
            for i in range(len(policy_infos)):
                infos[i].update(policy_infos[i])
            batch = batch_obs(  # type: ignore
                    try:
                        pass
                    except Exception:
                        failure_cause = "Unknown"

                    if len(self.config.habitat_baselines.eval.video_option) > 0:
                        pass
'''
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "vlfm" / "utils").mkdir(parents=True)
            (root / "vlfm" / "utils" / "vlfm_trainer.py").write_text(source)
            overlay = create_vlfm_evidence_overlay(root)
            patched = (pathlib.Path(overlay.name) / "vlfm" / "utils" / "vlfm_trainer.py").read_text()
            overlay.cleanup()
        self.assertIn("record_observation", patched)
        self.assertIn("record_transition", patched)
        self.assertIn("finish_episode", patched)

    @unittest.skipIf(EvidenceRecorder is None, "OpenCV is installed in the server VLFM environment")
    def test_capture_does_not_advance_numpy_rng(self):
        class Episode:
            scene_id = "data/scene.glb"
            episode_id = "7"
            object_category = "tv_monitor"
            start_position = [0.0, 0.0, 0.0]
            start_rotation = [0.0, 0.0, 0.0, 1.0]
            goals = []

        class Detections:
            image_source = np.zeros((24, 32, 3), dtype=np.uint8)
            boxes = np.array([[0.1, 0.1, 0.8, 0.8]], dtype=np.float32)
            logits = np.array([0.75], dtype=np.float32)
            phrases = ["tv"]

        np.random.seed(1234)
        expected = np.random.get_state()
        with tempfile.TemporaryDirectory() as directory:
            recorder = EvidenceRecorder(pathlib.Path(directory))
            recorder.record_observation(Episode(), {"rgb": Detections.image_source, "depth": np.ones((24, 32, 1))})
            recorder.record_detections(Detections(), "tv", "filtered_target")
            recorder.finish_episode({"success": 0.0}, "false_positive")
        actual = np.random.get_state()
        self.assertEqual(expected[0], actual[0])
        np.testing.assert_array_equal(expected[1], actual[1])
        self.assertEqual(expected[2:], actual[2:])


if __name__ == "__main__":
    unittest.main()
