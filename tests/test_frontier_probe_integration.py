import os
import gzip
import hashlib
import json
from pathlib import Path
import py_compile
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from decision_trace import install_decision_trace
from replay_frontier_probe import load_replay_class
from run_vlfm_frontier_probe import extend_overlay, prepare_selected_case
from vlfm_evidence_adapter import create_vlfm_evidence_overlay


class ProbeIntegrationTests(unittest.TestCase):
    def test_manifest_preparation_checks_shard_and_keeps_loaded_not_original_id(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            row = dict(scene_id="scene.glb", episode_id="0", object_category="bed",
                       start_position=[0, 0, 0], start_rotation=[0, 0, 0, 1])
            content = gzip.compress(json.dumps(dict(episodes=[row, row])).encode())
            shard = folder / "shard.json.gz"
            shard.write_bytes(content)
            selected = dict(task_content_path=str(shard), task_loaded_index_in_content=1,
                            task_content_sha256=hashlib.sha256(content).hexdigest(),
                            task_row_sha256=hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
            manifest = folder / "selection.json"
            manifest.write_text(json.dumps(dict(episodes=[selected])))
            self.assertEqual(prepare_selected_case(manifest, 0)["identity"]["episode_id"], "1")
            shard.write_bytes(content + b"changed")
            with self.assertRaisesRegex(ValueError, "content changed"):
                prepare_selected_case(manifest, 0)

    def test_real_overlay_compiles_and_captures_before_environment_step(self):
        with create_vlfm_evidence_overlay(ROOT / "external/vlfm") as directory:
            class Overlay:
                name = directory
            extend_overlay(Overlay())
            trainer = Path(directory) / "vlfm/utils/vlfm_trainer.py"
            policy = Path(directory) / "vlfm/policy/base_objectnav_policy.py"
            for path in (trainer, policy):
                py_compile.compile(str(path), doraise=True)
            source = trainer.read_text()
            self.assertLess(source.index("record_environment(self.envs)"), source.index("outputs = self.envs.step(step_data)"))
            self.assertIn("observations = select_episode(self.envs, observations)", source)
            self.assertIn("record_policy_frame(self, mode)", policy.read_text())

    def test_cpu_extracted_future_annotations_survive_hook(self):
        source = ROOT / "external/vlfm/vlfm/policy/itm_policy.py"
        policy = load_replay_class(source, "ITMPolicyV2", dict(np=np, os=os))
        # The original method has Union/Tensor annotations, deliberately not
        # loaded here. Instrumentation must preserve postponed annotations.
        handle = install_decision_trace(policy, lambda event: None)
        handle.uninstall()


if __name__ == "__main__":
    unittest.main()
