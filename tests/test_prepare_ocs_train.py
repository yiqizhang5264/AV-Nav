import copy
import gzip
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare_ocs_train", ROOT / "scripts/prepare_ocs_train.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TrainPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "train"
        (self.source / "content").mkdir(parents=True)
        self.assets = self.root / "assets"
        self.row = dict(scene_id="hm3d/train/00001-a/a.basis.glb", episode_id="12", object_category="bed",
                        start_position=[1, 2, 3], start_rotation=[0, 0, 0, 1], info={"test": True})
        asset = self.assets / self.row["scene_id"]
        asset.parent.mkdir(parents=True)
        asset.write_bytes(b"test asset")
        self.data = dict(episodes=[self.row], goals_by_category={"a.basis.glb_bed": [{"object_id": "1"}]},
                         content_scenes_path="{data_path}/__no_external_content__/{scene}.json.gz")
        (self.source / "content/a.json.gz").write_bytes(gzip.compress(json.dumps(self.data).encode()))

    def tearDown(self):
        self.temp.cleanup()

    def test_preserves_source_id_and_full_payload(self):
        result = module.validate_tasks(self.data, self.source, self.assets)
        self.assertEqual(result[0]["source_episode_id"], "12")
        self.assertEqual(result[0]["source_row_index"], 0)
        self.assertNotIn("loaded_episode_id", result[0])

    def test_changed_nongeometric_payload_rejected(self):
        changed = copy.deepcopy(self.data)
        changed["episodes"][0]["info"]["test"] = False
        with self.assertRaises(ValueError):
            module.validate_tasks(changed, self.source, self.assets)

    def test_changed_goals_rejected(self):
        changed = copy.deepcopy(self.data)
        changed["goals_by_category"]["a.basis.glb_bed"][0]["object_id"] = "2"
        with self.assertRaises(ValueError):
            module.validate_tasks(changed, self.source, self.assets)

    def test_external_content_rejected(self):
        changed = copy.deepcopy(self.data)
        changed["content_scenes_path"] = "{data_path}/content/{scene}.json.gz"
        with self.assertRaises(ValueError):
            module.validate_tasks(changed, self.source, self.assets)

    def test_duplicate_physical_identity_rejected(self):
        changed = copy.deepcopy(self.data)
        changed["episodes"].append(copy.deepcopy(changed["episodes"][0]))
        with self.assertRaises(ValueError):
            module.validate_tasks(changed, self.source, self.assets)


if __name__ == "__main__":
    unittest.main()
