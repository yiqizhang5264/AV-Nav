import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.frontier_probe_recorder import MapDeltaWriter, select_episode


class MapDeltaTests(unittest.TestCase):
    def test_exact_roundtrip_and_no_input_mutation(self):
        values = np.array([[0., -0., np.nan], [1., 2., 3.]], np.float32)
        with tempfile.TemporaryDirectory() as temp:
            writer = MapDeltaWriter(Path(temp) / "maps")
            saved = values.copy()
            writer.write(0, {"v": values})
            np.testing.assert_array_equal(values.view(np.uint8), saved.view(np.uint8))
            values[0, 0] = -0.
            values[1, 1] = 9.
            writer.write(1, {"v": values})
            with np.load(Path(temp) / "maps/0000.npz") as first:
                reconstructed = first["v__full"].copy()
            with np.load(Path(temp) / "maps/0001.npz") as delta:
                reconstructed.ravel()[delta["v__indices"]] = delta["v__values"]
            np.testing.assert_array_equal(values.view(np.uint8), reconstructed.view(np.uint8))
            with self.assertRaises(FileExistsError):
                writer.write(1, {"v": values})

    def test_select_full_identity_and_preserve_loader_id(self):
        from unittest.mock import patch
        import os
        identity = dict(scene_id="scene.basis.glb", episode_id="27", object_category="bed",
                        start_position=[0., 1., 2.], start_rotation=[0., 0., 0., 1.])

        class Env:
            num_envs = 1
            index = 0
            def call_at(self, index, method):
                result = dict(identity)
                if self.index == 0:
                    result["episode_id"] = "26"
                return {"episode_identity": result}
            def reset(self):
                self.index += 1
                return "selected observation"

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "case.json"
            path.write_text(json.dumps(dict(identity=identity)))
            with patch.dict(os.environ, VLFM_FRONTIER_CASE=str(path)):
                self.assertEqual(select_episode(Env(), "first"), "selected observation")
                identity["object_category"] = "chair"
                # A distinct wrong start must be rejected, not silently accepted.
                requested = dict(identity, start_position=[9., 1., 2.])
                path.write_text(json.dumps(dict(identity=requested)))
                with self.assertRaisesRegex(RuntimeError, "start_position"):
                    select_episode(Env(), "first")


if __name__ == "__main__":
    unittest.main()
