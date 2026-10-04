import gzip
import json
import pathlib
import tempfile
import unittest

from scripts.run_vlfm_evidence_suite import completed_episode_identities, episode_identity, prepare_remaining_dataset


class VLFMEvidenceSuiteTests(unittest.TestCase):
    def test_retry_dataset_excludes_only_completed_episodes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            source = root / "source"
            content = source / "val" / "content"
            content.mkdir(parents=True)
            with gzip.open(source / "val" / "val.json.gz", "wt", encoding="utf-8") as stream:
                json.dump({"episodes": []}, stream)
            data = {
                "episodes": [
                    {"episode_id": str(i), "scene_id": "scene.glb", "object_category": "bed",
                     "start_position": [i, 0, 0], "start_rotation": [0, 0, 0, 1]}
                    for i in range(4)
                ],
                "goals_by_category": {"scene.bed": []},
            }
            with gzip.open(content / "scene.json.gz", "wt", encoding="utf-8") as stream:
                json.dump(data, stream)

            destination, original_count, remaining_count = prepare_remaining_dataset(
                source, "scene", {episode_identity(data["episodes"][1]), episode_identity(data["episodes"][3])}, root / "derived"
            )
            with gzip.open(destination / "val" / "content" / "scene.json.gz", "rt", encoding="utf-8") as stream:
                derived = json.load(stream)

            self.assertEqual(original_count, 4)
            self.assertEqual(remaining_count, 2)
            self.assertEqual([episode["episode_id"] for episode in derived["episodes"]], ["0", "2"])
            self.assertEqual(derived["goals_by_category"], data["goals_by_category"])

    def test_completed_identities_preserve_repeated_episode_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            shard = pathlib.Path(directory)
            for attempt, position in (("001", [0, 0, 0]), ("002", [1, 0, 0])):
                episode = shard / "attempts" / attempt / "evidence" / "episodes" / attempt
                episode.mkdir(parents=True)
                (episode / "episode.json").write_text(json.dumps({
                    "episode_id": "7", "scene_id": "scene.glb", "object_category": "bed",
                    "start_position": position, "start_rotation": [0, 0, 0, 1],
                }))
                (episode / "result.json").write_text("{}")
            completed = completed_episode_identities(shard)
            self.assertEqual(len(completed), 2)


if __name__ == "__main__":
    unittest.main()
