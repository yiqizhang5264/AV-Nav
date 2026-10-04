import json
import os
import pathlib
import tempfile
import unittest
from unittest.mock import patch

from scripts.run_vlfm_evidence_suite import completed_episode_ids
from scripts.vlfm_episode_resume import skip_completed_initial_episodes


class Episode:
    def __init__(self, episode_id):
        self.episode_id = str(episode_id)


class FakeEnvs:
    num_envs = 1

    def __init__(self, ids):
        self.ids = list(map(str, ids))
        self.index = 0

    def current_episodes(self):
        return [Episode(self.ids[self.index])]

    def reset(self):
        self.index += 1
        return [{"episode": self.ids[self.index]}]


class VLFMEvidenceSuiteTests(unittest.TestCase):
    def test_completed_ids_are_deduplicated_across_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            shard = pathlib.Path(directory)
            for attempt in ("001", "002"):
                episode = shard / "attempts" / attempt / "evidence" / "episodes" / attempt
                episode.mkdir(parents=True)
                (episode / "episode.json").write_text(json.dumps({"episode_id": "7"}))
                (episode / "result.json").write_text("{}")
            self.assertEqual(completed_episode_ids(shard), {"7"})

    def test_resume_skips_exact_deterministic_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = pathlib.Path(directory) / "plan.json"
            plan.write_text(json.dumps({"episode_ids": ["5", "2", "9"]}))
            envs = FakeEnvs(["5", "2", "9", "4"])
            with patch.dict(os.environ, {"VLFM_SKIP_EPISODES_FILE": str(plan)}):
                observations = skip_completed_initial_episodes(envs, [{"episode": "5"}])
            self.assertEqual(observations, [{"episode": "4"}])

    def test_resume_rejects_non_prefix_completed_set(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = pathlib.Path(directory) / "plan.json"
            plan.write_text(json.dumps({"episode_ids": ["5", "9"]}))
            envs = FakeEnvs(["5", "2", "9", "4"])
            with patch.dict(os.environ, {"VLFM_SKIP_EPISODES_FILE": str(plan)}):
                with self.assertRaisesRegex(RuntimeError, "not the deterministic initial prefix"):
                    skip_completed_initial_episodes(envs, [{"episode": "5"}])


if __name__ == "__main__":
    unittest.main()
