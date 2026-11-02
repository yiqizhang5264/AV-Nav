from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from scripts.vlfm_episode_identity import FullEpisodeCache, validate_full_episode


def full(scene="s", eid="0"):
    return SimpleNamespace(scene_id=scene, episode_id=eid, object_category="chair",
                           start_position=[1., 2., 3.], start_rotation=[0., 0., 0., 1.])


class FullIdentityTests(unittest.TestCase):
    def test_brief_rejected_instead_of_silent_nulls(self):
        with self.assertRaisesRegex(ValueError, "all_info"):
            validate_full_episode(SimpleNamespace(scene_id="s", episode_id="0"))

    def test_one_read_rpc_per_episode_no_step_or_reset(self):
        envs = Mock()
        envs.call_at.side_effect = [full(), full(eid="1")]
        cache = FullEpisodeCache()
        brief = SimpleNamespace(scene_id="s", episode_id="0")
        first = cache.resolve(envs, 0, brief)
        self.assertIs(cache.resolve(envs, 0, brief), first)
        self.assertEqual(first.start_position, [1., 2., 3.])
        cache.resolve(envs, 0, SimpleNamespace(scene_id="s", episode_id="1"))
        self.assertEqual(envs.call_at.call_count, 2)
        envs.call_at.assert_called_with(0, "current_episode", {"all_info": True})
        envs.step.assert_not_called()
        envs.reset.assert_not_called()

    def test_mismatched_rpc_not_cached(self):
        envs = Mock()
        envs.call_at.return_value = full(eid="wrong")
        with self.assertRaisesRegex(ValueError, "disagrees"):
            FullEpisodeCache().resolve(envs, 0, SimpleNamespace(scene_id="s", episode_id="0"))

    def test_nan_and_zero_rotation_rejected(self):
        e = full()
        e.start_position[0] = float("nan")
        with self.assertRaises(ValueError):
            validate_full_episode(e)
        e = full()
        e.start_rotation = [0.] * 4
        with self.assertRaises(ValueError):
            validate_full_episode(e)

    def test_real_habitat_all_info_api_without_simulator(self):
        try:
            from habitat.core.env import RLEnv
        except ModuleNotFoundError:
            self.skipTest("Habitat available in server validation environment")
        task = full()
        holder = SimpleNamespace(_env=SimpleNamespace(current_episode=task))
        brief = RLEnv.current_episode(holder)
        self.assertFalse(hasattr(brief, "start_position"))
        envs = Mock()
        envs.call_at.side_effect = lambda index, name, args: RLEnv.current_episode(holder, **args)
        self.assertIs(FullEpisodeCache().resolve(envs, 0, brief), task)
