import os
import sys
import unittest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from strive_upstream_adapter import install_episode_over_guard


class _Environment:
    def __init__(self):
        self.episode_over = False
        self.actions = []

    def step(self, action, **kwargs):
        self.actions.append(action)
        return action


class StriveUpstreamAdapterTests(unittest.TestCase):
    def test_valid_steps_are_unchanged(self):
        class Agent:
            def __init__(self):
                self.env = _Environment()

            def step_mod(self):
                return self.env.step(1)

        install_episode_over_guard(Agent)
        agent = Agent()
        self.assertEqual(agent.step_mod(), 1)
        self.assertEqual(agent.env.actions, [1])

    def test_post_limit_step_ends_cleanly(self):
        class Agent:
            def __init__(self):
                self.env = _Environment()

            def step_mod(self):
                self.env.episode_over = True
                self.env.step(1)
                return True

        install_episode_over_guard(Agent)
        agent = Agent()
        self.assertFalse(agent.step_mod())
        self.assertEqual(agent.env.actions, [])


if __name__ == "__main__":
    unittest.main()
