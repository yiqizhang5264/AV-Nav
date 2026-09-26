"""Small runtime guards for deterministic failures in pinned STRIVE code."""

from __future__ import annotations

from typing import Any


class _EpisodeAlreadyOver(RuntimeError):
    pass


def install_episode_over_guard(agent_class: type[Any]) -> None:
    """End a step cleanly if STRIVE steps after Habitat's 500-step limit."""
    if getattr(agent_class, "_av_nav_episode_over_guard", False):
        return

    original_step_mod = agent_class.step_mod

    def guarded_step_mod(self: Any, *args: Any, **kwargs: Any) -> Any:
        original_env_step = self.env.step

        def guarded_env_step(action: Any, **step_kwargs: Any) -> Any:
            if self.env.episode_over:
                raise _EpisodeAlreadyOver
            return original_env_step(action, **step_kwargs)

        self.env.step = guarded_env_step
        try:
            return original_step_mod(self, *args, **kwargs)
        except _EpisodeAlreadyOver:
            return False
        finally:
            self.env.step = original_env_step

    agent_class.step_mod = guarded_step_mod
    agent_class._av_nav_episode_over_guard = True
