#!/usr/bin/env python3
"""Reset one STRIVE HM3D v2 episode and report sensor observations."""

from __future__ import annotations

import pathlib
import sys

import habitat


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external" / "strive"))

from config_utils import hm3d_config  # noqa: E402


def main() -> None:
    config = hm3d_config(stage="val", episodes=1)
    with habitat.Env(config=config) as env:
        observations = env.reset()
        episode = env.current_episode
        print("episode_id", episode.episode_id)
        print("scene_id", episode.scene_id)
        for name, value in observations.items():
            print(
                "observation",
                name,
                getattr(value, "shape", None),
                getattr(value, "dtype", type(value).__name__),
            )


if __name__ == "__main__":
    main()
