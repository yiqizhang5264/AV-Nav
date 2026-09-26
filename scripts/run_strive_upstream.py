#!/usr/bin/env python3
"""Run the pinned STRIVE entry point with explicit runtime-only overrides."""

from __future__ import annotations

import os
import pathlib
import runpy
import sys

from strive_vlm_runtime import VLMRuntime, install_openai_runtime
from strive_upstream_adapter import (
    install_episode_over_guard,
    install_interpolation_memory_guard,
    install_large_merge_memory_guard,
)


ROOT = pathlib.Path(__file__).resolve().parents[1]
STRIVE = ROOT / "external" / "strive"


def main() -> None:
    sys.path.insert(0, str(STRIVE))

    runtime = VLMRuntime.from_env()
    if runtime.backend == "openai_compatible":
        # The pinned constants module requires this variable even when requests
        # are routed to a local OpenAI-compatible server.
        os.environ.setdefault("GEMINI_API_KEY", "unused-local-placeholder")

    install_openai_runtime(runtime)

    # STRIVE imports MODEL_NAME by value in several modules. Set it before the
    # upstream entry point imports those modules.
    import constants

    constants.MODEL_NAME = runtime.model

    # STRIVE's final-check loop can issue one more action after Habitat ends an
    # episode at 500 steps. Keep the pinned submodule unchanged and guard that
    # runtime-only edge case here.
    import objnav_agent_with_process_obs
    import mapper_with_process_obs

    install_episode_over_guard(objnav_agent_with_process_obs.HM3D_Objnav_Agent)
    install_interpolation_memory_guard(mapper_with_process_obs.Instruct_Mapper)
    install_large_merge_memory_guard(mapper_with_process_obs)

    runpy.run_path(
        str(STRIVE / "objnav_benchmark_with_process_obs.py"),
        run_name="__main__",
    )


if __name__ == "__main__":
    main()
