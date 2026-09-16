#!/usr/bin/env python3
"""Run the pinned STRIVE entry point with explicit runtime-only overrides."""

from __future__ import annotations

import os
import pathlib
import runpy
import sys

from strive_vlm_runtime import VLMRuntime, install_openai_runtime


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

    runpy.run_path(
        str(STRIVE / "objnav_benchmark_with_process_obs.py"),
        run_name="__main__",
    )


if __name__ == "__main__":
    main()
