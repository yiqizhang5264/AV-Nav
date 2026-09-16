#!/usr/bin/env python3
"""Run the pinned STRIVE entry point with explicit runtime-only overrides."""

from __future__ import annotations

import os
import pathlib
import runpy
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
STRIVE = ROOT / "external" / "strive"


def main() -> None:
    sys.path.insert(0, str(STRIVE))

    # STRIVE pins a Gemini model in constants.py. Keep the submodule unchanged,
    # but allow a recorded model override when that API model is unavailable.
    import constants

    model = os.environ.get("STRIVE_GEMINI_MODEL", "").strip()
    if model:
        constants.MODEL_NAME = model

    runpy.run_path(
        str(STRIVE / "objnav_benchmark_with_process_obs.py"),
        run_name="__main__",
    )


if __name__ == "__main__":
    main()
