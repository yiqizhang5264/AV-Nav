#!/usr/bin/env python3
"""Compose STRIVE metrics, RGB and depth videos horizontally per episode."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess


INPUT_NAMES = ("metrics.mp4", "fps.mp4", "depth.mp4")
OUTPUT_NAME = "combined_metrics_fps_depth.mp4"


def episode_directories(roots: list[pathlib.Path]) -> list[pathlib.Path]:
    found: set[pathlib.Path] = set()
    for root in roots:
        for metrics in root.rglob(INPUT_NAMES[0]):
            directory = metrics.parent
            if all((directory / name).is_file() for name in INPUT_NAMES):
                found.add(directory)
    return sorted(found)


def compose(directory: pathlib.Path, height: int, overwrite: bool) -> str:
    output = directory / OUTPUT_NAME
    if output.is_file() and output.stat().st_size > 0 and not overwrite:
        return "skipped"
    temporary = directory / f".{OUTPUT_NAME}.tmp.mp4"
    temporary.unlink(missing_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
    ]
    for name in INPUT_NAMES:
        command.extend(["-i", str(directory / name)])
    filters = ";".join(
        [
            f"[{index}:v]scale=-2:{height},setsar=1,setpts=PTS-STARTPTS[v{index}]"
            for index in range(3)
        ]
        + ["[v0][v1][v2]hstack=inputs=3:shortest=1[out]"]
    )
    command.extend(
        [
            "-filter_complex",
            filters,
            "-map",
            "[out]",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(temporary),
        ]
    )
    subprocess.run(command, check=True)
    temporary.replace(output)
    return "created"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=pathlib.Path)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.height <= 0 or args.height % 2:
        parser.error("--height must be a positive even number")

    summary: dict[str, object] = {"created": 0, "skipped": 0, "failed": []}
    for directory in episode_directories(args.roots):
        try:
            status = compose(directory, args.height, args.overwrite)
            summary[status] = int(summary[status]) + 1
        except (OSError, subprocess.CalledProcessError) as error:
            failures = summary["failed"]
            assert isinstance(failures, list)
            failures.append({"directory": str(directory), "error": str(error)})
    print(json.dumps(summary, indent=2))
    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
