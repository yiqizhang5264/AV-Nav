#!/usr/bin/env python3
"""Summarize completed shards from a STRIVE full-evaluation suite."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import statistics

from run_strive_full_eval import attempt_is_complete, shard_ranges


ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-id", required=True)
    args = parser.parse_args()
    suite = ROOT / "runs" / args.suite_id
    manifest = json.loads((suite / "manifest.json").read_text())
    rows: dict[int, dict[str, str]] = {}
    attempts: dict[int, str] = {}
    completed_shards = 0
    for start, stop in shard_ranges(
        int(manifest["start"]), int(manifest["stop"]), int(manifest["shard_size"])
    ):
        pattern = f"shard_{start:04d}_{stop - 1:04d}_attempt_*"
        complete = next(
            (
                path
                for path in sorted((suite / "shards").glob(pattern))
                if attempt_is_complete(path, start, stop)
            ),
            None,
        )
        if complete is None:
            continue
        completed_shards += 1
        with (complete / "output" / "metrics.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            for row in csv.DictReader(stream):
                episode = int(row["Episode"])
                rows[episode] = row
                attempts[episode] = complete.name

    ordered = [rows[index] for index in sorted(rows)]
    summary: dict[str, object] = {
        "suite_id": args.suite_id,
        "expected_episodes": int(manifest["stop"]) - int(manifest["start"]),
        "completed_episodes": len(ordered),
        "completed_shards": completed_shards,
        "missing_episodes": [
            index
            for index in range(int(manifest["start"]), int(manifest["stop"]))
            if index not in rows
        ],
    }
    if ordered:
        summary.update(
            {
                "success": statistics.fmean(float(row["success"]) for row in ordered),
                "spl": statistics.fmean(float(row["spl"]) for row in ordered),
                "distance_to_goal": statistics.fmean(
                    float(row["distance_to_goal"]) for row in ordered
                ),
                "mean_episode_steps": statistics.fmean(
                    int(row["Episode Steps"]) for row in ordered
                ),
            }
        )
    (suite / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if ordered:
        with (suite / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=ordered[0].keys())
            writer.writeheader()
            writer.writerows(ordered)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
