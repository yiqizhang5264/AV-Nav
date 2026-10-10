"""Read every failed trace: trajectory recurrence screen, NOT ineffective-return labels."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


def trajectory_screen(xy, departure=3.0, return_radius=0.75, gap=20):
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError("invalid XY trajectory")
    if len(xy) < 2:
        return dict(spatial_return=False, witness=None, observed_path_m=0.0, jumps_over_1m=0)
    moves = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    distance_matrix = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
    # For anchor i and later end j, columns before i cannot introduce a larger
    # distance: explicitly mask them before accumulating along time.
    chronological = np.triu(distance_matrix)
    max_departure = np.maximum.accumulate(chronological, axis=1)
    # No same-frontier or category identity is needed. This is deliberately a screen.
    for end in range(gap, len(xy)):
        matches = np.flatnonzero((distance_matrix[:end - gap + 1, end] <= return_radius) &
                                 (max_departure[:end - gap + 1, end] >= departure))
        if len(matches):
            start = int(matches[0])
            from_anchor = distance_matrix[start, start:end + 1]
            far = int(start + np.argmax(from_anchor))
            return dict(spatial_return=True, witness=dict(old_step=start,
                        farthest_step=far, returned_step=end,
                        departure_m=float(from_anchor.max()),
                        return_distance_m=float(distance_matrix[start, end])),
                        observed_path_m=float(moves.sum()), jumps_over_1m=int((moves > 1).sum()))
    return dict(spatial_return=False, witness=None, observed_path_m=float(moves.sum()),
                jumps_over_1m=int((moves > 1).sum()))


def xy_from_row(row):
    value = row.get("policy_info", {}).get("gps")
    if isinstance(value, str):
        value = np.fromstring(value.strip("[]").replace(",", " "), sep=" ")
    xy = np.asarray(value, dtype=float)
    if xy.shape != (2,) or not np.isfinite(xy).all():
        raise ValueError("missing/nonfinite policy GPS")
    # Reflection convention is irrelevant to distances; preserve recorded axes.
    return xy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.selection)
    selection = json.loads(source.read_text())
    result = dict(schema="vlfm.allfail_trajectory_screen.v1",
                  selection_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  no_causal_claim=True, no_map_gain_evaluated=True,
                  parameters=dict(departure_m=3.0, return_radius_m=0.75, gap_actions=20),
                  cases=[], by_dataset={})
    counts = {}
    for index, case in enumerate(selection["episodes"]):
        dataset = case["dataset"]
        count = counts.setdefault(dataset, Counter())
        count["all_completed_failures"] += 1
        record = dict(case_index=index, dataset=dataset, episode_key=case["episode_key"],
                      terminal_class=case.get("terminal_class"),
                      reported_stairs=case.get("reported_stairs"),
                      strict_2d_input_eligible=case.get("strict_2d_input_eligible"))
        try:
            raw = Path(case["trace_source"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != case["source_trace_sha256"]:
                raise ValueError("trace hash changed")
            rows = [json.loads(line) for line in raw.splitlines()]
            if [r["step"] for r in rows] != list(range(len(rows))):
                raise ValueError("noncontiguous steps")
            screen = trajectory_screen([xy_from_row(row) for row in rows])
            record.update(screen, status="screened")
            count["screened"] += 1
            count["spatial_return_screen_positive"] += int(screen["spatial_return"])
            if case.get("strict_2d_input_eligible") and not screen["jumps_over_1m"]:
                count["strict_2d_screened"] += 1
                count["strict_2d_spatial_return_screen_positive"] += int(screen["spatial_return"])
            count["with_pose_jump_over_1m"] += int(screen["jumps_over_1m"] > 0)
        except Exception as exc:
            record.update(status="unknown", error=f"{type(exc).__name__}: {exc}")
            count["unknown"] += 1
        result["cases"].append(record)
        if index % 100 == 0:
            print("SCREEN", index, dataset, flush=True)
    result["by_dataset"] = {name: dict(c) for name, c in counts.items()}
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result["by_dataset"], indent=2), flush=True)


if __name__ == "__main__":
    main()
