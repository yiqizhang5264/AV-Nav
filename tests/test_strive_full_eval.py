import csv
import os
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from run_strive_full_eval import (
    _is_acceptable_open3d_postrun_abort,
    _run_shard_with_retries,
    attempt_is_complete,
    shard_ranges,
)


class StriveFullEvalTests(unittest.TestCase):
    def test_shard_ranges_cover_interval_once(self):
        ranges = shard_ranges(3, 11, 3)
        self.assertEqual(ranges, [(3, 6), (6, 9), (9, 11)])
        self.assertEqual(
            [index for start, stop in ranges for index in range(start, stop)],
            list(range(3, 11)),
        )

    def test_attempt_requires_zero_exit_and_exact_episode_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            attempt = pathlib.Path(directory)
            output = attempt / "output"
            output.mkdir()
            (attempt / "exit_code.txt").write_text("0\n")
            with (output / "metrics.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["Episode", "success"])
                writer.writeheader()
                writer.writerows(
                    [{"Episode": index, "success": 1} for index in range(10, 13)]
                )
            self.assertTrue(attempt_is_complete(attempt, 10, 13))
            self.assertFalse(attempt_is_complete(attempt, 10, 14))
            (attempt / "exit_code.txt").write_text("1\n")
            self.assertFalse(attempt_is_complete(attempt, 10, 13))

    def test_failed_shard_is_retried_until_complete(self):
        results = [
            {"complete": False, "attempt": "attempt_001"},
            {"complete": True, "attempt": "attempt_002"},
        ]
        with patch("run_strive_full_eval._run_shard", side_effect=results) as run:
            result = _run_shard_with_retries(
                pathlib.Path("suite"), 10, 20, "av", "strive", 3
            )
        self.assertTrue(result["complete"])
        self.assertEqual(result["attempts_in_invocation"], 2)
        self.assertEqual(run.call_count, 2)

    def test_only_complete_open3d_destructor_abort_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            console = pathlib.Path(directory) / "console.log"
            console.write_text(
                "Open3D Error Cacher::~Cacher() 1 leaking memory blocks on CUDA:0"
            )
            self.assertTrue(
                _is_acceptable_open3d_postrun_abort(
                    -6, console, [60, 61], 60, 62
                )
            )
            self.assertFalse(
                _is_acceptable_open3d_postrun_abort(-6, console, [60], 60, 62)
            )
            self.assertFalse(
                _is_acceptable_open3d_postrun_abort(1, console, [60, 61], 60, 62)
            )


if __name__ == "__main__":
    unittest.main()
