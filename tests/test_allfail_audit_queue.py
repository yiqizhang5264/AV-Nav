import json
from pathlib import Path
import tempfile
import unittest

from scripts.run_allfail_frontier_audit import case_priority, save_json, summarize


class AuditQueueTests(unittest.TestCase):
    def test_denominator_keeps_unknown_and_pending(self):
        records = [dict(dataset="hm3dv1", status="analyzed", episode_category="no_qualifying_spatial_return",
                        spatial_return_events=0, low_yield_return_events=0),
                   dict(dataset="hm3dv1", status="pending"),
                   dict(dataset="hm3dv1", status="unknown_input_or_floor_evidence")]
        result = summarize(records, 3)
        self.assertEqual(result["by_dataset"]["hm3dv1"]["all_completed_failures"], 3)
        self.assertEqual(result["by_dataset"]["hm3dv1"]["analyzed"], 1)
        self.assertTrue(result["no_causal_recoverability_claim"])

    def test_order_h1_h2_before_partial_mp3d(self):
        rows = [(0, dict(dataset="mp3d", terminal_class="stepout_nonstop")),
                (1, dict(dataset="hm3dv1", terminal_class="target_navigation_stop_failure")),
                (2, dict(dataset="hm3dv2", terminal_class="no_frontier_confirmed"))]
        self.assertEqual([i for i, _ in sorted(rows, key=case_priority)], [2, 1, 0])

    def test_atomic_summary_roundtrip(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "summary.json"
            save_json(path, {"count": 1})
            save_json(path, {"count": 2})
            self.assertEqual(json.loads(path.read_text()), {"count": 2})
            self.assertFalse(path.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
