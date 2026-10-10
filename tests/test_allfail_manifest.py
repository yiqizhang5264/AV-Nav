"""Unit checks for complete-failure cohort identity and terminal classification."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

PATH = Path(__file__).resolve().parents[1] / "scripts/build_allfail_manifest.py"
SPEC = importlib.util.spec_from_file_location("allfail_manifest", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AllFailureManifestTest(unittest.TestCase):
    def test_scene_identity(self):
        self.assertEqual(MODULE.scene_name("hm3d/val/foo/foo.basis.glb"), "foo")
        self.assertEqual(MODULE.scene_name("mp3d/foo/foo.glb"), "foo")

    def test_nf_at_budget_has_priority(self):
        console = {"nf_message_lines": [21], "modes": ["explore"], "edge_message_lines": []}
        self.assertEqual(MODULE.classify_terminal(False, [{"action": 0}], console, True, 500, 500),
                         ("no_frontier_confirmed", True))

    def test_nonstop_budget_is_not_false_positive(self):
        console = {"nf_message_lines": [], "modes": ["explore"], "edge_message_lines": []}
        self.assertEqual(MODULE.classify_terminal(False, [{"action": 1}], console, True, 500, 500),
                         ("stepout_nonstop", False))

    def test_success_not_relabelled_nf(self):
        console = {"nf_message_lines": [21], "modes": ["explore"], "edge_message_lines": []}
        self.assertEqual(MODULE.classify_terminal(True, [{"action": 0}], console, True, 500, 500),
                         ("success", False))

    def test_invalid_join_cannot_confirm_nf(self):
        console = {"nf_message_lines": [21], "modes": ["explore"], "edge_message_lines": []}
        self.assertEqual(MODULE.classify_terminal(False, [{"action": 0}], console, False, 12, 500),
                         ("other_stop_failure", False))

    def test_console_episode_reset_and_duplicate_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "console.log"
            path.write_text("Step: 0 | Mode: explore | Action: 1\n"
                            "Logging episode 1 to /r/episode_logs/2_foo.json\n"
                            "No frontiers found during exploration, stopping.\n"
                            "Step: 0 | Mode: explore | Action: 0\n"
                            "Logging episode 2 to /r/episode_logs/3_foo.json\n"
                            "Step: 0 | Mode: navigate | Action: 0\n"
                            "Logging episode 3 to /r/episode_logs/2_foo.json\n")
            parsed = MODULE.parse_console(path)
            self.assertTrue(parsed[("foo", "2")]["repeated_key"])
            self.assertEqual(parsed[("foo", "3")]["modes"], ["explore"])
            self.assertEqual(len(parsed[("foo", "3")]["nf_message_lines"]), 1)
            self.assertEqual(parsed[("foo", "2")]["nf_message_lines"], [])

    def test_duplicate_metrics_sensitive_to_outcome(self):
        a = {"metrics": {"success": 0}, "steps_recorded": 3, "failure_cause": "false_positive"}
        b = dict(a, metrics={"success": 1})
        self.assertNotEqual(MODULE.semantic_result(a), MODULE.semantic_result(b))


if __name__ == "__main__":
    unittest.main()
