import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace


spec = importlib.util.spec_from_file_location("run_ocs_cpu_suite", Path(__file__).parents[1] / "scripts/run_ocs_cpu_suite.py")
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)


def acceptance(key="one", decisions=2, steps=500, replayed=500):
    return {"dataset": "hm3dv1", "episode_key": key, "source_steps": steps,
            "replayed_steps": replayed, "frontier_decisions": decisions,
            "exact_frontier_goals": decisions, "original_printed_scores_matched": decisions,
            "complete_episode_reconstruction": replayed == steps, "mismatches": []}


class SuiteTests(unittest.TestCase):
    def test_indices_fixed_unique_and_bounded(self):
        self.assertEqual(suite.selected_indices(None, 3), [0, 1, 2])
        self.assertEqual(suite.selected_indices("2,0", 3), [2, 0])
        for value in ("1,1", "3", "-1"):
            with self.assertRaises(ValueError):
                suite.selected_indices(value, 3)

    def test_full_and_smoke_acceptance_are_distinct(self):
        case = {"dataset": "hm3dv1", "episode_key": "one"}
        summary = {"status": "complete", "with_ocs": True,
                   "baseline_acceptance": [acceptance()]}
        self.assertEqual(suite.baseline_issues(summary, case, None), [])
        summary.update(status="complete_prefix", baseline_acceptance=[acceptance(replayed=40)])
        self.assertEqual(suite.baseline_issues(summary, case, 40), [])
        issues = suite.baseline_issues(summary, case, None)
        self.assertIn("incomplete_full_episode_reconstruction", issues)
        self.assertIn("replayed_step_count_mismatch", issues)

    def test_acceptance_rejects_wrong_identity_missing_and_mismatch(self):
        case = {"dataset": "hm3dv1", "episode_key": "one"}
        summary = {"status": "complete", "with_ocs": True,
                   "baseline_acceptance": [acceptance("other")]}
        self.assertIn("baseline_identity_mismatch", suite.baseline_issues(summary, case, None))
        summary["baseline_acceptance"] = []
        self.assertIn("baseline_acceptance_count=0", suite.baseline_issues(summary, case, None))
        evidence = acceptance()
        evidence["exact_frontier_goals"] = 1
        summary["baseline_acceptance"] = [evidence]
        self.assertIn("baseline_goal_mismatch", suite.baseline_issues(summary, case, None))

    def test_fallback_counts_identity_queries_not_counterfactual_repetitions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            fallback = "geometry_unavailable_baseline_score_fallback"
            row = {"variants": [{"variant": "identity", "ocs": {"diagnostics": {"reason": fallback}}},
                                {"variant": "duplicate_each", "ocs": {"diagnostics": {"reason": fallback}}}]}
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            result = suite.frozen_diagnostics(path)
            self.assertEqual(result["sampled_identity_queries"], 1)
            self.assertEqual(result["fallback_count"], 1)
            row["variants"][0]["ocs"]["diagnostics"]["reason"] = "geometry_unavailable_upstream_fallback"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            result = suite.frozen_diagnostics(path)
            self.assertEqual(result["fallback_count"], 1)

    def test_summary_keeps_zero_exposure_and_rejects_failed_or_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "logs").mkdir()
            cases = []
            for index, key in enumerate(("one", "zero", "bad", "pending")):
                path = root / f"case_{index:02d}"
                path.mkdir()
                cases.append({"index": index, "dataset": "hm3dv1", "episode_key": key,
                              "stratum": "test", "directory": str(path)})
                if key == "pending":
                    continue
                suite.write_json(root / "logs" / f"case_{index:02d}.exit.json", {"returncode": 0})
                if key == "bad":
                    continue
                decisions = 0 if key == "zero" else 2
                summary = {"status": "complete", "with_ocs": True, "selection_sha256": "hash",
                           "baseline_acceptance": [acceptance(key, decisions)],
                           "counts": {"explore_decisions": decisions, "frozen_states_probed": int(bool(decisions))},
                           "variants": {"reverse_order": {"n": int(bool(decisions)), "ocs_coordinate_changed": 0}}}
                suite.write_json(path / "summary.json", summary)
                row = {"variants": [{"variant": "identity", "ocs": {"diagnostics": {"reason": "continue_valid_opportunity_anchor"}}}]}
                (path / "frozen_probes.jsonl").write_text((json.dumps(row) + "\n") if decisions else "", encoding="utf-8")
            suite.write_json(root / "manifest.json", {"cases": cases, "limit_steps": None,
                "selection_sha256": "hash", "git_commit": "commit", "interpretation": "shadow"})
            result = suite.summarize(root)
            self.assertEqual(result["accepted_cases"], [0, 1])
            self.assertEqual(result["zero_exploration_cases"], [1])
            self.assertEqual(result["failed_cases"], [2])
            self.assertEqual(result["pending_cases"], [3])
            self.assertEqual(result["counts_accepted_cases_only"]["explore_decisions"], 2)
            self.assertEqual(result["ocs_sampled_identity_fallback"]["denominator"], 1)
            self.assertEqual(result["ocs_all_exploration_fallback"]["unknown_cases"], [0])
            self.assertIsNone(result["ocs_all_exploration_fallback"]["fraction"])
            summary = suite.read_json(root / "case_00" / "summary.json")
            summary["counts"].update(ocs_calls=2, ocs_geometry_fallback_calls=1)
            suite.write_json(root / "case_00" / "summary.json", summary)
            result = suite.summarize(root)
            self.assertEqual(result["ocs_all_exploration_fallback"]["known_cases"], [0, 1])
            self.assertEqual(result["ocs_all_exploration_fallback"]["fraction"], 0.5)

    def test_prepare_refuses_existing_directory_and_freezes_cpu_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection = root / "input.json"
            selection.write_text(json.dumps({"episodes": [{"dataset": "hm3dv1", "episode_key": "one"}]}), encoding="utf-8")
            args = argparse.Namespace(workers=2, limit_steps=40, every_explore=20,
                selection=str(selection), vlfm_root=str(root), case_indices="0",
                output_dir=str(root / "result"))
            with mock.patch.object(suite, "git_revision", return_value="frozen"), \
                    mock.patch.object(suite, "validate_upstream", return_value=suite.PINNED_VLFM):
                output = suite.prepare(args)
                with self.assertRaises(FileExistsError):
                    suite.prepare(args)
            manifest = suite.read_json(output / "manifest.json")
            self.assertEqual(manifest["environment"]["CUDA_VISIBLE_DEVICES"], "")
            self.assertEqual(manifest["workers"], 2)
            command = manifest["cases"][0]["command"]
            self.assertIn("--with-ocs", command)
            self.assertEqual(command[-2:], ["--limit-steps", "40"])
            self.assertEqual((output / "selection.json").read_bytes(), selection.read_bytes())

    def test_upstream_requires_pinned_revision_and_clean_tracked_source(self):
        with mock.patch.object(suite, "git_revision", return_value="wrong"):
            with self.assertRaisesRegex(RuntimeError, "must be pinned"):
                suite.validate_upstream("upstream")
        with mock.patch.object(suite, "git_revision", return_value=suite.PINNED_VLFM), \
                mock.patch.object(suite.subprocess, "run", return_value=SimpleNamespace(returncode=1, stderr="")):
            with self.assertRaisesRegex(RuntimeError, "tracked modifications"):
                suite.validate_upstream("upstream")
        with mock.patch.object(suite, "git_revision", return_value=suite.PINNED_VLFM), \
                mock.patch.object(suite.subprocess, "run", return_value=SimpleNamespace(returncode=0, stderr="")) as run:
            self.assertEqual(suite.validate_upstream("upstream"), suite.PINNED_VLFM)
            self.assertEqual(run.call_args.args[0], ["git", "-C", "upstream", "diff", "HEAD", "--quiet"])

    def test_worker_bound_enforced_before_creation(self):
        args = argparse.Namespace(workers=3)
        with self.assertRaises(ValueError):
            suite.prepare(args)


if __name__ == "__main__":
    unittest.main()
