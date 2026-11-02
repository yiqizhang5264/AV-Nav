import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location("ocs_queue_test", Path(__file__).resolve().parents[1]
                                             / "scripts/run_ocs_train_queue.py")
queue = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(queue)


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.runner = self.root / "fake_runner.py"
        self.runner.write_text("# frozen fake runner\n", encoding="utf-8")
        self.identities = [{"scene_id": f"scene{i}.glb", "object_category": "bed",
                            "start_position": [i, 0, 0], "start_rotation": [0, 0, 0, 1],
                            "source_episode_id": str(i), "filtered_index": i,
                            "row_sha256": str(i) * 64} for i in range(5)]
        for mode in queue.MODES:
            run = self.root / mode
            run.mkdir()
            data = run / "tasks.gz"
            data.write_bytes(b"identical frozen fixture")
            self.put_manifest(mode, {"mode": mode, "status": "prepared_not_started",
                                     "identities": self.identities,
                                     "dataset_path": str(data), "dataset_sha256": queue.file_sha(data)})
        self.calls = []

    def put_manifest(self, mode, manifest):
        (self.root / mode / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def prepared(self, modes=None):
        return queue.prepare_queue(self.root, self.runner, modes=modes)

    def fake_run(self, command, **kwargs):
        self.assertEqual(command[2], "execute")
        self.assertEqual(command[-1], "--resources-ready")
        self.assertEqual(kwargs["stderr"], subprocess.STDOUT)
        self.assertFalse(kwargs["check"])
        run = Path(command[4])
        self.calls.append(run.name)
        completion = {"returncode": 0, "complete": True, "physical_identities_match": True,
                      "completed_result_files": 5, "expected_episodes": 5}
        (run / "started.json").write_text("{}")
        (run / "completion.json").write_text(json.dumps(completion))
        return SimpleNamespace(returncode=0)

    def test_runs_three_conditions_in_order_and_records_hashes(self):
        target = self.prepared()
        manifest = queue.read_json(target / "manifest.json")
        self.assertEqual(len(manifest["conditions"]), 3)
        self.assertEqual(manifest["identities"], self.identities)
        self.assertEqual(manifest["modes"], list(queue.MODES))
        with patch("builtins.print"):
            self.assertEqual(queue.supervise(target, self.fake_run), 0)
        self.assertEqual(self.calls, list(queue.MODES))
        summary = queue.read_json(target / "summary.json")
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["completed_modes"], list(queue.MODES))
        self.assertTrue(all(r["elapsed_seconds"] >= 0 for r in summary["conditions"]))

    def test_subset_runs_only_selected_modes_and_finishes(self):
        # No original manifest is required: this queue does not rerun or resume it.
        (self.root / "original/manifest.json").unlink()
        target = self.prepared(["anchor_only", "ocs"])
        self.assertEqual(queue.read_json(target / "manifest.json")["modes"], ["anchor_only", "ocs"])
        with patch("builtins.print"):
            self.assertEqual(queue.supervise(target, self.fake_run), 0)
        self.assertEqual(self.calls, ["anchor_only", "ocs"])
        summary = queue.read_json(target / "summary.json")
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["requested_modes"], ["anchor_only", "ocs"])
        self.assertEqual(summary["not_started_modes"], [])
        self.assertFalse((self.root / "original/started.json").exists())

    def test_single_selected_mode_finishes(self):
        target = self.prepared(["ocs"])
        with patch("builtins.print"):
            self.assertEqual(queue.supervise(target, self.fake_run), 0)
        self.assertEqual(self.calls, ["ocs"])
        self.assertTrue(queue.read_json(target / "summary.json")["complete"])

    def test_invalid_subsets_rejected_without_creating_queue(self):
        for modes in ([], (), ["original", "original"], ["unknown"],
                      ["ocs", "anchor_only"], ["anchor_only", "original"], "ocs"):
            with self.subTest(modes=modes), self.assertRaises(ValueError):
                self.prepared(modes)
            self.assertFalse((self.root / "queue").exists())

    def test_subset_failure_does_not_start_next_condition(self):
        target = self.prepared(["anchor_only", "ocs"])
        def fail(command, **kwargs):
            self.calls.append(Path(command[4]).name)
            return SimpleNamespace(returncode=9)
        with patch("builtins.print"):
            self.assertEqual(queue.supervise(target, fail), 9)
        self.assertEqual(self.calls, ["anchor_only"])
        summary = queue.read_json(target / "summary.json")
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["not_started_modes"], ["ocs"])

    def test_subset_still_refuses_started_selected_run(self):
        path = self.root / "anchor_only/started.json"
        path.write_text("preserve")
        with self.assertRaises(ValueError):
            self.prepared(["anchor_only", "ocs"])
        self.assertEqual(path.read_text(), "preserve")

    def test_nonzero_stops_before_next_condition(self):
        target = self.prepared()
        def fail(command, **kwargs):
            self.calls.append(Path(command[4]).name)
            return SimpleNamespace(returncode=7)
        with patch("builtins.print"):
            self.assertEqual(queue.supervise(target, fail), 7)
        self.assertEqual(self.calls, ["original"])
        summary = queue.read_json(target / "summary.json")
        self.assertEqual(summary["not_started_modes"], ["anchor_only", "ocs"])
        self.assertFalse(summary["complete"])

    def test_zero_return_but_incomplete_stops(self):
        target = self.prepared()
        def incomplete(command, **kwargs):
            self.fake_run(command, **kwargs)
            path = Path(command[4]) / "completion.json"
            value = queue.read_json(path)
            value["completed_result_files"] = 4
            path.write_text(json.dumps(value))
            return SimpleNamespace(returncode=0)
        with patch("builtins.print"):
            self.assertEqual(queue.supervise(target, incomplete), 2)
        self.assertEqual(self.calls, ["original"])

    def test_full_identity_mismatch_rejected_before_queue_created(self):
        path = self.root / "ocs/manifest.json"
        manifest = queue.read_json(path)
        manifest["identities"][0]["row_sha256"] = "changed"
        self.put_manifest("ocs", manifest)
        with self.assertRaises(ValueError):
            self.prepared()
        self.assertFalse((self.root / "queue").exists())

    def test_dataset_mismatch_rejected(self):
        (self.root / "anchor_only/tasks.gz").write_bytes(b"different")
        with self.assertRaises(ValueError):
            self.prepared()

    def test_mode_mismatch_rejected(self):
        manifest = queue.read_json(self.root / "ocs/manifest.json")
        manifest["mode"] = "original"
        self.put_manifest("ocs", manifest)
        with self.assertRaises(ValueError):
            self.prepared()

    def test_existing_started_or_completion_never_overwritten(self):
        for filename in ("started.json", "completion.json"):
            path = self.root / "ocs" / filename
            path.write_text("preserve")
            with self.assertRaises(ValueError):
                self.prepared()
            self.assertEqual(path.read_text(), "preserve")
            path.unlink()

    def test_run_started_after_preflight_is_rejected(self):
        target = self.prepared()
        (self.root / "original/started.json").write_text("preserve")
        with patch("builtins.print"):
            self.assertEqual(queue.supervise(target, self.fake_run), 2)
        self.assertFalse(self.calls)
        self.assertEqual((self.root / "original/started.json").read_text(), "preserve")

    def test_runner_change_after_prepare_stops_without_child(self):
        target = self.prepared()
        self.runner.write_text("changed")
        with patch("builtins.print"):
            self.assertEqual(queue.supervise(target, self.fake_run), 2)
        self.assertFalse(self.calls)

    def test_existing_queue_and_supervisor_are_not_resumed(self):
        target = self.prepared()
        with self.assertRaises(FileExistsError):
            self.prepared()
        with patch("builtins.print"):
            self.assertEqual(queue.supervise(target, self.fake_run), 0)
        with self.assertRaises(FileExistsError):
            queue.supervise(target, self.fake_run)

    def test_detach_creates_new_session_and_records_pid(self):
        target = self.prepared()
        with patch.object(queue.subprocess, "Popen", return_value=SimpleNamespace(pid=1234)) as popen:
            result = queue.detach(target)
        self.assertEqual(result["pid"], 1234)
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        self.assertEqual(popen.call_args.kwargs["stdin"], subprocess.DEVNULL)
        self.assertEqual(popen.call_args.args[0][-2:], ["--worker", str(target)])
        self.assertTrue((target / "stdout.log").is_file())
        self.assertEqual(queue.read_json(target / "launch.json")["pid"], 1234)


if __name__ == "__main__":
    unittest.main()
