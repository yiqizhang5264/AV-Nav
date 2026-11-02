"""Run the three pre-prepared train smoke5 conditions sequentially.

Resources/services must already have been checked by the operator. This program
does not load models, select GPUs, start services, resume runs, or alter settings.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

MODES = ("original", "anchor_only", "ocs")
EXPECTED_EPISODES = 5


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def file_sha(path):
    return digest_bytes(Path(path).read_bytes())


def json_sha(value):
    return digest_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                  ensure_ascii=False).encode("utf-8"))


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def assert_unstarted(run):
    for name in ("started.json", "completion.json"):
        if (run / name).exists():
            raise ValueError(f"Refusing previously started/completed run: {run / name}")


def prepare_queue(run_root, runner):
    root, runner = Path(run_root).resolve(), Path(runner).resolve()
    if not root.is_dir() or not runner.is_file():
        raise ValueError("Existing run root and runner file are required")
    records, reference = [], None
    for mode in MODES:
        run = root / mode
        assert_unstarted(run)
        path = run / "manifest.json"
        manifest = read_json(path)
        if manifest.get("mode") != mode:
            raise ValueError(f"Prepared mode mismatch: {path}")
        if manifest.get("status") != "prepared_not_started":
            raise ValueError(f"Run is not prepared_not_started: {path}")
        identities = manifest.get("identities")
        if not isinstance(identities, list) or len(identities) != EXPECTED_EPISODES:
            raise ValueError(f"Expected exactly five task identities: {path}")
        required = {"scene_id", "object_category", "start_position", "start_rotation",
                    "source_episode_id", "filtered_index", "row_sha256"}
        if any(not required.issubset(row) for row in identities):
            raise ValueError(f"Incomplete task identities: {path}")
        dataset_sha = manifest["dataset_sha256"]
        if file_sha(manifest["dataset_path"]) != dataset_sha:
            raise ValueError(f"Prepared dataset SHA mismatch: {path}")
        identity_sha = json_sha(identities)
        if reference is None:
            reference = (dataset_sha, identity_sha)
        elif reference != (dataset_sha, identity_sha):
            raise ValueError("The three conditions must use identical data and ordered full identities")
        command = [sys.executable, str(runner), "execute", "--run-dir", str(run),
                   "--resources-ready"]
        records.append({"mode": mode, "run_dir": str(run), "manifest_path": str(path),
                        "manifest_sha256": file_sha(path), "command": command,
                        "command_sha256": json_sha(command)})
    queue = root / "queue"
    queue.mkdir()  # Existing supervisor directories are never resumed/overwritten.
    manifest = {"schema": "vlfm.ocs.train_queue.v1", "created_utc": utc_now(),
                "run_root": str(root), "expected_episodes_per_mode": EXPECTED_EPISODES,
                "dataset_sha256": reference[0], "identities_sha256": reference[1],
                "identities": identities, "conditions": records,
                "resources_checked_by_operator": True,
                "source_hashes": {str(Path(__file__).resolve()): file_sha(__file__),
                                  str(runner): file_sha(runner)}}
    write_new(queue / "manifest.json", manifest)
    return queue


def verify_frozen(manifest, condition):
    for path, expected in manifest["source_hashes"].items():
        if file_sha(path) != expected:
            raise ValueError(f"Queue/runner source changed: {path}")
    if file_sha(condition["manifest_path"]) != condition["manifest_sha256"]:
        raise ValueError(f"Prepared run manifest changed: {condition['manifest_path']}")
    if json_sha(condition["command"]) != condition["command_sha256"]:
        raise ValueError("Child command hash mismatch")
    assert_unstarted(Path(condition["run_dir"]))


def completion_ok(completion):
    return (completion.get("complete") is True
            and completion.get("physical_identities_match") is True
            and completion.get("returncode") == 0
            and completion.get("completed_result_files") == EXPECTED_EPISODES
            and completion.get("expected_episodes") == EXPECTED_EPISODES)


def supervise(queue, run_command=None):
    queue = Path(queue).resolve()
    manifest = read_json(queue / "manifest.json")
    run_command = run_command or subprocess.run
    write_new(queue / "supervisor.json", {"pid": os.getpid(), "started_utc": utc_now(),
              "session_id": os.getsid(0) if hasattr(os, "getsid") else None,
              "queue_manifest_sha256": file_sha(queue / "manifest.json")})
    started = time.monotonic()
    records, exit_code = [], 0
    for index, condition in enumerate(manifest["conditions"]):
        mode, stage_start = condition["mode"], time.monotonic()
        record = {"mode": mode, "started_utc": utc_now(),
                  "command_sha256": condition["command_sha256"], "returncode": None,
                  "complete": False}
        print(f"START {mode}: {condition['run_dir']}", flush=True)
        try:
            verify_frozen(manifest, condition)
            with (queue / f"{index:02d}_{mode}.log").open("x", encoding="utf-8") as stream:
                result = run_command(condition["command"], stdout=stream,
                                     stderr=subprocess.STDOUT, check=False)
            record["returncode"] = int(result.returncode)
            completion_path = Path(condition["run_dir"]) / "completion.json"
            if completion_path.is_file():
                record["completion"] = read_json(completion_path)
                record["completion_sha256"] = file_sha(completion_path)
                record["complete"] = result.returncode == 0 and completion_ok(record["completion"])
            if not record["complete"]:
                record["error"] = "Child failed or completion missing/incomplete; later conditions not started"
                exit_code = int(result.returncode) if result.returncode > 0 else 2
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            exit_code = 2
        record.update(elapsed_seconds=time.monotonic() - stage_start, finished_utc=utc_now())
        write_new(queue / f"{index:02d}_{mode}.json", record)
        records.append(record)
        print(f"END {mode}: complete={record['complete']} returncode={record['returncode']}", flush=True)
        if exit_code:
            break
    summary = {"schema": "vlfm.ocs.train_queue_summary.v1", "finished_utc": utc_now(),
               "complete": exit_code == 0 and len(records) == len(MODES),
               "exit_code": exit_code, "elapsed_seconds": time.monotonic() - started,
               "completed_modes": [r["mode"] for r in records if r["complete"]],
               "not_started_modes": list(MODES[len(records):]), "conditions": records}
    write_new(queue / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return exit_code


def detach(queue):
    queue = Path(queue).resolve()
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", str(queue)]
    with (queue / "stdout.log").open("x", encoding="utf-8") as stream:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True,
                                 close_fds=True)
    launch = {"pid": child.pid, "launched_utc": utc_now(), "command": command,
              "command_sha256": json_sha(command), "stdout_path": str(queue / "stdout.log")}
    write_new(queue / "launch.json", launch)
    return launch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root")
    parser.add_argument("--runner")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if args.run_root or args.runner or args.detach:
            parser.error("--worker may not be combined with launch options")
        raise SystemExit(supervise(args.worker))
    if not args.run_root or not args.runner:
        parser.error("--run-root and --runner are required")
    queue = prepare_queue(args.run_root, args.runner)
    if args.detach:
        print(json.dumps(detach(queue), indent=2))
    else:
        raise SystemExit(supervise(queue))


if __name__ == "__main__":
    main()
