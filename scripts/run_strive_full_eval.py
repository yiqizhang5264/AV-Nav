#!/usr/bin/env python3
"""Run a resumable, sharded STRIVE evaluation from a fixed checkout."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime as dt
import json
import os
import pathlib
import subprocess
import sys
import threading
from typing import Iterable

from strive_vlm_runtime import VLMRuntime


ROOT = pathlib.Path(__file__).resolve().parents[1]
STRIVE = ROOT / "external" / "strive"
PYTHON = pathlib.Path(sys.executable)
JOBS_LOCK = threading.Lock()


def shard_ranges(start: int, stop: int, size: int) -> list[tuple[int, int]]:
    if start < 0 or stop <= start or size <= 0:
        raise ValueError("require 0 <= start < stop and shard_size > 0")
    return [(left, min(left + size, stop)) for left in range(start, stop, size)]


def read_metric_episode_ids(path: pathlib.Path) -> list[int]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return [int(row["Episode"]) for row in csv.DictReader(stream)]


def attempt_is_complete(path: pathlib.Path, start: int, stop: int) -> bool:
    try:
        exit_code = int((path / "exit_code.txt").read_text().strip())
    except (FileNotFoundError, ValueError):
        return False
    episode_ids = read_metric_episode_ids(path / "output" / "metrics.csv")
    accepted_postrun_abort = (path / "accepted_postrun_abort.txt").is_file()
    return (exit_code == 0 or accepted_postrun_abort) and episode_ids == list(
        range(start, stop)
    )


def _is_acceptable_open3d_postrun_abort(
    exit_code: int, console_path: pathlib.Path, episode_ids: list[int], start: int, stop: int
) -> bool:
    if exit_code != -6 or episode_ids != list(range(start, stop)):
        return False
    console_tail = console_path.read_text(encoding="utf-8", errors="replace")[-8192:]
    return "Cacher::~Cacher()" in console_tail and "leaking memory blocks on CUDA" in console_tail


def _git_head(path: pathlib.Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True
    ).strip()


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _append_jsonl(path: pathlib.Path, record: dict[str, object]) -> None:
    with JOBS_LOCK:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")


def _existing_success(
    shards_root: pathlib.Path, start: int, stop: int
) -> pathlib.Path | None:
    pattern = f"shard_{start:04d}_{stop - 1:04d}_attempt_*"
    for attempt in sorted(shards_root.glob(pattern)):
        if attempt_is_complete(attempt, start, stop):
            return attempt
    return None


def _next_attempt_path(
    shards_root: pathlib.Path, start: int, stop: int
) -> pathlib.Path:
    prefix = f"shard_{start:04d}_{stop - 1:04d}_attempt_"
    used = []
    for path in shards_root.glob(f"{prefix}*"):
        try:
            used.append(int(path.name.removeprefix(prefix)))
        except ValueError:
            continue
    return shards_root / f"{prefix}{max(used, default=0) + 1:03d}"


def _run_shard(
    suite_root: pathlib.Path,
    start: int,
    stop: int,
    av_nav_commit: str,
    strive_commit: str,
) -> dict[str, object]:
    shards_root = suite_root / "shards"
    previous = _existing_success(shards_root, start, stop)
    if previous is not None:
        return {
            "start": start,
            "stop": stop,
            "status": "skipped_complete",
            "attempt": previous.name,
        }

    if _git_head(ROOT) != av_nav_commit or _git_head(STRIVE) != strive_commit:
        raise RuntimeError("fixed checkout changed during evaluation")

    attempt = _next_attempt_path(shards_root, start, stop)
    attempt.mkdir(parents=True, exist_ok=False)
    (attempt / "av_nav_commit.txt").write_text(av_nav_commit + "\n")
    (attempt / "strive_commit.txt").write_text(strive_commit + "\n")
    (attempt / "vlm_runtime.json").write_text(
        json.dumps(VLMRuntime.from_env().public_dict(), indent=2) + "\n"
    )
    (attempt / "started_at.txt").write_text(_utc_now() + "\n")

    environment = os.environ.copy()
    environment["STRIVE_VLM_LOG"] = str(attempt / "vlm_calls.jsonl")
    environment["CUDA_VISIBLE_DEVICES"] = environment.get("STRIVE_GPU", "0")
    relative_output = f"../../../runs/{suite_root.name}/shards/{attempt.name}/output"
    command = [
        str(PYTHON),
        str(ROOT / "scripts" / "run_strive_upstream.py"),
        "--eval_episodes",
        str(stop),
        "--start_episode",
        str(start),
        "--save_dir",
        relative_output,
    ]
    console_path = attempt / "console.log"
    with console_path.open("w", encoding="utf-8") as console:
        code = subprocess.run(
            command,
            cwd=STRIVE,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=console,
            stderr=subprocess.STDOUT,
        ).returncode

    (attempt / "exit_code.txt").write_text(f"{code}\n")
    (attempt / "finished_at.txt").write_text(_utc_now() + "\n")
    episode_ids = read_metric_episode_ids(attempt / "output" / "metrics.csv")
    accepted_postrun_abort = _is_acceptable_open3d_postrun_abort(
        code, console_path, episode_ids, start, stop
    )
    if accepted_postrun_abort:
        (attempt / "accepted_postrun_abort.txt").write_text(
            "All requested metrics were written before the known Open3D CUDA cache destructor abort.\n"
        )
    complete = attempt_is_complete(attempt, start, stop)
    record: dict[str, object] = {
        "start": start,
        "stop": stop,
        "attempt": attempt.name,
        "returncode": code,
        "complete": complete,
        "metric_episode_ids": episode_ids,
        "accepted_postrun_abort": accepted_postrun_abort,
        "finished_at": _utc_now(),
    }
    (attempt / "status.json").write_text(json.dumps(record, indent=2) + "\n")
    _append_jsonl(suite_root / "jobs.jsonl", record)
    return record


def _run_shard_with_retries(
    suite_root: pathlib.Path,
    start: int,
    stop: int,
    av_nav_commit: str,
    strive_commit: str,
    max_attempts: int,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for attempt_number in range(1, max_attempts + 1):
        result = _run_shard(
            suite_root, start, stop, av_nav_commit, strive_commit
        )
        result["attempts_in_invocation"] = attempt_number
        if result.get("status") == "skipped_complete" or result.get(
            "complete", False
        ):
            return result
    return result


def _manifest_payload(args: argparse.Namespace) -> dict[str, object]:
    return {
        "suite_id": args.suite_id,
        "start": args.start,
        "stop": args.stop,
        "shard_size": args.shard_size,
        "workers": args.workers,
        "max_shard_attempts": args.max_shard_attempts,
        "av_nav_commit": _git_head(ROOT),
        "strive_commit": _git_head(STRIVE),
        "vlm": VLMRuntime.from_env().public_dict(),
        "created_at": _utc_now(),
    }


def _load_or_create_suite(args: argparse.Namespace) -> tuple[pathlib.Path, dict[str, object]]:
    suite_root = ROOT / "runs" / args.suite_id
    manifest_path = suite_root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        expected = {
            "suite_id": args.suite_id,
            "start": args.start,
            "stop": args.stop,
            "shard_size": args.shard_size,
            "workers": args.workers,
            "max_shard_attempts": args.max_shard_attempts,
        }
        actual = {key: manifest.get(key) for key in expected}
        if actual != expected:
            raise RuntimeError(f"resume arguments differ: {actual!r} != {expected!r}")
        if manifest.get("av_nav_commit") != _git_head(ROOT):
            raise RuntimeError("resume checkout differs from suite commit")
        if manifest.get("strive_commit") != _git_head(STRIVE):
            raise RuntimeError("resume STRIVE checkout differs from suite commit")
        return suite_root, manifest

    suite_root.mkdir(parents=True, exist_ok=False)
    (suite_root / "shards").mkdir()
    manifest = _manifest_payload(args)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return suite_root, manifest


def _ensure_preflight(suite_root: pathlib.Path) -> None:
    exit_path = suite_root / "preflight_exit_code.txt"
    if exit_path.exists():
        if int(exit_path.read_text().strip()) != 0:
            raise RuntimeError("suite preflight previously failed")
        return
    with (suite_root / "preflight.log").open("w", encoding="utf-8") as log:
        code = subprocess.run(
            [
                str(PYTHON),
                str(ROOT / "scripts" / "strive_preflight.py"),
                "--root",
                str(ROOT),
                "--output",
                str(suite_root / "preflight.json"),
            ],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
        ).returncode
    exit_path.write_text(f"{code}\n")
    if code != 0:
        raise RuntimeError(f"suite preflight failed with exit code {code}")
    with (suite_root / "environment.txt").open("w", encoding="utf-8") as stream:
        subprocess.run(
            [str(PYTHON), "-m", "pip", "freeze"],
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-id", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=1000)
    parser.add_argument("--shard-size", type=int, default=10)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-shard-attempts", type=int, default=3)
    args = parser.parse_args()
    if args.workers <= 0 or args.max_shard_attempts <= 0:
        parser.error("--workers and --max-shard-attempts must be positive")
    ranges = shard_ranges(args.start, args.stop, args.shard_size)
    suite_root, manifest = _load_or_create_suite(args)
    _ensure_preflight(suite_root)
    pending = [
        bounds
        for bounds in ranges
        if _existing_success(suite_root / "shards", *bounds) is None
    ]
    print(
        json.dumps(
            {
                "suite_root": str(suite_root),
                "total_shards": len(ranges),
                "pending_shards": len(pending),
                "workers": args.workers,
            }
        ),
        flush=True,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures: Iterable[concurrent.futures.Future[dict[str, object]]] = [
            pool.submit(
                _run_shard_with_retries,
                suite_root,
                start,
                stop,
                str(manifest["av_nav_commit"]),
                str(manifest["strive_commit"]),
                args.max_shard_attempts,
            )
            for start, stop in pending
        ]
        try:
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                print(json.dumps(result, sort_keys=True), flush=True)
                if result.get("status") != "skipped_complete" and not result.get(
                    "complete", False
                ):
                    for pending_future in futures:
                        pending_future.cancel()
                    break
        except Exception:
            for pending_future in futures:
                pending_future.cancel()
            raise

    completed = sum(
        _existing_success(suite_root / "shards", *bounds) is not None
        for bounds in ranges
    )
    final = {
        "finished_at": _utc_now(),
        "completed_shards": completed,
        "total_shards": len(ranges),
        "complete": completed == len(ranges),
    }
    (suite_root / "suite_status.json").write_text(json.dumps(final, indent=2) + "\n")
    print(json.dumps(final, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
