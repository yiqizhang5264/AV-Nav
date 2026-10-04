#!/usr/bin/env python3
"""Run original VLFM scene shards with passive, resumable evidence capture."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
from datetime import datetime, timezone


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def episode_identity(episode: dict) -> str:
    fields = {
        "scene_id": str(episode.get("scene_id", "unknown")),
        "episode_id": str(episode.get("episode_id", "unknown")),
        "object_category": str(episode.get("object_category", "unknown")),
        "start_position": episode.get("start_position"),
        "start_rotation": episode.get("start_rotation"),
    }
    return json.dumps(fields, sort_keys=True, separators=(",", ":"))


def completed_episode_metadata(shard: pathlib.Path) -> list[dict]:
    completed = {}
    for result in shard.glob("attempts/*/evidence/episodes/*/result.json"):
        metadata = result.parent / "episode.json"
        if metadata.exists():
            record = json.loads(metadata.read_text(encoding="utf-8"))
            completed[episode_identity(record)] = record
    return list(completed.values())


def resolve_completed_identities(episodes: list[dict], completed_metadata: list[dict]) -> set[str]:
    """Resolve recorder metadata to source rows without guessing repeated IDs."""
    resolved = set()
    for record in completed_metadata:
        candidates = [episode for episode in episodes if str(episode.get("episode_id")) == str(record.get("episode_id"))]
        has_full_identity = (
            record.get("start_position") is not None
            and record.get("start_rotation") is not None
            and record.get("object_category") not in (None, "unknown")
        )
        if has_full_identity:
            candidates = [episode for episode in candidates if episode_identity(episode) == episode_identity(record)]
        if len(candidates) != 1:
            raise RuntimeError(
                f"cannot uniquely resume episode_id={record.get('episode_id')!r}: {len(candidates)} source rows match"
            )
        resolved.add(episode_identity(candidates[0]))
    return resolved


def prepare_remaining_dataset(
    dataset_root: pathlib.Path,
    scene: str,
    completed_metadata: list[dict],
    destination: pathlib.Path,
) -> tuple[pathlib.Path, int, int]:
    """Create an attempt-local val split containing only unfinished episodes."""
    source_content = dataset_root / "val" / "content" / f"{scene}.json.gz"
    with gzip.open(source_content, "rt", encoding="utf-8") as stream:
        data = json.load(stream)
    original_count = len(data["episodes"])
    completed = resolve_completed_identities(data["episodes"], completed_metadata)
    data["episodes"] = [episode for episode in data["episodes"] if episode_identity(episode) not in completed]
    remaining_count = len(data["episodes"])
    content_dir = destination / "val" / "content"
    content_dir.mkdir(parents=True)
    shutil.copy2(dataset_root / "val" / "val.json.gz", destination / "val" / "val.json.gz")
    content_path = content_dir / f"{scene}.json.gz"
    with content_path.open("wb") as raw_stream:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_stream, mtime=0) as stream:
            stream.write(json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    derivation = {
        "source_content": str(source_content),
        "source_sha256": sha256(source_content),
        "completed_episode_identities": sorted(completed),
        "original_count": original_count,
        "remaining_count": remaining_count,
    }
    (destination / "derivation.json").write_text(json.dumps(derivation, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination, original_count, remaining_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("hm3dv1", "hm3dv2", "mp3d"))
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--scenes-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--limit-scenes", type=int)
    parser.add_argument("--episodes-per-scene", type=int, default=-1)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--resume-note")
    args = parser.parse_args()

    repo = pathlib.Path(__file__).resolve().parents[1]
    vlfm_root = repo / "external" / "vlfm"
    dataset_root = pathlib.Path(args.dataset_root).resolve()
    output_root = pathlib.Path(args.output_root).resolve()
    content = sorted((dataset_root / "val" / "content").glob("*.json.gz"))
    if args.limit_scenes is not None:
        content = content[: args.limit_scenes]
    if not content:
        raise SystemExit(f"no content files under {dataset_root / 'val' / 'content'}")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": args.dataset,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "av_nav_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "vlfm_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=vlfm_root, text=True).strip(),
        "dataset_root": str(dataset_root),
        "scenes_dir": str(pathlib.Path(args.scenes_dir).resolve()),
        "manifest_sha256": sha256(dataset_root / "val" / "val.json.gz"),
        "content": [{"scene": p.name[:-8], "sha256": sha256(p)} for p in content],
        "capture": ["rgb_png", "depth_float32_npy", "depth_uint16_png", "sensor_npz", "raw_and_filtered_detections_json",
                    "boxed_png", "all_detection_crops_png", "raw_rgb_mp4", "detections_mp4", "combined_vlfm_mp4",
                    "sam_masks", "vqa_calls", "itm_calls", "steps_jsonl", "maps_npy", "episode_metrics", "console_log"],
    }
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        comparable = ("dataset", "vlfm_commit", "dataset_root", "scenes_dir", "manifest_sha256", "content")
        if any(existing.get(key) != manifest.get(key) for key in comparable):
            raise SystemExit(f"existing run manifest does not match requested evaluation: {manifest_path}")
        if existing.get("av_nav_commit") != manifest.get("av_nav_commit"):
            if not args.resume_note:
                raise SystemExit("adapter commit changed; --resume-note is required")
            event = {
                "resumed_at": datetime.now(timezone.utc).isoformat(),
                "previous_av_nav_commit": existing.get("av_nav_commit"),
                "new_av_nav_commit": manifest.get("av_nav_commit"),
                "vlfm_commit": manifest.get("vlfm_commit"),
                "note": args.resume_note,
            }
            with (output_root / "resume_events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        manifest = existing
    else:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["PYTHONPATH"] = os.pathsep.join((str(repo / "scripts"), str(vlfm_root), env.get("PYTHONPATH", "")))
    for index, item in enumerate(manifest["content"]):
        scene = item["scene"]
        shard = output_root / "shards" / f"{index:03d}_{scene}"
        done = shard / "DONE"
        if done.exists():
            continue
        shard.mkdir(parents=True, exist_ok=True)
        completed = completed_episode_metadata(shard)
        for attempt_index in range(1, args.max_attempts + 1):
            attempt = shard / "attempts" / f"{attempt_index:03d}"
            if attempt.exists():
                continue
            attempt.mkdir(parents=True)
            attempt_dataset = dataset_root
            if completed:
                attempt_dataset, original_count, remaining_count = prepare_remaining_dataset(
                    dataset_root, scene, completed, attempt / "remaining_dataset"
                )
                if remaining_count == 0:
                    done.write_text(f"all {original_count} episodes completed across prior attempts\n", encoding="utf-8")
                    break
            evidence = attempt / "evidence"
            logs = attempt / "episode_logs"
            hydra_dir = attempt / "hydra"
            attempt_done = attempt / "DONE"
            env["VLFM_EVIDENCE_DIR"] = str(evidence)
            env["ZSOS_LOG_DIR"] = str(logs)
            env["ZSOS_DONE_PATH"] = str(attempt_done)
            command = [
                sys.executable, str(repo / "scripts" / "run_vlfm_evidence.py"), "--vlfm-root", str(vlfm_root),
                "habitat_baselines.evaluate=true", "habitat_baselines.eval.video_option=[disk]",
                f"habitat_baselines.video_dir={attempt / 'combined_videos'}",
                f"habitat_baselines.tensorboard_dir={attempt / 'tb'}",
                f"habitat_baselines.test_episode_count={args.episodes_per_scene}",
                "habitat_baselines.num_environments=1", "habitat_baselines.torch_gpu_id=0",
                f"habitat.dataset.data_path='{attempt_dataset}/{{split}}/{{split}}.json.gz'",
                f"habitat.dataset.scenes_dir={pathlib.Path(args.scenes_dir).resolve()}",
                f"habitat.dataset.content_scenes=[{scene}]", f"hydra.run.dir={hydra_dir}",
            ]
            (attempt / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
            with (attempt / "console.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
            (attempt / "exit_code.txt").write_text(str(result.returncode) + "\n", encoding="utf-8")
            if result.returncode == 0 and attempt_done.exists():
                done.write_text(f"attempts/{attempt_index:03d}\n", encoding="utf-8")
                break
            completed = completed_episode_metadata(shard)
        if not done.exists():
            raise SystemExit(f"scene {scene} failed after {args.max_attempts} independent attempts")


if __name__ == "__main__":
    main()
