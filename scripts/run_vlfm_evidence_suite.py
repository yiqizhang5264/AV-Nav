#!/usr/bin/env python3
"""Run original VLFM scene shards with passive, resumable evidence capture."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import pathlib
import subprocess
import sys
from datetime import datetime, timezone


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        comparable = ("dataset", "av_nav_commit", "vlfm_commit", "dataset_root", "scenes_dir", "manifest_sha256", "content")
        if any(existing.get(key) != manifest.get(key) for key in comparable):
            raise SystemExit(f"existing run manifest does not match requested evaluation: {manifest_path}")
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
        for attempt_index in range(1, args.max_attempts + 1):
            attempt = shard / "attempts" / f"{attempt_index:03d}"
            if attempt.exists():
                continue
            attempt.mkdir(parents=True)
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
                f"habitat.dataset.data_path={dataset_root}/{{split}}/{{split}}.json.gz",
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
        if not done.exists():
            raise SystemExit(f"scene {scene} failed after {args.max_attempts} independent attempts")


if __name__ == "__main__":
    main()
