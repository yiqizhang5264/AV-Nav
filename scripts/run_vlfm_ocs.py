"""Prepare or explicitly execute one frozen OCS online experiment.

Preparation uses no models/GPU. Execution reuses the lossless evidence overlay
and installs only exploration-selection hooks. This is not an auto-launch queue.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PINNED = "584ed56008754fde7997d904983607def8328322"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extend_overlay(overlay):
    trainer = Path(overlay.name) / "vlfm/utils/vlfm_trainer.py"
    source = trainer.read_text(encoding="utf-8")
    anchor = "from vlfm_evidence_recorder import get_evidence_recorder\n"
    if source.count(anchor) != 1:
        raise RuntimeError("Unexpected evidence-overlay import site")
    source = source.replace(anchor, anchor +
        "from av_nav.frontier_ocs_runtime import install_from_environment\n"
        "install_from_environment()\n")
    trainer.write_text(source, encoding="utf-8")
    return overlay


def git_head(root):
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def physical_identity(row):
    """Source IDs can be re-numbered by Habitat; compare task content instead."""
    return (str(row["scene_id"]).replace("\\", "/").split("/")[-1],
            str(row["object_category"]), tuple(float(v) for v in row["start_position"]),
            tuple(float(v) for v in row["start_rotation"]))


def verify_completed_identities(run, expected):
    paths = sorted((Path(run) / "evidence/episodes").glob("*/result.json"))
    observed = []
    for path in paths:
        metadata = json.loads((path.parent / "episode.json").read_text(encoding="utf-8"))
        result = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(result.get("metrics"), dict) or result.get("steps_recorded", 0) < 1:
            raise ValueError(f"Incomplete per-episode result: {path}")
        observed.append(metadata)
    match = Counter(map(physical_identity, observed)) == Counter(map(physical_identity, expected))
    return len(paths), match


def prepare(args):
    run, upstream = Path(args.run_dir).resolve(), Path(args.vlfm_root).resolve()
    dataset = Path(args.dataset).resolve()
    if run.exists():
        raise ValueError(f"Refusing existing run directory: {run}")
    if git_head(upstream) != PINNED:
        raise ValueError("Upstream VLFM does not match pinned commit")
    data = json.loads(gzip.decompress(dataset.read_bytes()))
    if not data.get("episodes"):
        raise ValueError("Expected a self-contained filtered task file with episodes")
    if "__no_external_content__" not in data.get("content_scenes_path", ""):
        raise ValueError("Filtered task file must disable loading additional scene content")
    settings = {"mode": args.mode, "anchor_match_radius_m": 0.5, "scheduler": {}}
    if args.settings:
        settings.update(json.loads(Path(args.settings).read_text(encoding="utf-8")))
        if settings["mode"] != args.mode:
            raise ValueError("Settings mode differs from requested mode")
    if args.mode == "ocs":
        from av_nav.frontier_ocs import OCSConfig
        from dataclasses import asdict
        settings["scheduler"] = asdict(OCSConfig(**settings.get("scheduler", {})))
    identities = []
    for index, row in enumerate(data["episodes"]):
        identity = {k: row[k] for k in ("scene_id", "object_category", "start_position", "start_rotation")}
        identity.update(source_episode_id=str(row["episode_id"]), filtered_index=index,
                        row_sha256=hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
        identities.append(identity)
        scene = Path(args.scenes_dir).resolve() / row["scene_id"]
        if not scene.is_file():
            raise ValueError(f"Training scene asset missing: {scene}")
    run.mkdir(parents=True)
    target_dataset = run / "hm3d_train_tasks.json.gz"
    shutil.copyfile(dataset, target_dataset)
    settings_path = run / "settings.json"
    settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    evidence = run / "evidence"
    environment = {"CUDA_VISIBLE_DEVICES": str(args.gpu), "PYTHONHASHSEED": str(args.seed),
                   "VLFM_EVIDENCE_DIR": str(evidence), "VLFM_OCS_SETTINGS": str(settings_path),
                   "VLFM_OCS_LOG": str(run / "ocs_events.jsonl"),
                   "ZSOS_LOG_DIR": str(run / "episode_logs"), "ZSOS_DONE_PATH": str(run / "UPSTREAM_DONE"),
                   "PYTHONPATH": os.pathsep.join((str(ROOT), str(ROOT / "scripts"), str(upstream))),
                   "GROUNDING_DINO_PORT": str(args.port_base), "BLIP2ITM_PORT": str(args.port_base + 1),
                   "SAM_PORT": str(args.port_base + 2), "YOLOV7_PORT": str(args.port_base + 3)}
    hydra = ["habitat_baselines.evaluate=true", "habitat_baselines.eval.video_option=[]",
             f"habitat.seed={args.seed}", "habitat_baselines.num_environments=1",
             "habitat_baselines.torch_gpu_id=0", "habitat.environment.max_episode_steps=500",
             "habitat.environment.iterator_options.shuffle=False",
             "habitat.environment.iterator_options.group_by_scene=False",
             f"habitat_baselines.test_episode_count={len(identities)}",
             "habitat_baselines.eval.split=train", "habitat.dataset.split=train",
             f"habitat.dataset.data_path={target_dataset}",
             f"habitat.dataset.scenes_dir={Path(args.scenes_dir).resolve()}",
             f"habitat_baselines.tensorboard_dir={run / 'tb'}", f"hydra.run.dir={run / 'hydra'}"]
    command = [sys.executable, str(Path(__file__).resolve()), "worker", "--vlfm-root", str(upstream), *hydra]
    source_paths = [Path(__file__).resolve(), ROOT / "av_nav/frontier_ocs_runtime.py",
                    ROOT / "scripts/run_vlfm_evidence.py", ROOT / "scripts/vlfm_evidence_adapter.py",
                    ROOT / "scripts/vlfm_evidence_recorder.py",
                    upstream / "vlfm/policy/itm_policy.py", upstream / "vlfm/policy/base_objectnav_policy.py",
                    upstream / "vlfm/policy/utils/acyclic_enforcer.py"]
    if args.mode == "ocs":
        source_paths.append(ROOT / "av_nav/frontier_ocs.py")
    manifest = {"schema": "vlfm.ocs.online_prepared.v1", "status": "prepared_not_started",
                "created_utc": datetime.now(timezone.utc).isoformat(), "mode": args.mode,
                "run_type": "ONLINE_ROLLOUT", "bugfix_applied": False,
                "av_nav_commit": git_head(ROOT), "upstream_commit": PINNED,
                "source_hashes": {str(p): sha(p) for p in source_paths},
                "dataset_path": str(target_dataset), "dataset_sha256": sha(target_dataset),
                "source_dataset_path": str(dataset), "settings_sha256": sha(settings_path),
                "identities": identities, "environment": environment, "command": command,
                "cwd": str(ROOT), "warning": "No online result yet; GPU/service readiness must be checked before execute."}
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def execute(args):
    run = Path(args.run_dir).resolve()
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    for path, digest in manifest["source_hashes"].items():
        if sha(path) != digest:
            raise ValueError(f"Frozen source changed: {path}")
    if sha(manifest["dataset_path"]) != manifest["dataset_sha256"]:
        raise ValueError("Frozen dataset changed")
    if sha(run / "settings.json") != manifest["settings_sha256"]:
        raise ValueError("Frozen settings changed")
    if (run / "started.json").exists():
        raise ValueError("Run already started; use a new run directory for every attempt")
    import socket
    for key in ("GROUNDING_DINO_PORT", "BLIP2ITM_PORT", "SAM_PORT", "YOLOV7_PORT"):
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", int(manifest["environment"][key]))) != 0:
                raise ValueError(f"Required service not listening: {key}")
    # Explicit user-side readiness attestation: this program never starts services
    # or steals a GPU. The read-only idle checker is run BEFORE loading services.
    if not args.resources_ready:
        raise ValueError("Execute requires --resources-ready after GPU and service checks")
    (run / "started.json").write_text(json.dumps({"started_utc": datetime.now(timezone.utc).isoformat(),
                                                  "actual_av_nav_commit": git_head(ROOT)}), encoding="utf-8")
    env = os.environ.copy()
    for name in ("AVNAV_ROOM_ONLINE_SETTINGS", "VLFM_SKIP_EPISODES_FILE", "AV_CONFIG", "AV_STRATEGY"):
        env.pop(name, None)
    env.update(manifest["environment"])
    with (run / "console.log").open("w", encoding="utf-8") as stream:
        result = subprocess.run(manifest["command"], cwd=manifest["cwd"], env=env,
                                stdout=stream, stderr=subprocess.STDOUT)
    completed_count, identities_match = verify_completed_identities(run, manifest["identities"])
    completion = {"returncode": result.returncode, "completed_result_files": completed_count,
                  "expected_episodes": len(manifest["identities"]),
                  "physical_identities_match": identities_match,
                  "complete": result.returncode == 0 and identities_match}
    (run / "completion.json").write_text(json.dumps(completion, indent=2), encoding="utf-8")
    print(json.dumps(completion, indent=2))
    raise SystemExit(0 if completion["complete"] else (result.returncode or 2))


def worker():
    import run_vlfm_evidence as runner
    original = runner.create_vlfm_evidence_overlay
    runner.create_vlfm_evidence_overlay = lambda *a, **kw: extend_overlay(original(*a, **kw))
    sys.argv = [sys.argv[0], *sys.argv[2:]]
    runner.main()


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        worker()
        return
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--mode", choices=("original", "anchor_only", "ocs"), required=True)
    prep.add_argument("--run-dir", required=True)
    prep.add_argument("--dataset", required=True)
    prep.add_argument("--vlfm-root", required=True)
    prep.add_argument("--scenes-dir", required=True)
    prep.add_argument("--settings")
    prep.add_argument("--gpu", required=True)
    prep.add_argument("--seed", type=int, default=17)
    prep.add_argument("--port-base", type=int, default=14181)
    execute_parser = sub.add_parser("execute")
    execute_parser.add_argument("--run-dir", required=True)
    execute_parser.add_argument("--resources-ready", action="store_true")
    args = p.parse_args()
    if args.action == "prepare":
        prepare(args)
    else:
        execute(args)


if __name__ == "__main__":
    main()
