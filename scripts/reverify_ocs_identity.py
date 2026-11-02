"""Audit historical results with missing metadata; never rewrite original evidence.

Uses the real dataset loader plus episode-origin transforms recorded by the
running environment, not expected poses copied into supposedly observed fields.
No simulator/model/GPU is created. Writes a separate, exclusively new report.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def scene_name(scene):
    return str(scene).replace("\\", "/").split("/")[-1]


def canonical_target(value):
    # Existing Habitat/VLFM vocabulary mapping, not a new detector vocabulary.
    return {"sofa": "couch", "plant": "potted plant", "tv_monitor": "tv"}.get(value, value)


def physical_identity(ep):
    return (scene_name(ep["scene_id"]), ep["object_category"],
            tuple(ep["start_position"]), tuple(ep["start_rotation"]))


def check_episode(folder, expected, expected_transform):
    folder = Path(folder)
    meta, result = read(folder / "episode.json"), read(folder / "result.json")
    trace = rows(folder / "steps.jsonl")
    if not trace or [r["step"] for r in trace] != list(range(len(trace))):
        raise ValueError(f"Noncontiguous/empty steps: {folder.name}")
    if len(trace) != result.get("steps_recorded"):
        raise ValueError(f"Result/trace step count mismatch: {folder.name}")
    if trace[-1].get("done") is not True or any(r.get("done") is not False for r in trace[:-1]):
        raise ValueError(f"Missing/duplicate terminal transition: {folder.name}")
    if (scene_name(meta["scene_id"]), str(meta["episode_id"])) != (
            scene_name(expected["scene_id"]), str(expected["loaded_episode_id"])):
        raise ValueError(f"Runtime scene/loaded ID mismatch: {folder.name}")
    target = canonical_target(expected["object_category"])
    observed_targets = {canonical_target(r["policy_info"]["target_object"]) for r in trace}
    if observed_targets != {target}:
        raise ValueError(f"Runtime target mismatch: {folder.name}: {observed_targets}")
    transforms = np.asarray([r["info"]["top_down_map"]["tf_episodic_to_global"] for r in trace], dtype=float)
    if transforms.shape != (len(trace), 4, 4) or not np.isfinite(transforms).all():
        raise ValueError("Missing/malformed runtime episode-origin transforms")
    error = float(np.max(np.abs(transforms - expected_transform)))
    if not np.allclose(transforms, expected_transform, rtol=0, atol=1e-6):
        raise ValueError(f"Runtime origin transform differs from frozen task: max error={error}")
    # Full quaternion agreement is available for these upright-yaw tasks. For
    # arbitrary tilt, an XY heading transform alone cannot identify pitch/roll.
    rotation = np.asarray(expected["start_rotation"], dtype=float)
    if not np.allclose(rotation[[0, 2]], 0., rtol=0, atol=1e-12):
        raise ValueError("Non-upright task cannot be fully identified by heading transform")
    for key in ("success", "spl", "distance_to_goal", "stop_called"):
        if not np.isclose(float(result["metrics"][key]), float(trace[-1]["info"][key]), rtol=0, atol=1e-6):
            raise ValueError(f"Terminal/result metric mismatch: {key}")
    paths = [folder / n for n in ("episode.json", "result.json", "steps.jsonl")]
    return {"scene_id": meta["scene_id"], "loaded_episode_id": meta["episode_id"],
            "source_episode_id": expected["source_episode_id"],
            "object_category_from_runtime_steps": sorted(observed_targets)[0],
            "original_metadata_incomplete": meta.get("start_position") is None,
            "runtime_origin_transform": transforms[0].tolist(),
            "max_transform_abs_error": error, "steps": len(trace),
            "metrics": result["metrics"], "failure_cause": result.get("failure_cause"),
            "evidence_hashes": {str(p): sha(p) for p in paths},
            "identity_verified": True, "terminal_verified": True}


def load_tasks(manifest):
    from omegaconf import OmegaConf
    from habitat.datasets.object_nav.object_nav_dataset import ObjectNavDatasetV1
    from habitat.tasks.nav.nav import HeadingSensor
    from habitat.utils.geometry_utils import quaternion_from_coeff
    from frontier_exploration.utils.general_utils import habitat_to_xyz

    overrides = [s.split("=", 1)[1] for s in manifest["command"] if s.startswith("habitat.dataset.scenes_dir=")]
    if len(overrides) != 1:
        raise ValueError("Cannot identify frozen scene root")
    dataset = ObjectNavDatasetV1(OmegaConf.create(dict(data_path=manifest["dataset_path"],
        scenes_dir=overrides[0], split="train", content_scenes=["*"])))
    expected = {physical_identity(r): r for r in manifest["identities"]}
    if len(expected) != len(manifest["identities"]):
        raise ValueError("Duplicate expected physical identity")
    tasks = {}
    for ep in dataset.episodes:
        row = {k: getattr(ep, k) for k in ("scene_id", "object_category", "start_position", "start_rotation")}
        key = physical_identity(row)
        if key not in expected:
            raise ValueError("Loader physical task not in frozen manifest")
        row.update(source_episode_id=expected[key]["source_episode_id"], loaded_episode_id=str(ep.episode_id))
        runtime_key = (scene_name(ep.scene_id), str(ep.episode_id))
        if runtime_key in tasks:
            raise ValueError("Duplicate loader key")
        yaw = HeadingSensor._quat_to_xy_heading(None, quaternion_from_coeff(ep.start_rotation).inverse())[0]
        x, y, z = habitat_to_xyz(np.array(ep.start_position))
        tf = np.array([[np.cos(yaw), -np.sin(yaw), 0, x], [np.sin(yaw), np.cos(yaw), 0, y],
                       [0, 0, 1, z], [0, 0, 0, 1]])
        tasks[runtime_key] = (row, tf)
    if len(tasks) != len(expected):
        raise ValueError("Loader count mismatch")
    return tasks


def audit(run):
    run = Path(run).resolve()
    manifest = read(run / "manifest.json")
    if sha(manifest["dataset_path"]) != manifest["dataset_sha256"]:
        raise ValueError("Frozen task data changed")
    if sha(run / "settings.json") != manifest["settings_sha256"]:
        raise ValueError("Frozen method settings changed")
    for path, digest in manifest["source_hashes"].items():
        if sha(path) != digest:
            raise ValueError(f"Frozen run source changed: {path}")
    tasks = load_tasks(manifest)
    paths = sorted((run / "evidence/episodes").glob("*/result.json"))
    if len(paths) != len(tasks):
        raise ValueError("Result file count differs from loaded tasks")
    checked, seen = [], set()
    for path in paths:
        meta = read(path.parent / "episode.json")
        key = scene_name(meta["scene_id"]), str(meta["episode_id"])
        if key in seen or key not in tasks:
            raise ValueError("Duplicate/unexpected runtime identity")
        row, tf = tasks[key]
        checked.append(check_episode(path.parent, row, tf))
        seen.add(key)
    if seen != set(tasks):
        raise ValueError("Missing runtime tasks")
    if not (run / "UPSTREAM_DONE").is_file():
        raise ValueError("Upstream end-of-evaluation marker missing")
    return {"schema": "vlfm.ocs.identity_reverification.v1", "verified_utc": datetime.now(timezone.utc).isoformat(),
            "run_dir": str(run), "data_complete": True, "physical_identities_match": True,
            "verified_episodes": len(checked), "original_evidence_modified": False,
            "worker_returncode": None, "worker_returncode_note": "Not persisted by old runner; do not infer zero from results",
            "upstream_done_marker_present": True,
            "identity_basis": "Actual loader scene+ID, runtime target, runtime episode-origin transform and contiguous terminal trace",
            "position_rotation_tolerance": 1e-6,
            "source_sha256": sha(__file__), "run_manifest_sha256": sha(run / "manifest.json"),
            "dataset_sha256": manifest["dataset_sha256"], "episodes": checked}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError("Refusing to overwrite an existing re-verification")
    try:
        result = audit(args.run_dir)
    except Exception as exc:
        result = {"data_complete": False, "physical_identities_match": False,
                  "error": f"{type(exc).__name__}: {exc}", "run_dir": args.run_dir}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["data_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
