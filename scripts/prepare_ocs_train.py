"""Verify the existing train smoke tasks against source shards; never resample.

This creates a byte-identical, HM3D-named copy and a full-payload provenance
manifest. Optional Habitat dataset loading verifies actual runtime episode IDs
without instantiating a simulator or loading a visual model.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(row):
    return {key: row[key] for key in ("scene_id", "object_category", "start_position", "start_rotation")}


def validate_tasks(data, source_root, scenes_root):
    if "__no_external_content__" not in data.get("content_scenes_path", ""):
        raise ValueError("Filtered dataset must disable external scene-content loading")
    if not data.get("episodes"):
        raise ValueError("No tasks")
    records, cache = [], {}
    seen = set()
    for index, row in enumerate(data["episodes"]):
        if "/train/" not in row["scene_id"].replace("\\", "/"):
            raise ValueError("Non-train scene encountered")
        sig = canonical(identity(row))
        if sig in seen:
            raise ValueError("Duplicate physical task identity")
        seen.add(sig)
        asset = scenes_root / row["scene_id"]
        if not asset.is_file():
            raise FileNotFoundError(asset)
        shard = source_root / "content" / (Path(row["scene_id"]).name.replace(".basis.glb", "") + ".json.gz")
        if shard not in cache:
            cache[shard] = json.loads(gzip.decompress(shard.read_bytes()))
        original = cache[shard]
        matched = [i for i, source_row in enumerate(original["episodes"]) if canonical(source_row) == canonical(row)]
        if len(matched) != 1:
            raise ValueError(f"Expected one exact source payload, got {len(matched)}: {row['scene_id']}/{row['episode_id']}")
        goal_key = Path(row["scene_id"]).name + "_" + row["object_category"]
        if not original.get("goals_by_category", {}).get(goal_key):
            raise ValueError(f"Missing source goals: {goal_key}")
        if canonical(data.get("goals_by_category", {}).get(goal_key)) != canonical(original["goals_by_category"][goal_key]):
            raise ValueError(f"Goal payload changed: {goal_key}")
        records.append(dict(identity(row), source_episode_id=str(row["episode_id"]), filtered_index=index,
                            source_shard=str(shard), source_row_index=matched[0],
                            source_shard_sha256=digest(shard),
                            row_sha256=hashlib.sha256(canonical(row).encode()).hexdigest(),
                            goals_sha256=hashlib.sha256(canonical(data["goals_by_category"][goal_key]).encode()).hexdigest(),
                            asset_path=str(asset), asset_bytes=asset.stat().st_size))
    return records


def verify_loader(dataset_path, scenes_root, records):
    from omegaconf import OmegaConf
    from habitat.datasets.object_nav.object_nav_dataset import ObjectNavDatasetV1
    config = OmegaConf.create(dict(data_path=str(dataset_path), scenes_dir=str(scenes_root),
                                   split="train", content_scenes=["*"]))
    dataset = ObjectNavDatasetV1(config)
    if len(dataset.episodes) != len(records):
        raise ValueError("Habitat loader episode count differs from frozen tasks")
    remaining = set(range(len(records)))
    for episode in dataset.episodes:
        # Habitat may prepend scenes_dir and renumber ID; physical identity is authoritative.
        episode_scene = str(episode.scene_id).replace("\\", "/")
        candidates = [i for i in remaining if episode_scene.endswith(records[i]["scene_id"])
                      and episode.object_category == records[i]["object_category"]
                      and list(episode.start_position) == records[i]["start_position"]
                      and list(episode.start_rotation) == records[i]["start_rotation"]]
        if len(candidates) != 1:
            raise ValueError(f"Cannot bijectively identify loaded episode: {episode_scene}/{episode.episode_id}")
        i = candidates[0]
        records[i]["loaded_episode_id"] = str(episode.episode_id)
        records[i]["loaded_scene_id"] = episode_scene
        remaining.remove(i)
    if remaining:
        raise ValueError("Some selected tasks were not loaded")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--source-train", required=True)
    parser.add_argument("--scenes-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-manifest")
    parser.add_argument("--verify-loader", action="store_true")
    args = parser.parse_args()
    dataset, source, scenes = map(lambda p: Path(p).resolve(), (args.dataset, args.source_train, args.scenes_dir))
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise ValueError(f"Refusing to overwrite {output}")
    if source.name != "train":
        raise ValueError("Source shard directory must be train")
    data = json.loads(gzip.decompress(dataset.read_bytes()))
    if args.source_manifest:
        manifest = json.loads(Path(args.source_manifest).read_text(encoding="utf-8"))
        if digest(dataset) != manifest["sha256"]:
            raise ValueError("Source split SHA256 differs from original manifest")
    records = validate_tasks(data, source, scenes)
    output.mkdir(parents=True)
    target = output / "hm3d_train_smoke5.json.gz"
    shutil.copyfile(dataset, target)
    if args.verify_loader:
        verify_loader(target, scenes, records)
    frozen = dict(schema="vlfm.ocs.train_tasks.v1", source_dataset=str(dataset),
                  source_dataset_sha256=digest(dataset), dataset_path=str(target), dataset_sha256=digest(target),
                  split="train", selection_changed=False, count=len(records), loader_verified=args.verify_loader,
                  no_simulator_or_models_started=True, episodes=records)
    (output / "manifest.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    print(json.dumps(frozen, indent=2))


if __name__ == "__main__":
    main()
