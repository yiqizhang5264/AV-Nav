#!/usr/bin/env python3
"""Check the external STRIVE checkout and server-only runtime resources."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys


REQUIRED_ENV = (
    "GEMINI_API_KEY",
    "HABITAT_LAB_PATH",
    "SAM_CHECKPOINT",
    "GROUNDING_DINO_PATH",
    "GROUNDING_DINO_CHECKPOINT",
    "HM3D_DATA_PATH",
)

EXPECTED_DEPENDENCIES = {
    "HABITAT_LAB_PATH": "cb02f030655f9a475b379ec8d269979d9e17688d",
    "GROUNDING_DINO_PATH": "cfd5d3a985b0249de009b67d04f37263e11cdf3d",
}


def git_head(path: pathlib.Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def sha256(path: pathlib.Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    root = args.root.resolve()
    strive = root / "external" / "strive"
    values = {name: os.environ.get(name, "") for name in REQUIRED_ENV}
    checks: dict[str, object] = {
        "python": sys.version,
        "av_nav_commit": git_head(root),
        "strive_commit": git_head(strive),
        "expected_strive_commit": "1872d73b7db297705d251df73bf5f08ffed0d749",
        "environment": {name: bool(value) for name, value in values.items()},
    }

    file_vars = ("SAM_CHECKPOINT", "GROUNDING_DINO_CHECKPOINT")
    dir_vars = ("HABITAT_LAB_PATH", "GROUNDING_DINO_PATH", "HM3D_DATA_PATH")
    checks["paths"] = {
        **{name: pathlib.Path(values[name]).is_file() if values[name] else False for name in file_vars},
        **{name: pathlib.Path(values[name]).is_dir() if values[name] else False for name in dir_vars},
    }
    checks["dependency_commits"] = {
        name: {
            "actual": git_head(pathlib.Path(values[name])) if values[name] else None,
            "expected": expected,
        }
        for name, expected in EXPECTED_DEPENDENCIES.items()
    }
    hm3d_root = pathlib.Path(values["HM3D_DATA_PATH"]) if values["HM3D_DATA_PATH"] else pathlib.Path()
    expected_data = {
        "episode_dataset": hm3d_root / "objectnav_hm3d_v2" / "val" / "val.json.gz",
        "scene_config": hm3d_root
        / "scene_datasets"
        / "hm3d_v0.2"
        / "hm3d_annotated_basis.scene_dataset_config.json",
    }
    checks["hm3d"] = {
        name: {
            "path": str(path),
            "exists": path.is_file(),
            "sha256": sha256(path),
        }
        for name, path in expected_data.items()
    }
    checks["ready"] = (
        checks["strive_commit"] == checks["expected_strive_commit"]
        and all(checks["environment"].values())
        and all(checks["paths"].values())
        and all(
            item["actual"] == item["expected"]
            for item in checks["dependency_commits"].values()
        )
        and all(item["exists"] for item in checks["hm3d"].values())
    )

    rendered = json.dumps(checks, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite {args.output}")
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if checks["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
