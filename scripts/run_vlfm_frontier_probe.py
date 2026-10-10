"""Run one unchanged VLFM episode with extended decision/geometry instrumentation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from frontier_probe_recorder import install_environment


def extend_overlay(overlay):
    root = Path(overlay.name) / "vlfm"
    trainer = root / "utils/vlfm_trainer.py"
    source = trainer.read_text(encoding="utf-8")
    anchor = "from vlfm_evidence_recorder import get_evidence_recorder\n"
    if source.count(anchor) != 1:
        raise RuntimeError("unexpected evidence trainer imports")
    source = source.replace(anchor, anchor +
        "from frontier_probe_recorder import record_environment, select_episode\n"
        "from frontier_probe_install import install_policy_hooks\n"
        "install_policy_hooks()\n")
    anchor = "        observations = skip_completed_initial_episodes(self.envs, observations)\n"
    if source.count(anchor) != 1:
        raise RuntimeError("unexpected initial reset hook")
    source = source.replace(anchor, "        observations = select_episode(self.envs, observations)\n")
    anchor = "                evidence_recorder.record_observation(evidence_episode, observations[evidence_i])\n"
    if source.count(anchor) != 1:
        raise RuntimeError("unexpected observation hook")
    source = source.replace(anchor, anchor + "            record_environment(self.envs)\n")
    trainer.write_text(source, encoding="utf-8")
    base = root / "policy/base_objectnav_policy.py"
    source = base.read_text(encoding="utf-8")
    anchor = "        self._policy_info.update(self._get_policy_info(detections[0]))\n"
    if source.count(anchor) != 1:
        raise RuntimeError("unexpected policy frame capture site")
    source = source.replace(anchor, anchor +
        "        from frontier_probe_recorder import record_policy_frame\n"
        "        record_policy_frame(self, mode)\n")
    base.write_text(source, encoding="utf-8")
    return overlay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-case", required=True)
    args, remaining = parser.parse_known_args()
    if os.environ.get("VLFM_SKIP_EPISODES_FILE") or os.environ.get("AVNAV_ROOM_ONLINE_SETTINGS"):
        raise RuntimeError("probe must not be combined with resume or room interventions")
    case_path = Path(args.probe_case).resolve()
    case = json.loads(case_path.read_text(encoding="utf-8"))
    if "identity" not in case:
        raise ValueError("probe case requires full original episode identity")
    root = Path(os.environ["VLFM_EVIDENCE_DIR"])
    root.mkdir(parents=True, exist_ok=False)
    (root / "probe_case.json").write_text(json.dumps(case, indent=2), encoding="utf-8")
    os.environ["VLFM_FRONTIER_CASE"] = str(case_path)
    # Exactly one full-length episode; never silently use a shortened budget.
    for arg in remaining:
        if arg.startswith("habitat_baselines.test_episode_count=") and arg.split("=", 1)[1] != "1":
            raise ValueError("probe requires test_episode_count=1")
    remaining.append("habitat_baselines.test_episode_count=1")
    install_environment()
    import run_vlfm_evidence as runner
    original = runner.create_vlfm_evidence_overlay
    runner.create_vlfm_evidence_overlay = lambda *a, **kw: extend_overlay(original(*a, **kw))
    sys.argv = [sys.argv[0], *remaining]
    runner.main()


if __name__ == "__main__":
    main()
