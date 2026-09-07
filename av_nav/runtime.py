"""Non-policy instrumentation for pinned VLFM evaluation."""
import json
import os
from pathlib import Path
import numpy as np


def install():
    import vlfm.utils.episode_stats_logger as logger
    original = logger.log_episode_stats

    def log(episode_id, scene_id, infos):
        try:
            failure = original(episode_id, scene_id, infos)
        except Exception:
            failure = "unclassified"
        def scalar(value):
            if isinstance(value, np.generic):
                return value.item()
            return value
        metrics = {k:scalar(v) for k,v in infos.items()
                   if isinstance(v,(str,int,float,bool,np.generic))}
        result = dict(scene_id=scene_id, episode_id=str(episode_id),
                      failure_cause=failure, metrics=metrics)
        with (Path(os.environ["AV_RUN_DIR"])/"episodes.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, allow_nan=False)+"\n")
        return failure
    logger.log_episode_stats = log


def main():
    install()
    if os.environ.get("AV_STRATEGY") != "baseline":
        import av_nav.policy  # noqa: F401
    # Execute as __main__ so Hydra resolves upstream's relative config directory
    # as a filesystem path, rather than trying to import a Python package 'config'.
    import runpy
    runpy.run_module("vlfm.run", run_name="__main__")


if __name__ == "__main__":
    main()
