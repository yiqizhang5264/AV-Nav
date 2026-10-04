"""Load the explicit SAP adapter, then run the pinned Habitat evaluator."""
import os
import runpy
import gzip
import json
from pathlib import Path
import numpy as np


def install():
    import vlfm.utils.episode_stats_logger as logger
    original = logger.log_episode_stats
    with gzip.open(os.environ['AV_DATASET_FILE'], 'rt') as stream:
        source = json.load(stream)['sap_source_identities']

    def log(episode_id, scene_id, infos):
        identity = source[int(episode_id)]
        if not scene_id.endswith(identity['scene_id']):
            raise RuntimeError('SAP source scene identity mismatch')
        try:
            failure = original(episode_id, scene_id, infos)
        except Exception:
            failure = 'unclassified'
        metrics = {}
        for key, value in infos.items():
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, (str,int,float,bool)):
                metrics[key] = None if isinstance(value,float) and not np.isfinite(value) else value
        row = dict(identity, runtime_episode_id=str(episode_id), failure_cause=failure, metrics=metrics)
        with (Path(os.environ['AV_RUN_DIR'])/'episodes.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        return failure

    logger.log_episode_stats = log

def main():
    install()
    if os.environ['SAP_VARIANT'] != 'baseline':
        import av_nav.sap_policy  # noqa: F401
    runpy.run_module('vlfm.run', run_name='__main__')


if __name__ == '__main__':
    main()
