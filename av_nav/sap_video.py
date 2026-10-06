"""Display-only video adapter and immutable source-to-video index for SAP."""
import json
import os
from pathlib import Path
import numpy as np


def bounded_pixels(pixels, shape):
    pixels = np.asarray(pixels)
    valid = ((pixels[:, 0] >= 0) & (pixels[:, 0] < shape[0]) &
             (pixels[:, 1] >= 0) & (pixels[:, 1] < shape[1]))
    return pixels[valid]


def install_video(source, video_saved):
    import vlfm.utils.habitat_visualizer as visualizer
    import vlfm.utils.vlfm_trainer as trainer

    def color_cloud(infos, policy_info):
        cloud = policy_info[0]['target_point_cloud']
        if len(cloud) == 0:
            return
        top = infos[0]['top_down_map']
        global_xyz = visualizer.transform_points(top['tf_episodic_to_global'], cloud[:, :3])
        xy = visualizer.xyz_to_habitat(global_xyz)[:, [2, 0]]
        pixels = visualizer.sim_xy_to_grid_xy(top['upper_bound'], top['lower_bound'],
                                             top['grid_resolution'], xy)
        image = top['map'].copy()
        pixels = bounded_pixels(pixels, image.shape)
        image[pixels[:, 0], pixels[:, 1]] = visualizer.MAP_TARGET_POINT_INDICATOR
        top['map'] = image

    # Only the display map is bounded; navigation coordinates are untouched.
    visualizer.color_point_cloud_on_map = color_cloud
    original = trainer.generate_video

    def generate(*args, **kwargs):
        folder = Path(kwargs['video_dir'])
        before = set(folder.glob('*.mp4'))
        result = original(*args, **kwargs)
        identity = source[int(kwargs['episode_id'])]
        files = sorted(set(folder.glob('*.mp4')) - before)
        if not files:
            raise RuntimeError('Video encoding returned without a new MP4')
        run = Path(os.environ['AV_RUN_DIR'])
        with (run/'videos.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(identity, runtime_episode_id=str(kwargs['episode_id']),
                videos=[str(p.relative_to(run)) for p in files],
                frames=len(kwargs['images']), fps=kwargs['fps']))+'\n')
        video_saved(kwargs['episode_id'])
        return result

    trainer.generate_video = generate
