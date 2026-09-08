"""Pure helpers for historical action replay; no policy or metric changes."""
import re
import numpy as np


def episode_actions(text, scene, episode_id):
    heads = list(re.finditer(r'^=== Episode (.*?) \(scene (.*?)\) - num (\d+) ===\s*$', text, re.M))
    matches = []
    for i, h in enumerate(heads):
        if h[1] == str(episode_id) and h[2].split('/')[-1].split('.')[0] == scene:
            body = text[h.end():heads[i+1].start() if i+1 < len(heads) else len(text)]
            matches.append(re.findall(r'Step: (\d+) \| Mode: (.*?) \| Action: (\d+)', body))
    if len(matches) != 1 or not matches[0]:
        raise ValueError('Missing, empty or duplicate scene/episode log')
    records = [(int(s), mode, int(a)) for s, mode, a in matches[0]]
    if [s for s, _, _ in records] != list(range(len(records))):
        raise ValueError('Non-contiguous action trace')
    if any(a not in (0, 1, 2, 3, 4, 5) for _, _, a in records):
        raise ValueError('Unknown action')
    return records


def camera_point(u, v, depth, width=640, height=480, hfov=79):
    """Habitat OpenGL optical frame: right/up/back; depth is forward Z distance."""
    if not np.isfinite(depth) or depth <= 0:
        raise ValueError('Invalid depth')
    focal = width / (2 * np.tan(np.deg2rad(hfov) / 2))
    return np.array([(u-width/2)*depth/focal, -(v-height/2)*depth/focal, -depth])
