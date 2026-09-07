"""VLFM coordinate convention: camera x forward, y left, z up; metres."""
from dataclasses import dataclass
import numpy as np


@dataclass
class Observation:
    center: np.ndarray
    points: np.ndarray
    position: np.ndarray
    area: float
    boundary: float
    depth_valid: float
    depth: float
    quality: float
    rgb: np.ndarray
    bbox: tuple
    step: int
    score: object = None


def observe(rgb, normalized_depth, mask, transform, min_depth, max_depth, fx, fy, step):
    mask = np.asarray(mask) > 0
    d = np.asarray(normalized_depth).squeeze()
    valid = mask & np.isfinite(d) & (d > 0) & (d < 0.95)
    if valid.sum() < 16:
        return None
    h, w = mask.shape
    v, u = np.where(valid)
    z = d[v, u] * (max_depth - min_depth) + min_depth
    lo, hi = np.quantile(z, [0.1, 0.9])
    keep = (z >= lo) & (z <= hi)
    v, u, z = v[keep], u[keep], z[keep]
    xyz = np.stack((z, -(u - w // 2) * z / fx, -(v - h // 2) * z / fy), axis=1)
    xyz = xyz @ transform[:3, :3].T + transform[:3, 3]
    xyz = xyz[::max(1, len(xyz) // 500)]
    rows, cols = np.where(mask)
    edge = (rows < max(1, h * .02)) | (rows >= h * .98) | (cols < max(1, w * .02)) | (cols >= w * .98)
    area = float(mask.mean())
    boundary = float(edge.mean())
    depth_valid = float(valid.sum() / mask.sum())
    quality = float((min(area / .05, 1) * (1 - boundary) * depth_valid) ** (1 / 3))
    pad = int(max(cols.max() - cols.min(), rows.max() - rows.min()) * .1)
    bbox = (max(0, int(cols.min()) - pad), max(0, int(rows.min()) - pad),
            min(w, int(cols.max()) + 1 + pad), min(h, int(rows.max()) + 1 + pad))
    return Observation(np.median(xyz, axis=0), xyz, transform[:2, 3].copy(), area,
                       boundary, depth_valid, float(np.median(z)), quality, rgb, bbox, step)


def adaptive_range(obs, cfg):
    # Area scaling is unreliable under truncation; use conservative fixed range.
    if not cfg.adaptive or obs.boundary > .15 or obs.depth_valid < .6:
        return cfg.fixed_range
    return float(np.clip(obs.depth * np.sqrt(obs.area / cfg.target_area), cfg.min_range, cfg.max_range))


def angle_delta(a, b):
    return (a - b + np.pi) % (2 * np.pi) - np.pi
