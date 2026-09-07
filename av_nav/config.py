from dataclasses import dataclass, fields
import json
from pathlib import Path


@dataclass
class Config:
    strategy: str = "active"
    seed: int = 17
    low: float = -0.03
    high: float = 0.03
    min_quality: float = 0.35
    target_area: float = 0.08
    min_range: float = 1.0
    max_range: float = 2.5
    fixed_range: float = 1.5
    adaptive: bool = True
    surface_correction: bool = True
    diversity_weight: float = 1.0
    scale_weight: float = 1.0
    cost_weight: float = 0.6
    max_path: float = 3.0
    max_actions: int = 24
    max_rounds: int = 2
    max_episode_actions: int = 64
    arrival_radius: float = 0.2
    association_radius: float = 0.75
    independent_distance: float = 0.25
    cooldown_steps: int = 30
    directions: int = 8
    angles: tuple = (0.0, -30.0, 30.0)
    negatives: tuple = ("chair", "bed", "potted plant", "toilet", "tv", "couch", "table", "cabinet", "picture", "lamp")

    @classmethod
    def load(cls, path):
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        cfg = cls(**values)
        if cfg.strategy not in {"baseline", "single", "passive", "random", "nearest", "active"}:
            raise ValueError("Invalid strategy")
        if not (cfg.low < cfg.high and 0 < cfg.target_area < 1 and 0 < cfg.min_range <= cfg.max_range):
            raise ValueError("Invalid evidence thresholds or observation ranges")
        if min(cfg.max_actions, cfg.max_rounds, cfg.max_episode_actions, cfg.directions) < 1 or cfg.max_path <= 0:
            raise ValueError("Budgets and direction count must be positive")
        return cfg
