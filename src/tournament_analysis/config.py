from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np


SAFE_DIFF_PREFIXES = ("bart", "kp", "resume", "teamrank", "evan", "rppf", "z")
PLUS_DIFF_PREFIXES = ("kp_pre", "rppf_pre", "shoot", "ap")
DEFAULT_THRESHOLD_GRID = tuple(float(value) for value in np.arange(0.35, 0.661, 0.01))


@dataclass(frozen=True)
class ExperimentSettings:
    validation_years: list[int]
    holdout_start: int
    default_threshold: float = 0.5
    threshold_grid: Sequence[float] = field(default_factory=lambda: DEFAULT_THRESHOLD_GRID)
    safe_diff_prefixes: tuple[str, ...] = SAFE_DIFF_PREFIXES
    plus_diff_prefixes: tuple[str, ...] = PLUS_DIFF_PREFIXES
