from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


DataFrameLoader = Callable[[Path], pd.DataFrame]
ModelBuilder = Callable[[], LogisticRegression]


@dataclass(frozen=True)
class FeatureSourceSpec:
    prefix: str
    columns: tuple[str, ...]
    filename: str | None = None
    loader: DataFrameLoader | None = None

    def load(self, data_dir: Path) -> pd.DataFrame:
        if self.loader is not None:
            return self.loader(data_dir)
        if self.filename is None:
            raise ValueError(f"Feature source {self.prefix} must define a filename or loader")
        return pd.read_csv(data_dir / self.filename)


@dataclass(frozen=True)
class ModelSpec:
    start_year: int
    builder: ModelBuilder


@dataclass(frozen=True)
class SplitEvaluation:
    pipe: Pipeline
    probs: np.ndarray
    preds: np.ndarray
    kept_features: list[str]
    accuracy: float
    by_year: dict[int, float]


@dataclass(frozen=True)
class ValidationSummary:
    results: list[tuple[int, float]]
    skipped: list[tuple[int, str]]
