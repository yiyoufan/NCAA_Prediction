from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ExperimentSettings
from .modeling import ModelRegistry, evaluate_split, split_issue
from .types import SplitEvaluation, ValidationSummary


class ExperimentRunner:
    def __init__(self, settings: ExperimentSettings, model_registry: ModelRegistry | None = None) -> None:
        self.settings = settings
        self.model_registry = model_registry or ModelRegistry()

    def validation_results(
        self,
        df: pd.DataFrame,
        name: str,
        features: list[str],
        threshold: float | None = None,
    ) -> ValidationSummary:
        applied_threshold = self.settings.default_threshold if threshold is None else threshold
        results = []
        skipped = []
        start_year = self.model_registry.start_year(name)

        for year in self.settings.validation_years:
            train = df[(df["YEAR"] >= start_year) & (df["YEAR"] < year)]
            test = df[df["YEAR"] == year]
            issue = split_issue(train, test, features)
            if issue is not None:
                skipped.append((year, issue))
                continue
            evaluation = evaluate_split(train, test, features, self.model_registry.create(name), applied_threshold)
            results.append((year, evaluation.accuracy))

        return ValidationSummary(results=results, skipped=skipped)

    def holdout_evaluation(
        self,
        df: pd.DataFrame,
        name: str,
        features: list[str],
        threshold: float | None = None,
    ) -> SplitEvaluation:
        applied_threshold = self.settings.default_threshold if threshold is None else threshold
        start_year = self.model_registry.start_year(name)
        train = df[(df["YEAR"] >= start_year) & (df["YEAR"] < self.settings.holdout_start)]
        test = df[df["YEAR"] >= self.settings.holdout_start]
        return evaluate_split(train, test, features, self.model_registry.create(name), applied_threshold)

    def tune_threshold(self, df: pd.DataFrame, name: str, features: list[str]) -> tuple[float, float]:
        baseline_validation = self.validation_results(df, name, features)
        if not baseline_validation.results:
            skipped = ", ".join(f"{year} ({reason})" for year, reason in baseline_validation.skipped)
            raise ValueError(f"No usable validation years for {name}. Skipped: {skipped}")

        scores: list[tuple[float, float]] = []
        for threshold in self.settings.threshold_grid:
            validation = self.validation_results(df, name, features, float(threshold))
            yearly_scores = [score for _, score in validation.results]
            scores.append((float(np.mean(yearly_scores)), float(threshold)))
        return max(scores)
