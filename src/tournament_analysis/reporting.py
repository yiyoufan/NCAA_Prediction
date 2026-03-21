from __future__ import annotations

import numpy as np
import pandas as pd

from .types import SplitEvaluation, ValidationSummary


def print_seed_baseline(df: pd.DataFrame, holdout_start: int) -> None:
    seed_baseline = df[df["YEAR"] >= holdout_start].copy()
    seed_baseline_pred = (seed_baseline["SEED 1"] < seed_baseline["SEED 2"]).astype(int)
    print(f"Seed baseline holdout accuracy: {(seed_baseline_pred == seed_baseline['TEAM 1 WIN']).mean():.4f}")
    print()


def print_model_summary(name: str, validation: ValidationSummary, holdout: SplitEvaluation) -> None:
    print(name)
    if validation.results:
        print(
            "  validation mean:",
            f"{np.mean([score for _, score in validation.results]):.4f}",
            "|",
            ", ".join(f"{year}:{score:.4f}" for year, score in validation.results),
        )
    else:
        print("  validation mean: n/a | no usable validation years")

    if validation.skipped:
        skipped = ", ".join(f"{year} ({reason})" for year, reason in validation.skipped)
        print("  skipped validation years:", skipped)

    print(
        "  holdout @0.50:",
        f"{holdout.accuracy:.4f}",
        "| by year:",
        holdout.by_year,
        "| feature count:",
        len(holdout.kept_features),
    )
    print()


def print_threshold_summary(
    name: str,
    threshold_score: float,
    threshold: float,
    holdout: SplitEvaluation,
) -> None:
    print(f"{name} threshold tuning")
    print("  best validation threshold:", f"{threshold:.2f}", "| validation mean:", f"{threshold_score:.4f}")
    print("  holdout @tuned threshold:", f"{holdout.accuracy:.4f}", "| by year:", holdout.by_year)
    print()


def print_coefficients(name: str, coefficients: list[tuple[str, float]]) -> None:
    print(f"{name} top non-zero coefficients")
    for feature_name, coef in coefficients:
        print(f"  {coef:+.4f} {feature_name}")
