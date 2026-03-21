from __future__ import annotations

from pathlib import Path
import warnings

from .config import ExperimentSettings
from .data import TournamentFeatureBuilder
from .evaluation import ExperimentRunner
from .features import FeatureSetBuilder
from .modeling import ModelRegistry, top_nonzero_coefficients
from .reporting import (
    print_coefficients,
    print_model_summary,
    print_seed_baseline,
    print_threshold_summary,
)


def run_analysis(project_root: Path, settings: ExperimentSettings) -> None:
    warnings.filterwarnings("ignore")

    df = TournamentFeatureBuilder(project_root / "data").build()
    feature_sets = FeatureSetBuilder(settings).build(df)
    model_registry = ModelRegistry()
    runner = ExperimentRunner(settings, model_registry)

    print_seed_baseline(df, settings.holdout_start)

    selected_pipe = None
    for name, features in feature_sets.items():
        validation = runner.validation_results(df, name, features)
        holdout = runner.holdout_evaluation(df, name, features)
        print_model_summary(name, validation, holdout)

        if name == "ratings_plus_l1":
            selected_pipe = holdout.pipe

    threshold_score, threshold = runner.tune_threshold(df, "ratings_plus_l1", feature_sets["ratings_plus_l1"])
    tuned_holdout = runner.holdout_evaluation(
        df,
        "ratings_plus_l1",
        feature_sets["ratings_plus_l1"],
        threshold,
    )
    print_threshold_summary("ratings_plus_l1", threshold_score, threshold, tuned_holdout)

    if selected_pipe is not None:
        print_coefficients("ratings_plus_l1", top_nonzero_coefficients(selected_pipe))


def export_model_data(project_root: Path, output_path: Path | None = None) -> Path:
    warnings.filterwarnings("ignore")

    resolved_output = output_path or (project_root / "data.csv")
    df = TournamentFeatureBuilder(project_root / "data").build()
    df.to_csv(resolved_output, index=False)
    return resolved_output
