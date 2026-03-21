from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .types import ModelSpec, SplitEvaluation


class ModelRegistry:
    def __init__(self) -> None:
        self._specs = {
            "notebook_3f": ModelSpec(start_year=2008, builder=self._build_default_model),
            "base_round": ModelSpec(start_year=2008, builder=self._build_default_model),
            "ratings_safe_l1": ModelSpec(start_year=2015, builder=self._build_sparse_model),
            "ratings_plus_l1": ModelSpec(start_year=2008, builder=self._build_sparse_model),
        }

    def create(self, name: str) -> LogisticRegression:
        return self.require(name).builder()

    def start_year(self, name: str) -> int:
        return self.require(name).start_year

    def require(self, name: str) -> ModelSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise KeyError(f"Unknown model spec: {name}") from exc

    @staticmethod
    def _build_default_model() -> LogisticRegression:
        return LogisticRegression(max_iter=6000, C=0.1)

    @staticmethod
    def _build_sparse_model() -> LogisticRegression:
        return LogisticRegression(max_iter=6000, C=0.1, penalty="l1", solver="liblinear")


def usable_feature_columns(train_df: pd.DataFrame, features: list[str]) -> list[str]:
    return [col for col in features if col in train_df.columns and not train_df[col].isna().all()]


def split_issue(train_df: pd.DataFrame, test_df: pd.DataFrame, features: list[str]) -> str | None:
    if train_df.empty:
        return "no training rows before this year"
    if test_df.empty:
        return "no games found for this year"
    if not usable_feature_columns(train_df, features):
        return "no usable feature columns in training data"
    if train_df["TEAM 1 WIN"].nunique() < 2:
        return "training data has only one target class"
    return None


def fit_pipeline(
    train_df: pd.DataFrame, test_df: pd.DataFrame, features: list[str], model: LogisticRegression
) -> tuple[Pipeline, np.ndarray, list[str]]:
    issue = split_issue(train_df, test_df, features)
    if issue is not None:
        raise ValueError(f"Cannot fit split: {issue}")

    x_train = train_df[features].copy()
    x_test = test_df[features].copy()
    keep = usable_feature_columns(train_df, features)
    x_train = x_train[keep]
    x_test = x_test[keep]

    numeric = [col for col in keep if col != "CURRENT ROUND"]
    categorical = ["CURRENT ROUND"] if "CURRENT ROUND" in keep else []
    preprocessor = ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), numeric),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )

    pipe = Pipeline([("pre", preprocessor), ("model", model)])
    pipe.fit(x_train, train_df["TEAM 1 WIN"])
    probs = pipe.predict_proba(x_test)[:, 1]
    return pipe, probs, keep


def accuracy_by_year(test_df: pd.DataFrame, preds: np.ndarray) -> dict[int, float]:
    years = test_df["YEAR"].to_numpy()
    actual = test_df["TEAM 1 WIN"].to_numpy()
    return {
        int(year): accuracy_score(actual[years == year], preds[years == year])
        for year in sorted(test_df["YEAR"].unique())
    }


def evaluate_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    model: LogisticRegression,
    threshold: float,
) -> SplitEvaluation:
    pipe, probs, keep = fit_pipeline(train_df, test_df, features, model)
    preds = (probs >= threshold).astype(int)
    return SplitEvaluation(
        pipe=pipe,
        probs=probs,
        preds=preds,
        kept_features=keep,
        accuracy=accuracy_score(test_df["TEAM 1 WIN"], preds),
        by_year=accuracy_by_year(test_df, preds),
    )


def top_nonzero_coefficients(pipe: Pipeline, limit: int = 15) -> list[tuple[str, float]]:
    feature_names = pipe.named_steps["pre"].get_feature_names_out()
    coefs = pipe.named_steps["model"].coef_[0]
    nonzero = [(name, coef) for name, coef in zip(feature_names, coefs) if abs(coef) > 1e-8]
    nonzero.sort(key=lambda item: abs(item[1]), reverse=True)
    return nonzero[:limit]
