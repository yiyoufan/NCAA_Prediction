from __future__ import annotations

import pandas as pd

from .config import ExperimentSettings


def dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


class FeatureSetBuilder:
    def __init__(self, settings: ExperimentSettings) -> None:
        self.settings = settings

    def build(self, df: pd.DataFrame) -> dict[str, list[str]]:
        base = [
            "SEED DIFF",
            "ABS SEED DIFF",
            "TEAM1 BETTER SEED",
            "SAME SEED",
            "ROUND LOG2",
            "CURRENT ROUND",
        ]
        notebook_3f = ["SEED DIFF", "bart_BADJ O_DIFF", "bart_BADJ D_DIFF"]
        ratings_safe = dedupe(base + self._diff_columns(df, self.settings.safe_diff_prefixes))
        ratings_plus = dedupe(
            ratings_safe
            + self._diff_columns(df, self.settings.plus_diff_prefixes)
            + [col for col in df.columns if col.startswith("loc_")]
        )
        return {
            "notebook_3f": notebook_3f,
            "base_round": base,
            "ratings_safe_l1": ratings_safe,
            "ratings_plus_l1": ratings_plus,
        }

    @staticmethod
    def _diff_columns(df: pd.DataFrame, prefixes: tuple[str, ...]) -> list[str]:
        return [
            col
            for col in df.columns
            if col.endswith("_DIFF") and any(col.startswith(f"{prefix}_") for prefix in prefixes)
        ]
