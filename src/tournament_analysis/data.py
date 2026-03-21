from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .types import FeatureSourceSpec


def split_paired_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return df.iloc[::2].reset_index(drop=True), df.iloc[1::2].reset_index(drop=True)


def merge_diff_features(
    games: pd.DataFrame, source: pd.DataFrame, cols: list[str], prefix: str
) -> pd.DataFrame:
    cols = [col for col in cols if col in source.columns]
    left = source[["YEAR", "TEAM NO"] + cols].copy()
    right = source[["YEAR", "TEAM NO"] + cols].copy()

    merged = games.merge(
        left,
        left_on=["YEAR", "TEAM NO 1"],
        right_on=["YEAR", "TEAM NO"],
        how="left",
    ).drop(columns=["TEAM NO"])
    merged = merged.rename(columns={col: f"{prefix}_{col}_1" for col in cols})

    merged = merged.merge(
        right,
        left_on=["YEAR", "TEAM NO 2"],
        right_on=["YEAR", "TEAM NO"],
        how="left",
    ).drop(columns=["TEAM NO"])
    merged = merged.rename(columns={col: f"{prefix}_{col}_2" for col in cols})

    for col in cols:
        merged[f"{prefix}_{col}_DIFF"] = merged[f"{prefix}_{col}_1"] - merged[f"{prefix}_{col}_2"]
    return merged


def load_ap_final(data_dir: Path) -> pd.DataFrame:
    ap = pd.read_csv(data_dir / "AP Poll Data.csv").sort_values(["YEAR", "TEAM NO", "WEEK"])
    ap_final = ap.groupby(["YEAR", "TEAM NO"], as_index=False).tail(1).copy()
    ap_final["AP_RANKED"] = ap_final["RANK?"].fillna(0)
    return ap_final


def build_location_features(data_dir: Path) -> pd.DataFrame:
    loc = pd.read_csv(data_dir / "Tournament Locations.csv")
    loc = loc.dropna(subset=["DISTANCE (MI)"]).reset_index(drop=True)
    loc1, loc2 = split_paired_rows(loc)

    loc_games = pd.DataFrame(
        {
            "YEAR": loc1["YEAR"],
            "TEAM NO 1": loc1["TEAM NO"],
            "TEAM NO 2": loc2["TEAM NO"],
            "loc_DISTANCE1": loc1["DISTANCE (MI)"],
            "loc_DISTANCE2": loc2["DISTANCE (MI)"],
            "loc_TZ1": loc1["TIME ZONES CROSSED VALUE"],
            "loc_TZ2": loc2["TIME ZONES CROSSED VALUE"],
        }
    )
    loc_games["loc_DISTANCE_DIFF"] = loc_games["loc_DISTANCE1"] - loc_games["loc_DISTANCE2"]
    loc_games["loc_DISTANCE_ABS"] = loc_games["loc_DISTANCE_DIFF"].abs()
    loc_games["loc_TZ_DIFF"] = loc_games["loc_TZ1"] - loc_games["loc_TZ2"]
    loc_games["loc_TZ_ABS"] = loc_games["loc_TZ_DIFF"].abs()
    return loc_games


def default_feature_source_specs() -> tuple[FeatureSourceSpec, ...]:
    return (
        FeatureSourceSpec(
            prefix="bart",
            filename="Barttorvik Away-Neutral.csv",
            columns=(
                "BADJ EM",
                "BADJ O",
                "BADJ D",
                "BARTHAG",
                "WIN%",
                "BADJ T",
                "EXP",
                "TALENT",
                "FT%",
                "OP FT%",
                "PPPO",
                "PPPD",
                "ELITE SOS",
                "WAB",
            ),
        ),
        FeatureSourceSpec(
            prefix="kp",
            filename="KenPom Barttorvik.csv",
            columns=("K TEMPO", "KADJ T", "K OFF", "KADJ O", "K DEF", "KADJ D", "KADJ EM"),
        ),
        FeatureSourceSpec(
            prefix="resume",
            filename="Resumes.csv",
            columns=(
                "NET RPI",
                "RESUME",
                "WAB RANK",
                "ELO",
                "B POWER",
                "Q1 W",
                "Q2 W",
                "Q1 PLUS Q2 W",
                "Q3 Q4 L",
                "PLUS 500",
                "R SCORE",
            ),
        ),
        FeatureSourceSpec(
            prefix="teamrank",
            filename="TeamRankings.csv",
            columns=("TR RATING", "SOS RATING", "LUCK RATING", "CONSISTENCY TR RATING"),
        ),
        FeatureSourceSpec(
            prefix="evan",
            filename="EvanMiya.csv",
            columns=(
                "O RATE",
                "D RATE",
                "RELATIVE RATING",
                "OPPONENT ADJUST",
                "PACE ADJUST",
                "TRUE TEMPO",
                "KILLSHOTS PER GAME",
                "KILL SHOTS CONCEDED PER GAME",
                "TOTAL KILL SHOTS",
                "TOTAL KILL SHOTS CONCEDED",
            ),
        ),
        FeatureSourceSpec(
            prefix="rppf",
            filename="RPPF Ratings.csv",
            columns=(
                "RPPF RATING",
                "NPB RATING",
                "RADJ O",
                "RADJ D",
                "RADJ EM",
                "R PACE",
                "R SOS",
                "STROE",
                "STRDE",
                "STREM",
            ),
        ),
        FeatureSourceSpec(
            prefix="kp_pre",
            filename="KenPom Preseason.csv",
            columns=(
                "PRESEASON KADJ EM RANK",
                "PRESEASON KADJ EM",
                "PRESEASON KADJ O",
                "PRESEASON KADJ D",
                "PRESEASON KADJ T",
                "KADJ EM RANK CHANGE",
                "KADJ EM CHANGE",
                "KADJ T CHANGE",
            ),
        ),
        FeatureSourceSpec(
            prefix="rppf_pre",
            filename="RPPF Preseason Ratings.csv",
            columns=(
                "RPPF PRESEASON RANK",
                "PRESEASON RPPF RATING",
                "RPPF RATING CHANGE RANK",
                "RPPF RATING CHANGE",
                "RPPF RATING RANK CHANGE RANK",
            ),
        ),
        FeatureSourceSpec(
            prefix="z",
            filename="Z Rating Teams.csv",
            columns=("Z RATING RANK", "SEED LIST", "Z RATING"),
        ),
        FeatureSourceSpec(
            prefix="shoot",
            filename="Shooting Splits.csv",
            columns=(
                "DUNKS FG%",
                "DUNKS SHARE",
                "DUNKS FG%D",
                "DUNKS D SHARE",
                "CLOSE TWOS FG%",
                "CLOSE TWOS SHARE",
                "CLOSE TWOS FG%D",
                "CLOSE TWOS D SHARE",
                "FARTHER TWOS FG%",
                "FARTHER TWOS SHARE",
                "FARTHER TWOS FG%D",
                "FARTHER TWOS D SHARE",
                "THREES FG%",
                "THREES SHARE",
                "THREES FG%D",
                "THREES D SHARE",
            ),
        ),
        FeatureSourceSpec(
            prefix="ap",
            columns=("AP VOTES", "AP RANK", "AP_RANKED"),
            loader=load_ap_final,
        ),
    )


class TournamentFeatureBuilder:
    def __init__(
        self,
        data_dir: Path,
        source_specs: tuple[FeatureSourceSpec, ...] | None = None,
    ) -> None:
        self.data_dir = data_dir
        self.source_specs = source_specs or default_feature_source_specs()

    def build_games(self) -> pd.DataFrame:
        matchups = pd.read_csv(self.data_dir / "Tournament Matchups.csv")
        played = matchups.dropna(subset=["SCORE"]).reset_index(drop=True)
        team1, team2 = split_paired_rows(played)

        games = pd.DataFrame(
            {
                "YEAR": team1["YEAR"],
                "CURRENT ROUND": team1["CURRENT ROUND"],
                "TEAM NO 1": team1["TEAM NO"],
                "TEAM 1": team1["TEAM"],
                "SEED 1": team1["SEED"],
                "TEAM NO 2": team2["TEAM NO"],
                "TEAM 2": team2["TEAM"],
                "SEED 2": team2["SEED"],
                "TEAM 1 WIN": (team1["SCORE"] > team2["SCORE"]).astype(int),
            }
        )
        games["SEED DIFF"] = games["SEED 1"] - games["SEED 2"]
        games["ABS SEED DIFF"] = games["SEED DIFF"].abs()
        games["TEAM1 BETTER SEED"] = (games["SEED 1"] < games["SEED 2"]).astype(int)
        games["SAME SEED"] = (games["SEED 1"] == games["SEED 2"]).astype(int)
        games["ROUND LOG2"] = np.log2(games["CURRENT ROUND"])
        return games

    def build(self) -> pd.DataFrame:
        full = self.build_games().copy()
        for spec in self.source_specs:
            full = merge_diff_features(full, spec.load(self.data_dir), list(spec.columns), spec.prefix)
        return full.merge(
            build_location_features(self.data_dir),
            on=["YEAR", "TEAM NO 1", "TEAM NO 2"],
            how="left",
        )
