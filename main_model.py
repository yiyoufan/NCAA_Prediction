"""
This is the main model for the project.
It contains a clean and efficient implementation of predicting the outcome of NCAA march madness games
Using existing data and a ML model.
The result of this output can be scaled to predict future NCAA games.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import LassoCV
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer


ACTUAL_COLS = [
    "YEAR",
    "TEAM NO 1",
    "TEAM 1",
    "SEED 1",
    "TEAM NO 2",
    "TEAM 2",
    "SEED 2",
    "CURRENT ROUND",
    "kp_pre_PRESEASON KADJ EM_DIFF",
    "rppf_RADJ EM_DIFF",
    "kp_pre_KADJ EM CHANGE_DIFF",
    "rppf_pre_RPPF RATING CHANGE RANK_DIFF",
    "kp_pre_PRESEASON KADJ O_DIFF",
    "z_SEED LIST_DIFF",
    "bart_TALENT_DIFF",
    "ap_AP VOTES_DIFF",
    "bart_BADJ O_DIFF",
    "rppf_RADJ O_DIFF",
    "shoot_FARTHER TWOS FG%_DIFF",
    "teamrank_CONSISTENCY TR RATING_DIFF",
    "SEED DIFF",
    "bart_WAB_DIFF",
    "WIN ALL FIVE PRIOR 1",
    "WIN ALL FIVE PRIOR 2",
    "TEAM 1 WIN",
]

SPARSE_MODEL_COLUMNS = [
    "rppf_pre_RPPF RATING CHANGE RANK_DIFF",
    "z_SEED LIST_DIFF",
    "ap_AP VOTES_DIFF",
]

IDENTIFIER_COLUMNS = ["YEAR", "TEAM NO 1", "TEAM 1", "TEAM NO 2", "TEAM 2"]
TARGET_COLUMN = "TEAM 1 WIN"
NAIVE_LOGREG_FEATURES = ["SEED DIFF", "CURRENT ROUND", "kp_pre_PRESEASON KADJ EM_DIFF"]
ROUND_LABELS = {
    64: "Round of 64",
    32: "Round of 32",
    16: "Sweet 16",
    8: "Elite 8",
    4: "Final Four",
    2: "National Championship",
}
TEAM_FEATURE_BUILD_MAP = {
    "kp_pre_PRESEASON KADJ EM_DIFF": "kp_pre_PRESEASON KADJ EM",
    "rppf_RADJ EM_DIFF": "rppf_RADJ EM",
    "kp_pre_KADJ EM CHANGE_DIFF": "kp_pre_KADJ EM CHANGE",
    "kp_pre_PRESEASON KADJ O_DIFF": "kp_pre_PRESEASON KADJ O",
    "bart_TALENT_DIFF": "bart_TALENT",
    "bart_BADJ O_DIFF": "bart_BADJ O",
    "rppf_RADJ O_DIFF": "rppf_RADJ O",
    "shoot_FARTHER TWOS FG%_DIFF": "shoot_FARTHER TWOS FG%",
    "teamrank_CONSISTENCY TR RATING_DIFF": "teamrank_CONSISTENCY TR RATING",
    "bart_WAB_DIFF": "bart_WAB",
}

DEFAULT_TRAINING_PATH = "aidata_clean_v2.csv"
DEFAULT_FUTURE_MATCHUP_PATH = "2026_actual_cols_v2.csv"
DEFAULT_TEAM_FEATURES_PATH = "2026_matched_team_features.csv"
DEFAULT_WIN_LOOKUP_PATH = "win_all_five_prior_lookup.csv"

TRAINING_BUNDLE = None
logistic_naive = None
svc_model = None
lasso_model = None
lr_model = None
sc_naive = None
model_pipelines = {}


def normalize_team_name(team_name):
    if pd.isna(team_name):
        return ""
    return "".join(ch for ch in str(team_name).lower() if ch.isalnum())


def sigmoid(values):
    clipped_values = np.clip(np.asarray(values, dtype=float), -50, 50)
    return 1.0 / (1.0 + np.exp(-clipped_values))


def get_round_label(round_value):
    return ROUND_LABELS.get(round_value, f"Round {round_value}")


def prepare_training_data(training_path=DEFAULT_TRAINING_PATH):
    df_complete = pd.read_csv(training_path)
    df_complete = df_complete[df_complete["CURRENT ROUND"] > 4].copy()

    df_complete2 = df_complete[ACTUAL_COLS].copy()
    df_complete2 = df_complete2[df_complete2["YEAR"] > 2011].copy()
    df_complete2.drop(columns=SPARSE_MODEL_COLUMNS, inplace=True)

    df_for_modeling = df_complete2.drop(columns=["TEAM NO 1", "TEAM 1", "TEAM NO 2", "TEAM 2"])
    df_for_training = df_for_modeling[df_for_modeling["YEAR"] < 2023].copy()
    df_for_testing = df_for_modeling[df_for_modeling["YEAR"] >= 2023].copy()

    X_train = df_for_training.drop(columns=["YEAR", TARGET_COLUMN])
    y_train = df_for_training[TARGET_COLUMN]
    X_test = df_for_testing.drop(columns=["YEAR", TARGET_COLUMN])
    y_test = df_for_testing[TARGET_COLUMN]

    return {
        "prepared_training_frame": df_complete2,
        "df_for_modeling": df_for_modeling,
        "df_for_training": df_for_training,
        "df_for_testing": df_for_testing,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }


def build_preprocessor(X_frame):
    numeric = [col for col in X_frame.columns if col != "CURRENT ROUND"]
    categorical = ["CURRENT ROUND"] if "CURRENT ROUND" in X_frame.columns else []

    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        ("sc", StandardScaler()),
                    ]
                ),
                numeric,
            ),
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


def load_models(training_path=DEFAULT_TRAINING_PATH):
    global TRAINING_BUNDLE
    global logistic_naive
    global svc_model
    global lasso_model
    global lr_model
    global sc_naive
    global model_pipelines

    bundle = prepare_training_data(training_path)
    X_train = bundle["X_train"]
    y_train = bundle["y_train"]

    sc_naive = StandardScaler()
    X_train_naive_scaled = sc_naive.fit_transform(X_train[NAIVE_LOGREG_FEATURES])

    logistic_naive = LogisticRegression(
        random_state=8,
        solver="liblinear",
        max_iter=6000,
        C=10,
    )
    logistic_naive.fit(X_train_naive_scaled, y_train)

    lr_model = LogisticRegression(
        random_state=8,
        solver="liblinear",
        max_iter=6000,
        C=10,
    )
    lasso_model = LogisticRegression(
        max_iter=1000,
        random_state=8,
        penalty="l1",
        solver="liblinear",
        C=2,
    )
    svc_model = SVC(
        random_state=8,
        C=5,
        kernel="linear",
    )

    model_pipelines = {}
    for model_name, model_object in {
        "LR": lr_model,
        "LASSO": lasso_model,
        "SVC": svc_model,
    }.items():
        pipeline = Pipeline(
            [
                ("preprocessor", build_preprocessor(X_train)),
                ("model", model_object),
            ]
        )
        pipeline.fit(X_train, y_train)
        model_pipelines[model_name] = pipeline

    bundle["logistic_naive"] = logistic_naive
    bundle["sc_naive"] = sc_naive
    bundle["lr_model"] = lr_model
    bundle["lasso_model"] = lasso_model
    bundle["svc_model"] = svc_model
    bundle["model_pipelines"] = model_pipelines
    bundle["historical_test_summary"] = evaluate_historical_test_split(bundle)

    TRAINING_BUNDLE = bundle
    return bundle


def evaluate_historical_test_split(bundle):
    X_test = bundle["X_test"]
    y_test = bundle["y_test"]

    if X_test.empty:
        return pd.DataFrame(columns=["Model", "Accuracy", "F1"])

    naive_scaled = bundle["sc_naive"].transform(X_test[NAIVE_LOGREG_FEATURES])
    prediction_map = {
        "Naive_LogReg": bundle["logistic_naive"].predict(naive_scaled),
        "LR": bundle["model_pipelines"]["LR"].predict(X_test),
        "LASSO": bundle["model_pipelines"]["LASSO"].predict(X_test),
        "SVC": bundle["model_pipelines"]["SVC"].predict(X_test),
    }

    summary_rows = []
    for model_name, predictions in prediction_map.items():
        summary_rows.append(
            {
                "Model": model_name,
                "Accuracy": round(accuracy_score(y_test, predictions), 4),
                "F1": round(f1_score(y_test, predictions, average="weighted"), 4),
            }
        )

    return pd.DataFrame(summary_rows)


def ensure_model_bundle(model_bundle=None):
    if model_bundle is not None:
        return model_bundle
    if TRAINING_BUNDLE is not None:
        return TRAINING_BUNDLE
    return load_models()


def prepare_matchup_dataframe(matchup_source):
    if isinstance(matchup_source, pd.DataFrame):
        matchup_frame = matchup_source.copy()
    else:
        matchup_frame = pd.read_csv(matchup_source)

    missing_columns = [col for col in ACTUAL_COLS if col not in matchup_frame.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    matchup_frame = matchup_frame[ACTUAL_COLS].copy()
    return matchup_frame


def get_model_feature_frame(matchup_frame):
    model_frame = matchup_frame.drop(columns=IDENTIFIER_COLUMNS + [TARGET_COLUMN], errors="ignore").copy()
    model_frame.drop(columns=SPARSE_MODEL_COLUMNS, inplace=True, errors="ignore")
    return model_frame


def get_prediction_components(model_bundle, model_frame):
    naive_scaled = model_bundle["sc_naive"].transform(model_frame[NAIVE_LOGREG_FEATURES])

    binary_predictions = {
        "Naive_LogReg": model_bundle["logistic_naive"].predict(naive_scaled).astype(int),
        "LR": model_bundle["model_pipelines"]["LR"].predict(model_frame).astype(int),
        "LASSO": model_bundle["model_pipelines"]["LASSO"].predict(model_frame).astype(int),
        "SVC": model_bundle["model_pipelines"]["SVC"].predict(model_frame).astype(int),
    }

    probability_estimates = {
        "Naive_LogReg": model_bundle["logistic_naive"].predict_proba(naive_scaled)[:, 1],
        "LR": model_bundle["model_pipelines"]["LR"].predict_proba(model_frame)[:, 1],
        "LASSO": model_bundle["model_pipelines"]["LASSO"].predict_proba(model_frame)[:, 1],
        "SVC": sigmoid(model_bundle["model_pipelines"]["SVC"].decision_function(model_frame)),
    }

    vote_matrix = np.column_stack(list(binary_predictions.values()))
    vote_share = vote_matrix.mean(axis=1)
    average_probability = np.column_stack(list(probability_estimates.values())).mean(axis=1)
    ensemble_prediction = np.where(
        vote_share > 0.5,
        1,
        np.where(vote_share < 0.5, 0, (average_probability >= 0.5).astype(int)),
    )

    return {
        "binary_predictions": binary_predictions,
        "probability_estimates": probability_estimates,
        "vote_share": vote_share,
        "average_probability": average_probability,
        "ensemble_prediction": ensemble_prediction.astype(int),
    }


def predict_matchups(matchup_source, model_bundle=None):
    model_bundle = ensure_model_bundle(model_bundle)
    matchup_frame = prepare_matchup_dataframe(matchup_source)

    if matchup_frame.empty:
        return matchup_frame.copy()

    model_frame = get_model_feature_frame(matchup_frame)
    prediction_components = get_prediction_components(model_bundle, model_frame)

    results = matchup_frame.copy()
    for model_name, probabilities in prediction_components["probability_estimates"].items():
        results[f"{model_name} TEAM 1 WIN CHANCE"] = np.round(probabilities, 4)
    for model_name, predictions in prediction_components["binary_predictions"].items():
        results[f"{model_name} PRED"] = predictions

    results["MODEL VOTES FOR TEAM 1"] = np.column_stack(
        list(prediction_components["binary_predictions"].values())
    ).sum(axis=1)
    results["TEAM 1 WIN CHANCE"] = np.round(prediction_components["average_probability"], 4)
    results["ENSEMBLE PRED"] = prediction_components["ensemble_prediction"]
    results["PREDICTED WINNER"] = np.where(
        results["ENSEMBLE PRED"] == 1,
        results["TEAM 1"],
        results["TEAM 2"],
    )
    results["PREDICTED WINNER CHANCE"] = np.round(
        np.where(
            results["ENSEMBLE PRED"] == 1,
            prediction_components["average_probability"],
            1 - prediction_components["average_probability"],
        ),
        4,
    )
    results["ROUND LABEL"] = results["CURRENT ROUND"].map(get_round_label)

    return results


def predict_game_from_dataset(
    team_1,
    team_2,
    matchup_path=DEFAULT_FUTURE_MATCHUP_PATH,
    current_round=None,
    model_bundle=None,
    team_features_path=DEFAULT_TEAM_FEATURES_PATH,
    win_lookup_path=DEFAULT_WIN_LOOKUP_PATH,
):
    matchup_frame = prepare_matchup_dataframe(matchup_path)

    norm_team_1 = normalize_team_name(team_1)
    norm_team_2 = normalize_team_name(team_2)

    direct_mask = (
        matchup_frame["TEAM 1"].map(normalize_team_name).eq(norm_team_1)
        & matchup_frame["TEAM 2"].map(normalize_team_name).eq(norm_team_2)
    )
    reverse_mask = (
        matchup_frame["TEAM 1"].map(normalize_team_name).eq(norm_team_2)
        & matchup_frame["TEAM 2"].map(normalize_team_name).eq(norm_team_1)
    )

    if current_round is not None:
        direct_mask &= matchup_frame["CURRENT ROUND"].eq(current_round)
        reverse_mask &= matchup_frame["CURRENT ROUND"].eq(current_round)

    if direct_mask.any():
        game_row = matchup_frame.loc[direct_mask].head(1)
        prediction = predict_matchups(game_row, model_bundle=model_bundle).iloc[0]
        prediction = prediction.to_dict()
        prediction["REQUESTED TEAM 1"] = team_1
        prediction["REQUESTED TEAM 2"] = team_2
        prediction["REQUESTED TEAM 1 WIN CHANCE"] = prediction["TEAM 1 WIN CHANCE"]
        prediction["REQUESTED WINNER"] = prediction["PREDICTED WINNER"]
        prediction["REQUESTED WINNER CHANCE"] = prediction["PREDICTED WINNER CHANCE"]
        return pd.Series(prediction)

    if reverse_mask.any():
        game_row = matchup_frame.loc[reverse_mask].head(1)
        prediction = predict_matchups(game_row, model_bundle=model_bundle).iloc[0]
        prediction = prediction.to_dict()

        requested_team_1_win_chance = round(1 - prediction["TEAM 1 WIN CHANCE"], 4)
        requested_winner = team_1 if requested_team_1_win_chance >= 0.5 else team_2
        requested_winner_chance = requested_team_1_win_chance if requested_winner == team_1 else round(1 - requested_team_1_win_chance, 4)

        prediction["REQUESTED TEAM 1"] = team_1
        prediction["REQUESTED TEAM 2"] = team_2
        prediction["REQUESTED TEAM 1 WIN CHANCE"] = requested_team_1_win_chance
        prediction["REQUESTED WINNER"] = requested_winner
        prediction["REQUESTED WINNER CHANCE"] = requested_winner_chance
        return pd.Series(prediction)

    if current_round is not None:
        tournament_years = matchup_frame["YEAR"].dropna().unique()
        if len(tournament_years) != 1:
            raise ValueError("The single-game builder expects a single tournament year per matchup file")

        _, feature_lookup = build_team_feature_lookup(team_features_path)
        feature_lookup = expand_team_feature_lookup_from_matchups(matchup_frame, feature_lookup)
        win_lookup = build_win_lookup(win_lookup_path)

        built_game = build_matchup_from_team_features(
            team_1,
            team_2,
            current_round,
            int(tournament_years[0]),
            feature_lookup,
            win_lookup,
        )
        prediction = predict_matchups(built_game, model_bundle=model_bundle).iloc[0].to_dict()
        prediction["REQUESTED TEAM 1"] = team_1
        prediction["REQUESTED TEAM 2"] = team_2
        prediction["REQUESTED TEAM 1 WIN CHANCE"] = prediction["TEAM 1 WIN CHANCE"]
        prediction["REQUESTED WINNER"] = prediction["PREDICTED WINNER"]
        prediction["REQUESTED WINNER CHANCE"] = prediction["PREDICTED WINNER CHANCE"]
        return pd.Series(prediction)

    raise ValueError(f"Could not find matchup for {team_1} vs {team_2} in {matchup_path}")


def build_team_profile(row_dict):
    profile = {
        "TEAM": row_dict["TEAM"],
        "TEAM NO": row_dict.get("TEAM NO", row_dict.get("INPUT_TEAM_NO", np.nan)),
        "SEED": int(row_dict["SEED"]),
    }

    for raw_column in TEAM_FEATURE_BUILD_MAP.values():
        profile[raw_column] = float(row_dict[raw_column])

    return profile


def build_team_feature_lookup(team_features_path):
    feature_frame = pd.read_csv(team_features_path)
    feature_lookup = {}

    for _, row in feature_frame.iterrows():
        row_dict = row.to_dict()
        profile = build_team_profile(row_dict)
        for name_column in ["TEAM", "INPUT_TEAM"]:
            if name_column in row_dict:
                normalized_name = normalize_team_name(row_dict[name_column])
                if normalized_name:
                    feature_lookup[normalized_name] = profile.copy()

    return feature_frame, feature_lookup


def build_win_lookup(win_lookup_path):
    try:
        lookup_frame = pd.read_csv(win_lookup_path)
    except FileNotFoundError:
        return {}

    win_lookup = {}
    for _, row in lookup_frame.iterrows():
        year = int(row["year"])
        team_key = normalize_team_name(row["team"])
        win_lookup[(year, team_key)] = int(row["win_all_five_prior"])

    return win_lookup


def get_team_feature_row(team_name, feature_lookup):
    normalized_name = normalize_team_name(team_name)
    if normalized_name not in feature_lookup:
        raise KeyError(f"Missing team feature row for {team_name}")
    return feature_lookup[normalized_name]


def expand_team_feature_lookup_from_matchups(matchup_frame, feature_lookup):
    expanded_lookup = {key: value.copy() for key, value in feature_lookup.items()}
    changed = True

    while changed:
        changed = False
        for _, row in matchup_frame.iterrows():
            team_1_key = normalize_team_name(row["TEAM 1"])
            team_2_key = normalize_team_name(row["TEAM 2"])
            team_1_known = team_1_key in expanded_lookup
            team_2_known = team_2_key in expanded_lookup

            if team_1_known and not team_2_known:
                known_profile = expanded_lookup[team_1_key]
                inferred_profile = {
                    "TEAM": row["TEAM 2"],
                    "TEAM NO": row["TEAM NO 2"],
                    "SEED": int(row["SEED 2"]),
                }
                for diff_column, raw_column in TEAM_FEATURE_BUILD_MAP.items():
                    inferred_profile[raw_column] = float(known_profile[raw_column]) - float(row[diff_column])

                expanded_lookup[team_2_key] = inferred_profile
                changed = True

            if team_2_known and not team_1_known:
                known_profile = expanded_lookup[team_2_key]
                inferred_profile = {
                    "TEAM": row["TEAM 1"],
                    "TEAM NO": row["TEAM NO 1"],
                    "SEED": int(row["SEED 1"]),
                }
                for diff_column, raw_column in TEAM_FEATURE_BUILD_MAP.items():
                    inferred_profile[raw_column] = float(known_profile[raw_column]) + float(row[diff_column])

                expanded_lookup[team_1_key] = inferred_profile
                changed = True

    return expanded_lookup


def get_win_all_five_value(year, team_name, win_lookup):
    return int(win_lookup.get((int(year), normalize_team_name(team_name)), 0))


def build_matchup_from_team_features(team_1, team_2, current_round, year, feature_lookup, win_lookup):
    team_1_row = get_team_feature_row(team_1, feature_lookup)
    team_2_row = get_team_feature_row(team_2, feature_lookup)

    matchup_row = {
        "YEAR": int(year),
        "TEAM NO 1": team_1_row.get("TEAM NO", team_1_row.get("INPUT_TEAM_NO", np.nan)),
        "TEAM 1": team_1_row["TEAM"],
        "SEED 1": int(team_1_row["SEED"]),
        "TEAM NO 2": team_2_row.get("TEAM NO", team_2_row.get("INPUT_TEAM_NO", np.nan)),
        "TEAM 2": team_2_row["TEAM"],
        "SEED 2": int(team_2_row["SEED"]),
        "CURRENT ROUND": int(current_round),
        "rppf_pre_RPPF RATING CHANGE RANK_DIFF": 0.0,
        "z_SEED LIST_DIFF": 0.0,
        "ap_AP VOTES_DIFF": 0.0,
        "SEED DIFF": int(team_1_row["SEED"]) - int(team_2_row["SEED"]),
        "WIN ALL FIVE PRIOR 1": get_win_all_five_value(year, team_1_row["TEAM"], win_lookup),
        "WIN ALL FIVE PRIOR 2": get_win_all_five_value(year, team_2_row["TEAM"], win_lookup),
        "TEAM 1 WIN": np.nan,
    }

    for diff_column, raw_column in TEAM_FEATURE_BUILD_MAP.items():
        matchup_row[diff_column] = float(team_1_row[raw_column]) - float(team_2_row[raw_column])

    return pd.DataFrame([matchup_row])[ACTUAL_COLS]


def pair_consecutive_winners(winners):
    if len(winners) % 2 != 0:
        raise ValueError("Winner list must contain an even number of teams")

    paired_matchups = []
    for index in range(0, len(winners), 2):
        paired_matchups.append((winners[index], winners[index + 1]))
    return paired_matchups


def find_candidate_matchup(round_candidates, team_1, team_2):
    if round_candidates is None or round_candidates.empty:
        return None

    norm_team_1 = normalize_team_name(team_1)
    norm_team_2 = normalize_team_name(team_2)

    direct_mask = (
        round_candidates["TEAM 1"].map(normalize_team_name).eq(norm_team_1)
        & round_candidates["TEAM 2"].map(normalize_team_name).eq(norm_team_2)
    )
    if direct_mask.any():
        return round_candidates.loc[direct_mask].head(1)[ACTUAL_COLS].copy()

    reverse_mask = (
        round_candidates["TEAM 1"].map(normalize_team_name).eq(norm_team_2)
        & round_candidates["TEAM 2"].map(normalize_team_name).eq(norm_team_1)
    )
    if reverse_mask.any():
        return round_candidates.loc[reverse_mask].head(1)[ACTUAL_COLS].copy()

    return None


def build_round_matchups(pairings, round_value, year, round_candidates=None, feature_lookup=None, win_lookup=None):
    matchup_rows = []
    for team_1, team_2 in pairings:
        candidate_matchup = find_candidate_matchup(round_candidates, team_1, team_2)
        if candidate_matchup is not None:
            matchup_rows.append(candidate_matchup)
            continue

        if feature_lookup is None:
            raise ValueError(
                f"Could not find a predefined matchup for {team_1} vs {team_2} in round {round_value}"
            )

        matchup_rows.append(
            build_matchup_from_team_features(
                team_1,
                team_2,
                round_value,
                year,
                feature_lookup,
                win_lookup or {},
            )
        )

    return pd.concat(matchup_rows, ignore_index=True)[ACTUAL_COLS]


def resolve_opening_round_branches(predicted_round):
    working_frame = predicted_round.copy()
    working_frame["ORIGINAL ORDER"] = np.arange(len(working_frame))

    group_columns = ["CURRENT ROUND", "TEAM NO 1", "TEAM 1", "SEED 1", "SEED 2"]
    resolved_rows = []
    for _, group in working_frame.groupby(group_columns, sort=False):
        if len(group) == 1:
            resolved_rows.append(group.iloc[[0]])
            continue

        # When a play-in slot has two possible opponents, keep the stronger opponent.
        # The stronger branch is the one that gives Team 1 the lower ensemble win chance.
        selected_row = group.sort_values(
            by=["TEAM 1 WIN CHANCE", "ORIGINAL ORDER"],
            ascending=[True, True],
        ).iloc[[0]]
        resolved_rows.append(selected_row)

    resolved_frame = pd.concat(resolved_rows, ignore_index=True)
    resolved_frame = resolved_frame.sort_values("ORIGINAL ORDER").drop(columns=["ORIGINAL ORDER"])
    return resolved_frame.reset_index(drop=True)


def simulate_tournament_bracket(
    matchup_path=DEFAULT_FUTURE_MATCHUP_PATH,
    team_features_path=DEFAULT_TEAM_FEATURES_PATH,
    win_lookup_path=DEFAULT_WIN_LOOKUP_PATH,
    model_bundle=None,
):
    model_bundle = ensure_model_bundle(model_bundle)
    future_matchups = prepare_matchup_dataframe(matchup_path)

    tournament_years = future_matchups["YEAR"].dropna().unique()
    if len(tournament_years) != 1:
        raise ValueError("The bracket simulator expects a single tournament year per matchup file")
    tournament_year = int(tournament_years[0])

    _, feature_lookup = build_team_feature_lookup(team_features_path)
    feature_lookup = expand_team_feature_lookup_from_matchups(future_matchups, feature_lookup)
    win_lookup = build_win_lookup(win_lookup_path)

    round_candidates = {
        round_value: future_matchups[future_matchups["CURRENT ROUND"] == round_value].copy()
        for round_value in sorted(future_matchups["CURRENT ROUND"].dropna().unique(), reverse=True)
    }

    all_round_predictions = {}

    opening_round_predictions = predict_matchups(round_candidates.get(64, pd.DataFrame(columns=ACTUAL_COLS)), model_bundle=model_bundle)
    all_round_predictions[64] = resolve_opening_round_branches(opening_round_predictions)

    next_round_sequence = [32, 16, 8, 4, 2]
    current_winners = all_round_predictions[64]["PREDICTED WINNER"].tolist()

    for round_value in next_round_sequence:
        pairings = pair_consecutive_winners(current_winners)
        round_matchups = build_round_matchups(
            pairings=pairings,
            round_value=round_value,
            year=tournament_year,
            round_candidates=round_candidates.get(round_value),
            feature_lookup=feature_lookup,
            win_lookup=win_lookup,
        )
        round_predictions = predict_matchups(round_matchups, model_bundle=model_bundle)
        all_round_predictions[round_value] = round_predictions
        current_winners = round_predictions["PREDICTED WINNER"].tolist()

    regional_champions = all_round_predictions[8]["PREDICTED WINNER"].tolist()
    finalists = all_round_predictions[4]["PREDICTED WINNER"].tolist()
    national_champion = all_round_predictions[2]["PREDICTED WINNER"].iloc[0]

    return {
        "round_predictions": all_round_predictions,
        "regional_champions": regional_champions,
        "finalists": finalists,
        "national_champion": national_champion,
    }


def print_round_summary(round_predictions):
    if round_predictions.empty:
        return

    print(f"\n{round_predictions['ROUND LABEL'].iloc[0]}")
    for _, row in round_predictions.iterrows():
        print(
            f"{row['TEAM 1']} vs {row['TEAM 2']} -> {row['PREDICTED WINNER']} "
            f"({row['PREDICTED WINNER CHANCE']:.2%})"
        )


def print_bracket_summary(bracket_results):
    for round_value in [64, 32, 16, 8, 4, 2]:
        print_round_summary(bracket_results["round_predictions"][round_value])

    print("\nRegional Champions:")
    for champion in bracket_results["regional_champions"]:
        print(f"- {champion}")

    print("\nNational Finalists:")
    for finalist in bracket_results["finalists"]:
        print(f"- {finalist}")

    print(f"\nPredicted National Champion: {bracket_results['national_champion']}")


def main():
    trained_bundle = load_models()

    print("Historical holdout summary (trained on seasons before 2023, tested on 2023-2025):")
    print(trained_bundle["historical_test_summary"].to_string(index=False))

    future_predictions = predict_matchups(DEFAULT_FUTURE_MATCHUP_PATH, model_bundle=trained_bundle)
    print("\nSample future-matchup predictions:")
    print(
        future_predictions[
            [
                "TEAM 1",
                "TEAM 2",
                "CURRENT ROUND",
                "PREDICTED WINNER",
                "PREDICTED WINNER CHANCE",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    bracket_results = simulate_tournament_bracket(model_bundle=trained_bundle)
    print_bracket_summary(bracket_results)


if __name__ == "__main__":
    main()
