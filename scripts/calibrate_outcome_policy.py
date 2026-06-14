"""Calibrate the final outcome ensemble policy.

The result classifier and goal regressor each produce a Team-A-perspective
outcome distribution. This script sweeps blend weights on the World Cup
validation split and writes the best policy to model/artifacts/outcome_policy.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("STREAMLIT_LOG_LEVEL", "error")
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.caching.cache_data_api").disabled = True

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.copa_america import load_copa_america_data
from data.euro_2024 import load_euro_2024_data
from data.international_friendlies import append_friendlies_matches, load_friendlies_data
from data.ingest import load_award_winners, load_player_appearances, load_player_goals, load_rankings
from model import goals as goal_model
from model import train as result_train
from model.features import normalize_tournament_name
from model.predict import DEFAULT_OUTCOME_POLICY, OUTCOME_POLICY_PATH


def _normalize(probs: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probs, dtype=np.float64), 1e-6, 1.0)
    return clipped / clipped.sum(axis=1, keepdims=True)


def _goal_probabilities_for_entry(
    bundle: dict,
    rankings: dict,
    goal_matches: pd.DataFrame,
    team_a: str,
    team_b: str,
    tournament: str,
    match_date,
    venue: dict,
    mirrored_venue: dict,
) -> np.ndarray:
    history, h2h = goal_model._history_until(goal_matches, match_date)
    rows = [
        goal_model._build_goal_row(team_a, team_b, tournament, match_date, venue, history, h2h, rankings),
        goal_model._build_goal_row(team_b, team_a, tournament, match_date, mirrored_venue, history, h2h, rankings),
    ]
    x_goal = pd.DataFrame(rows, columns=bundle["feature_columns"])
    for col in bundle["categorical_columns"]:
        x_goal[col] = x_goal[col].fillna("unknown").astype(str)
    for col in bundle["numeric_columns"]:
        x_goal[col] = pd.to_numeric(x_goal[col], errors="coerce").fillna(0.0)

    predicted_a = np.clip(bundle["team_a_model"].predict(x_goal), 0.0, 10.0)
    predicted_b = np.clip(bundle["team_b_model"].predict(x_goal), 0.0, 10.0)
    pred_a = float((predicted_a[0] + predicted_b[1]) / 2.0)
    pred_b = float((predicted_b[0] + predicted_a[1]) / 2.0)
    scoreline = goal_model._scoreline_from_expected(pred_a, pred_b)
    outcome = scoreline["goal_outcome_probabilities"]
    return np.array([outcome["loss_prob"], outcome["draw_prob"], outcome["win_prob"]], dtype=np.float64)


def _load_result_training_context(meta: dict) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None, dict | None, dict | None, dict | None]:
    if result_train.USE_HISTORICAL_WEIGHTED:
        matches_df = result_train._load_historical_matches()
        goalscorers_df = result_train._load_historical_goalscorers()
        shootouts_df = result_train._load_historical_shootouts()
    else:
        from data.ingest import load_goalscorers, load_matches, load_shootouts

        matches_df = load_matches()
        goalscorers_df = load_goalscorers()
        shootouts_df = load_shootouts()

    rankings_df = load_rankings()
    copa_data = load_copa_america_data() if meta.get("include_copa", True) else None
    euro_data = load_euro_2024_data()
    friendlies_data = load_friendlies_data() if meta.get("include_friendlies", True) else None
    if result_train.USE_HISTORICAL_WEIGHTED and friendlies_data:
        matches_df = append_friendlies_matches(matches_df, friendlies_data)
    return matches_df, goalscorers_df, shootouts_df, rankings_df, copa_data, friendlies_data, euro_data


def _validation_source(matches_df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    validation_year = int(meta.get("validation_year", result_train.VALIDATION_YEAR))
    candidates = result_train._training_matches(
        matches_df,
        include_curated_friendlies=bool(meta.get("include_friendlies_train", False)),
        train_from_year=int(meta.get("train_from_year", 1872 if result_train.USE_HISTORICAL_WEIGHTED else result_train.TRAIN_FROM_YEAR)),
    )
    wc_mask = (
        candidates["_is_wc"].fillna(False)
        if "_is_wc" in candidates.columns
        else candidates["tournament"].str.contains("world cup", case=False, na=False)
    )
    validation = candidates[(candidates["date"].dt.year == validation_year) & wc_mask].copy()
    if validation.empty:
        raise RuntimeError(f"No World Cup validation rows found for {validation_year}.")
    return validation.sort_values("date").reset_index(drop=True)


def _validation_entries(
    validation_source: pd.DataFrame,
    model_rows: pd.DataFrame,
    model_dates: pd.Series,
) -> list[dict]:
    entries = []
    source = validation_source.copy()
    source["date"] = pd.to_datetime(source["date"])
    model_dates = pd.to_datetime(model_dates).reset_index(drop=True)
    model_rows = model_rows.reset_index(drop=True)

    for idx, row in model_rows.iterrows():
        team_a = row["team_a"]
        team_b = row["team_b"]
        match_date = pd.Timestamp(model_dates.iloc[idx])
        same_date = source[source["date"].eq(match_date)]
        candidates = same_date[
            (
                same_date["home_team"].eq(team_a)
                & same_date["away_team"].eq(team_b)
            )
            | (
                same_date["home_team"].eq(team_b)
                & same_date["away_team"].eq(team_a)
            )
        ]
        if candidates.empty:
            raise RuntimeError(f"Could not align validation goal row for {team_a} vs {team_b} on {match_date.date()}.")
        match = candidates.iloc[0]
        entries.append(
            {
                "team_a": team_a,
                "team_b": team_b,
                "date": match["date"],
                "tournament": match["tournament"],
                "venue": goal_model._venue_context(match, team_a, team_b),
                "mirrored_venue": goal_model._venue_context(match, team_b, team_a),
            }
        )
    return entries


def calibrate(write: bool = True) -> dict:
    meta_path = ROOT / "model" / "artifacts" / "meta.json"
    goal_model_path = ROOT / "model" / "artifacts" / "goals_model.pkl"
    with open(meta_path) as f:
        meta = json.load(f)

    matches_df, goalscorers_df, shootouts_df, rankings_df, copa_data, friendlies_data, euro_data = _load_result_training_context(meta)
    validation_source = _validation_source(matches_df, meta)
    train_source = result_train._training_matches(
        matches_df,
        include_curated_friendlies=bool(meta.get("include_friendlies_train", False)),
        train_from_year=int(meta.get("train_from_year", 1872 if result_train.USE_HISTORICAL_WEIGHTED else result_train.TRAIN_FROM_YEAR)),
    )
    train_source = train_source[train_source["date"].dt.year < int(meta.get("validation_year", result_train.VALIDATION_YEAR))].copy()
    if result_train.USE_HISTORICAL_WEIGHTED:
        train_source = result_train._select_historical_training_source(
            train_source,
            validation_year=int(meta.get("validation_year", result_train.VALIDATION_YEAR)),
        )
    skip_expensive = bool(meta.get("historical_weighted")) and bool(meta.get("historical_fast_features"))
    x_train, y_train, train_dates, train_weights = result_train.build_training_data(
        matches_df,
        goalscorers_df,
        shootouts_df,
        rankings_df,
        None,
        None if skip_expensive else load_player_appearances(),
        None if skip_expensive else load_player_goals(),
        None if skip_expensive else load_award_winners(),
        copa_data,
        friendlies_data,
        euro_data,
        include_curated_friendlies=bool(meta.get("include_friendlies_train", False)),
        source_matches_df=train_source,
        skip_goalscoring=skip_expensive,
        skip_supersub=skip_expensive,
        skip_player_stats=skip_expensive,
        return_weights=True,
        verbose=False,
    )
    x_test, y_test, test_dates = result_train.build_training_data(
        matches_df,
        goalscorers_df,
        shootouts_df,
        rankings_df,
        None,
        None if skip_expensive else load_player_appearances(),
        None if skip_expensive else load_player_goals(),
        None if skip_expensive else load_award_winners(),
        copa_data,
        friendlies_data,
        euro_data,
        include_curated_friendlies=bool(meta.get("include_friendlies_train", False)),
        source_matches_df=validation_source,
        skip_goalscoring=skip_expensive,
        skip_supersub=skip_expensive,
        skip_player_stats=skip_expensive,
        verbose=False,
    )

    threshold_calibration = result_train._calibrate_result_threshold_from_training(
        x_train,
        y_train,
        train_dates,
        train_weights,
        int(meta.get("validation_year", result_train.VALIDATION_YEAR)),
    )
    candidates = []
    multiclass_model = result_train._fit_multiclass_result_model(x_train, y_train, sample_weight=train_weights)
    candidates.append((multiclass_model, result_train._evaluate_result_model(multiclass_model, x_test, y_test)))
    if result_train.USE_TWO_STAGE_RESULT:
        two_stage_model = result_train._fit_two_stage_model(x_train, y_train, sample_weight=train_weights)
        two_stage_model["draw_threshold"] = threshold_calibration["threshold"]
        candidates.append((two_stage_model, result_train._evaluate_result_model(two_stage_model, x_test, y_test)))

    validation_model, validation_metrics = max(
        candidates,
        key=lambda item: (item[1]["macro_f1"], item[1]["accuracy"], item[1]["draw_f1"]),
    )
    classifier_probs = _normalize(result_train._predict_result_proba(validation_model, x_test))

    goal_bundle = joblib.load(goal_model_path)
    goal_matches = goal_model.load_goal_matches()
    goal_rankings = goal_model._ranking_index(rankings_df)
    entries = _validation_entries(validation_source, x_test, test_dates)
    labels = y_test
    goal_probs = _normalize(
        np.vstack(
            [
                _goal_probabilities_for_entry(
                    goal_bundle,
                    goal_rankings,
                    goal_matches,
                    entry["team_a"],
                    entry["team_b"],
                    normalize_tournament_name(entry["tournament"]),
                    entry["date"],
                    entry["venue"],
                    entry["mirrored_venue"],
                )
                for entry in entries
            ]
        )
    )

    rows = []
    for classifier_weight in np.round(np.arange(0.0, 1.0001, 0.05), 4):
        probs = _normalize((classifier_probs * classifier_weight) + (goal_probs * (1.0 - classifier_weight)))
        pred = probs.argmax(axis=1).astype(np.int32)
        rows.append(
            {
                "classifier_weight": float(classifier_weight),
                "goal_weight": float(1.0 - classifier_weight),
                "accuracy": float(accuracy_score(labels, pred)),
                "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
                "macro_f1": float(f1_score(labels, pred, average="macro", zero_division=0)),
                "log_loss": float(log_loss(labels, probs, labels=[0, 1, 2])),
            }
        )

    best = max(rows, key=lambda row: (row["macro_f1"], row["balanced_accuracy"], -row["log_loss"]))
    policy = {
        **DEFAULT_OUTCOME_POLICY,
        "policy_name": "calibrated_outcome_ensemble",
        "classifier_weight": round(float(best["classifier_weight"]), 4),
        "goal_weight": round(float(best["goal_weight"]), 4),
        "source": "world_cup_validation_sweep",
        "selection_policy": "highest_macro_f1_then_balanced_accuracy_then_lowest_log_loss",
        "validation_year": int(meta.get("validation_year", result_train.VALIDATION_YEAR)),
        "validation_rows": int(len(labels)),
        "metrics": {key: round(float(best[key]), 4) for key in ("accuracy", "balanced_accuracy", "macro_f1", "log_loss")},
        "result_validation_model": validation_model.get("model_name", "unknown"),
        "result_validation_metrics": {
            "accuracy": round(float(validation_metrics["accuracy"]), 4),
            "balanced_accuracy": round(float(validation_metrics["balanced_accuracy"]), 4),
            "macro_f1": round(float(validation_metrics["macro_f1"]), 4),
            "log_loss": round(float(validation_metrics["log_loss"]), 4),
        },
        "candidates": [
            {
                "classifier_weight": round(float(row["classifier_weight"]), 4),
                "goal_weight": round(float(row["goal_weight"]), 4),
                "accuracy": round(float(row["accuracy"]), 4),
                "balanced_accuracy": round(float(row["balanced_accuracy"]), 4),
                "macro_f1": round(float(row["macro_f1"]), 4),
                "log_loss": round(float(row["log_loss"]), 4),
            }
            for row in rows
        ],
    }
    if write:
        os.makedirs(os.path.dirname(OUTCOME_POLICY_PATH), exist_ok=True)
        with open(OUTCOME_POLICY_PATH, "w") as f:
            json.dump(policy, f, indent=2)
    return policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-write", action="store_true", help="Print the selected policy without writing it.")
    args = parser.parse_args()
    policy = calibrate(write=not args.no_write)
    print(json.dumps(policy, indent=2))


if __name__ == "__main__":
    main()
