"""Compare baseline vs friendlies-augmented accuracy on a 2026 friendly holdout."""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from data.copa_america import load_copa_america_data
from data.ingest import (
    load_award_winners,
    load_goalscorers,
    load_matches,
    load_player_appearances,
    load_player_goals,
    load_rankings,
    load_shootouts,
    load_substitutions,
)
from data.international_friendlies import FRIENDLIES_TOURNAMENT, load_friendlies_data
from data.preprocess import build_feature_row
from model.features import FEATURE_COLUMNS, MODEL_INPUT_COLUMNS, RAW_CONTEXT_COLUMNS, normalize_tournament_name
from model.train import (
    _catboost_model,
    _evaluate_catboost,
    _fit_catboost,
    _label,
    _shootout_winners,
    _training_matches,
)

REPORT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts", "friendlies_impact.json")


def _context_row(team_a: str, team_b: str, tournament: str) -> dict:
    return {
        "team_a": team_a,
        "team_b": team_b,
        "tournament": normalize_tournament_name(tournament),
    }


def _build_xy(
    label: str,
    source_matches: pd.DataFrame,
    matches_df: pd.DataFrame,
    goalscorers_df,
    shootouts_df,
    rankings_df,
    substitutions_df,
    player_appearances_df,
    player_goals_df,
    award_winners_df,
    copa_data,
    friendlies_data,
    use_friendlies_features: bool,
) -> tuple[pd.DataFrame, np.ndarray]:
    rows = []
    labels = []
    shootout_winners = _shootout_winners(shootouts_df)
    total = len(source_matches)

    for idx, (_, row) in enumerate(source_matches.iterrows(), start=1):
        home = row["home_team"]
        away = row["away_team"]
        build_kwargs = dict(
            as_of_date=row["date"],
            matches_df=matches_df,
            goalscorers_df=goalscorers_df,
            shootouts_df=shootouts_df,
            rankings_df=rankings_df,
            substitutions_df=substitutions_df,
            player_appearances_df=player_appearances_df,
            player_goals_df=player_goals_df,
            award_winners_df=award_winners_df,
            copa_data=copa_data,
            friendlies_data=friendlies_data if use_friendlies_features else None,
            skip_goalscoring=False,
            skip_supersub=False,
            skip_player_stats=False,
        )

        feat_a, _ = build_feature_row(team_a=home, team_b=away, **build_kwargs)
        label_a = _label(row, home, shootout_winners)
        rows.append({**_context_row(home, away, row["tournament"]), **feat_a})
        labels.append(label_a)

        feat_b, _ = build_feature_row(team_a=away, team_b=home, **build_kwargs)
        rows.append({**_context_row(away, home, row["tournament"]), **feat_b})
        labels.append(2 - label_a)

        if idx % 25 == 0 or idx == total:
            print(f"{label}: {idx}/{total} source matches")

    X = pd.DataFrame(rows, columns=MODEL_INPUT_COLUMNS)
    for col in RAW_CONTEXT_COLUMNS:
        X[col] = X[col].fillna("unknown").astype(str)
    for col in FEATURE_COLUMNS:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)
    return X, np.array(labels, dtype=np.int32)


def _friendly_holdout_split(matches_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = _training_matches(matches_df, include_curated_friendlies=True)
    friendlies = candidates[candidates["tournament"] == FRIENDLIES_TOURNAMENT].sort_values("date").reset_index(drop=True)
    if len(friendlies) < 6:
        raise RuntimeError("Not enough curated friendly matches to run the friendlies impact evaluation.")

    cutoff = max(1, int(len(friendlies) * 0.8))
    if cutoff >= len(friendlies):
        cutoff = len(friendlies) - 1

    train_friendlies = friendlies.iloc[:cutoff].copy()
    test_friendlies = friendlies.iloc[cutoff:].copy()
    competitive = candidates[candidates["tournament"] != FRIENDLIES_TOURNAMENT].copy()
    train_source = pd.concat([competitive, train_friendlies], ignore_index=True).sort_values("date").reset_index(drop=True)
    test_source = test_friendlies.sort_values("date").reset_index(drop=True)
    return train_source, test_source


def run_friendlies_experiment(verbose: bool = True) -> dict:
    matches_df = load_matches(include_copa=True, include_friendlies=True)
    goalscorers_df = load_goalscorers()
    shootouts_df = load_shootouts()
    rankings_df = load_rankings()
    substitutions_df = load_substitutions()
    player_appearances_df = load_player_appearances()
    player_goals_df = load_player_goals()
    award_winners_df = load_award_winners()
    copa_data = load_copa_america_data()
    friendlies_data = load_friendlies_data()

    if matches_df is None:
        raise RuntimeError("matches.csv is required for the experiment.")

    train_source, test_source = _friendly_holdout_split(matches_df)

    results = {}
    for name, use_friendlies_features in [("baseline", False), ("augmented", True)]:
        X_train, y_train = _build_xy(
            f"{name} train",
            train_source,
            matches_df,
            goalscorers_df,
            shootouts_df,
            rankings_df,
            substitutions_df,
            player_appearances_df,
            player_goals_df,
            award_winners_df,
            copa_data,
            friendlies_data,
            use_friendlies_features,
        )
        X_test, y_test = _build_xy(
            f"{name} test",
            test_source,
            matches_df,
            goalscorers_df,
            shootouts_df,
            rankings_df,
            substitutions_df,
            player_appearances_df,
            player_goals_df,
            award_winners_df,
            copa_data,
            friendlies_data,
            use_friendlies_features,
        )
        model = _catboost_model()
        _fit_catboost(model, X_train, y_train)
        metrics = _evaluate_catboost(model, X_test, y_test)
        metrics["train_rows"] = int(len(X_train))
        metrics["test_rows"] = int(len(X_test))
        results[name] = metrics

    delta = {
        "accuracy": round(results["augmented"]["accuracy"] - results["baseline"]["accuracy"], 4),
        "balanced_accuracy": round(results["augmented"]["balanced_accuracy"] - results["baseline"]["balanced_accuracy"], 4),
        "log_loss": round(results["augmented"]["log_loss"] - results["baseline"]["log_loss"], 4),
    }
    summary = {
        "train_source_matches": int(len(train_source)),
        "test_source_matches": int(len(test_source)),
        "test_date_start": str(test_source["date"].min().date()),
        "test_date_end": str(test_source["date"].max().date()),
        "baseline": results["baseline"],
        "augmented": results["augmented"],
        "delta": delta,
    }

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run_friendlies_experiment(verbose=True)
