"""Dataset ablation audit for the World Cup predictor."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

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
from model.features import MODEL_INPUT_COLUMNS
from model.train import (
    RANDOM_STATE,
    _catboost_model,
    _evaluate_catboost,
    _fit_catboost,
    _training_matches,
    build_training_data,
)


ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
REPORT_PATH = os.path.join(ARTIFACTS_DIR, "dataset_audit.json")
SOURCE_CAP = int(os.getenv("ML_PRJCT_AUDIT_SOURCE_CAP", "900"))
MODEL_ITERATIONS = int(os.getenv("ML_PRJCT_AUDIT_ITERATIONS", "300"))
WORLD_CUP_YEARS = [2022]


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    include_copa: bool
    include_friendlies: bool
    include_friendlies_train: bool
    use_copa_features: bool
    use_friendlies_features: bool


CONFIGS = [
    DatasetConfig("base_open_2006", False, False, False, False, False),
    DatasetConfig("base_plus_copa", True, False, False, True, False),
    DatasetConfig("current_all_without_friendlies_target", True, True, False, True, True),
    DatasetConfig("current_all_with_friendlies_target", True, True, True, True, True),
]


def _model():
    model = _catboost_model()
    model.set_params(iterations=MODEL_ITERATIONS)
    return model


def _sample_train_source(source: pd.DataFrame) -> pd.DataFrame:
    if SOURCE_CAP <= 0 or len(source) <= SOURCE_CAP:
        return source.sort_values("date").reset_index(drop=True)

    wc_mask = source["tournament"].str.contains("world cup", case=False, na=False)
    wc_rows = source[wc_mask]
    other = source[~wc_mask]
    remaining = max(0, SOURCE_CAP - len(wc_rows))
    sampled_other = other.sample(n=min(len(other), remaining), random_state=RANDOM_STATE)
    return pd.concat([wc_rows, sampled_other], ignore_index=True).sort_values("date").reset_index(drop=True)


def _load_bundle(config: DatasetConfig) -> dict:
    return {
        "matches_df": load_matches(include_copa=config.include_copa, include_friendlies=config.include_friendlies),
        "goalscorers_df": load_goalscorers(),
        "shootouts_df": load_shootouts(),
        "rankings_df": load_rankings(),
        "substitutions_df": load_substitutions(),
        "player_appearances_df": load_player_appearances(),
        "player_goals_df": load_player_goals(),
        "award_winners_df": load_award_winners(),
        "copa_data": load_copa_america_data() if config.use_copa_features else None,
        "friendlies_data": load_friendlies_data() if config.use_friendlies_features else None,
    }


def _build_xy(bundle: dict, source: pd.DataFrame, config: DatasetConfig):
    return build_training_data(
        bundle["matches_df"],
        bundle["goalscorers_df"],
        bundle["shootouts_df"],
        bundle["rankings_df"],
        bundle["substitutions_df"],
        bundle["player_appearances_df"],
        bundle["player_goals_df"],
        bundle["award_winners_df"],
        bundle["copa_data"],
        bundle["friendlies_data"],
        include_curated_friendlies=config.include_friendlies_train,
        source_matches_df=source,
        skip_goalscoring=True,
        skip_supersub=True,
        skip_player_stats=True,
        verbose=False,
    )


def _run_fold(config: DatasetConfig, target: str, validation_source: pd.DataFrame, training_source: pd.DataFrame) -> dict:
    bundle = _load_bundle(config)
    X_train, y_train, _ = _build_xy(bundle, _sample_train_source(training_source), config)
    X_test, y_test, _ = _build_xy(bundle, validation_source, config)

    model = _model()
    _fit_catboost(model, X_train, y_train)
    metrics = _evaluate_catboost(model, X_test, y_test)
    metrics.update(
        {
            "config": config.name,
            "target": target,
            "train_source_matches": int(len(training_source)),
            "sampled_train_source_matches": int(len(X_train) / 2),
            "validation_source_matches": int(len(validation_source)),
            "train_rows": int(len(X_train)),
            "validation_rows": int(len(X_test)),
            "feature_columns": int(len(MODEL_INPUT_COLUMNS)),
        }
    )
    return metrics


def _world_cup_folds(config: DatasetConfig) -> list[dict]:
    bundle = _load_bundle(config)
    candidates = _training_matches(bundle["matches_df"], include_curated_friendlies=config.include_friendlies_train)
    out = []
    for year in WORLD_CUP_YEARS:
        year_rows = candidates[candidates["date"].dt.year == year]
        validation = year_rows[
            year_rows["tournament"].str.contains("world cup", case=False, na=False)
            & ~year_rows["tournament"].str.contains("qualification|qualifier", case=False, na=False)
        ].copy()
        training = candidates[candidates["date"].dt.year < year].copy()
        if validation.empty or training.empty:
            continue
        out.append(_run_fold(config, f"world_cup_{year}", validation, training))
    return out


def _single_target_fold(config: DatasetConfig, target: str, year: int, tournament_contains: str) -> dict | None:
    bundle = _load_bundle(config)
    candidates = _training_matches(bundle["matches_df"], include_curated_friendlies=config.include_friendlies_train)
    year_rows = candidates[candidates["date"].dt.year == year]
    validation = year_rows[year_rows["tournament"].str.contains(tournament_contains, case=False, na=False)].copy()
    training = candidates[candidates["date"].dt.year < year].copy()
    if validation.empty or training.empty:
        return None
    return _run_fold(config, target, validation, training)


def _friendlies_tail_fold(config: DatasetConfig) -> dict | None:
    bundle = _load_bundle(config)
    candidates = _training_matches(bundle["matches_df"], include_curated_friendlies=True)
    friendlies = candidates[candidates["tournament"] == FRIENDLIES_TOURNAMENT].sort_values("date")
    if len(friendlies) < 10:
        return None
    cutoff = int(len(friendlies) * 0.8)
    training = pd.concat(
        [candidates[candidates["tournament"] != FRIENDLIES_TOURNAMENT], friendlies.iloc[:cutoff]],
        ignore_index=True,
    ).sort_values("date")
    validation = friendlies.iloc[cutoff:].copy()
    return _run_fold(config, "friendlies_2026_tail", validation, training)


def _summarize(results: list[dict]) -> dict:
    df = pd.DataFrame(results)
    summary = {}
    if df.empty:
        return summary
    for target, target_df in df.groupby("target"):
        rows = []
        for config, group in target_df.groupby("config"):
            rows.append(
                {
                    "config": config,
                    "folds": int(len(group)),
                    "accuracy": round(float(group["accuracy"].mean()), 4),
                    "balanced_accuracy": round(float(group["balanced_accuracy"].mean()), 4),
                    "log_loss": round(float(group["log_loss"].mean()), 4),
                }
            )
        summary[target] = sorted(rows, key=lambda row: (-row["accuracy"], row["log_loss"]))
    world_cup = df[df["target"].str.startswith("world_cup_")]
    if not world_cup.empty:
        rows = []
        for config, group in world_cup.groupby("config"):
            rows.append(
                {
                    "config": config,
                    "folds": int(len(group)),
                    "accuracy": round(float(group["accuracy"].mean()), 4),
                    "balanced_accuracy": round(float(group["balanced_accuracy"].mean()), 4),
                    "log_loss": round(float(group["log_loss"].mean()), 4),
                }
            )
        summary["world_cup_average"] = sorted(rows, key=lambda row: (-row["accuracy"], row["log_loss"]))
    return summary


def run_audit() -> dict:
    results = []
    for config in CONFIGS:
        print(f"Auditing {config.name}", flush=True)
        results.extend(_world_cup_folds(config))
        copa = _single_target_fold(config, "copa_2024", 2024, "copa")
        if copa is not None:
            results.append(copa)
        if config.include_friendlies:
            friendlies = _friendlies_tail_fold(config)
            if friendlies is not None:
                results.append(friendlies)

    report = {
        "source_cap": SOURCE_CAP,
        "model_iterations": MODEL_ITERATIONS,
        "results": results,
        "summary": _summarize(results),
    }
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report["summary"], indent=2))
    return report


if __name__ == "__main__":
    run_audit()
