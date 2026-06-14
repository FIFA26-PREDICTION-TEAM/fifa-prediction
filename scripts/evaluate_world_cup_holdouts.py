"""Evaluate clean World Cup holdouts without overwriting saved model artifacts."""

from __future__ import annotations

import json
import os

import pandas as pd

from data.copa_america import load_copa_america_data
from data.euro_2024 import load_euro_2024_data
from data.ingest import load_rankings
from data.international_friendlies import load_friendlies_data
from model.features import MODEL_INPUT_COLUMNS
from model.train import (
    VALIDATION_YEAR,
    USE_TWO_STAGE_RESULT,
    _calibrate_result_threshold_from_training,
    _evaluate_result_model,
    _fit_multiclass_result_model,
    _fit_two_stage_model,
    _load_historical_goalscorers,
    _load_historical_matches,
    _load_historical_shootouts,
    _select_historical_training_source,
    _training_matches,
    build_training_data,
)


def _world_cup_rows(candidates: pd.DataFrame, year: int) -> pd.DataFrame:
    year_rows = candidates[candidates["date"].dt.year == year]
    if "_is_wc" in year_rows.columns:
        return year_rows[year_rows["_is_wc"].fillna(False)].copy()
    return year_rows[
        year_rows["tournament"].str.contains("world cup", case=False, na=False)
        & ~year_rows["tournament"].str.contains("qualification|qualifier", case=False, na=False)
    ].copy()


def evaluate_year(year: int) -> dict:
    matches_df = _load_historical_matches()
    goalscorers_df = _load_historical_goalscorers()
    shootouts_df = _load_historical_shootouts()
    rankings_df = load_rankings()
    copa_data = load_copa_america_data()
    euro_data = load_euro_2024_data()
    friendlies_data = load_friendlies_data()

    if matches_df is None:
        raise RuntimeError("Historical matches are required.")

    candidates = _training_matches(matches_df, include_curated_friendlies=False, train_from_year=1872)
    validation_source = _world_cup_rows(candidates, year)
    train_source = candidates[candidates["date"].dt.year < year].copy()
    train_source = _select_historical_training_source(train_source, validation_year=year)

    if validation_source.empty:
        raise RuntimeError(f"No World Cup validation rows found for {year}.")

    skip_expensive = True
    X_train, y_train, train_dates, train_weights = build_training_data(
        matches_df,
        goalscorers_df,
        shootouts_df,
        rankings_df,
        None,
        None,
        None,
        None,
        copa_data,
        friendlies_data,
        euro_data,
        source_matches_df=train_source,
        skip_goalscoring=skip_expensive,
        skip_supersub=skip_expensive,
        skip_player_stats=skip_expensive,
        return_weights=True,
        verbose=True,
    )
    X_test, y_test, _ = build_training_data(
        matches_df,
        goalscorers_df,
        shootouts_df,
        rankings_df,
        None,
        None,
        None,
        None,
        copa_data,
        friendlies_data,
        euro_data,
        source_matches_df=validation_source,
        skip_goalscoring=skip_expensive,
        skip_supersub=skip_expensive,
        skip_player_stats=skip_expensive,
        verbose=True,
    )

    default_threshold_calibration = {
        "threshold": 0.24,
        "calibration_year": None,
        "source": "default_multiclass_rule",
    }
    candidates = []
    multiclass_model = _fit_multiclass_result_model(X_train, y_train, sample_weight=train_weights)
    multiclass_metrics = _evaluate_result_model(multiclass_model, X_test, y_test)
    candidates.append((multiclass_model, multiclass_metrics, default_threshold_calibration))

    if USE_TWO_STAGE_RESULT:
        threshold_calibration = _calibrate_result_threshold_from_training(
            X_train,
            y_train,
            train_dates,
            train_weights,
            year,
        )
        two_stage_model = _fit_two_stage_model(X_train, y_train, sample_weight=train_weights)
        two_stage_model["draw_threshold"] = threshold_calibration["threshold"]
        two_stage_metrics = _evaluate_result_model(two_stage_model, X_test, y_test)
        candidates.append((two_stage_model, two_stage_metrics, threshold_calibration))

    model, metrics, threshold_calibration = max(
        candidates,
        key=lambda item: (item[1]["macro_f1"], item[1]["accuracy"], item[1]["draw_f1"]),
    )
    return {
        "target": f"world_cup_{year}",
        "accuracy": round(metrics["accuracy"], 4),
        "balanced_accuracy": round(metrics["balanced_accuracy"], 4),
        "macro_f1": round(metrics["macro_f1"], 4),
        "weighted_f1": round(metrics["weighted_f1"], 4),
        "log_loss": round(metrics["log_loss"], 4),
        "draw_f1": round(metrics["draw_f1"], 4),
        "draw_precision": round(metrics["draw_precision"], 4),
        "draw_recall": round(metrics["draw_recall"], 4),
        "predicted_draws": int(metrics["predicted_draws"]),
        "actual_draws": int(metrics["actual_draws"]),
        "train_source_matches": int(len(train_source)),
        "validation_source_matches": int(len(validation_source)),
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_test)),
        "features": int(len(MODEL_INPUT_COLUMNS)),
        "result_model_type": model.get("result_model_type"),
        "draw_threshold": round(float(threshold_calibration["threshold"]), 4),
        "threshold_calibration_year": threshold_calibration.get("calibration_year"),
        "threshold_calibration_source": threshold_calibration.get("source"),
    }


def main() -> None:
    raw_years = os.getenv("ML_PRJCT_EVAL_WORLD_CUPS", "2018,2022")
    years = [int(part.strip()) for part in raw_years.split(",") if part.strip()]
    results = [evaluate_year(year) for year in years]
    report = {
        "years": years,
        "historical_source_cap": int(os.getenv("ML_PRJCT_HISTORICAL_SOURCE_CAP", "5000")),
        "default_validation_year": VALIDATION_YEAR,
        "results": results,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
