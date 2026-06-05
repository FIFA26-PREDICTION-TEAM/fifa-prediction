"""
Model training script.

Run directly:   python -m model.train
Called by app:  from model.train import train_and_save

The training pipeline builds leakage-safe rolling features, keeps swapped match
orientations in the same validation fold, trains CatBoost, and saves the model
bundle for inference.
"""

import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

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
from data.copa_america import load_copa_america_data
from data.preprocess import build_feature_row
from model.features import FEATURE_COLUMNS, MODEL_INPUT_COLUMNS, RAW_CONTEXT_COLUMNS, normalize_tournament_name

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pkl")
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(ARTIFACTS_DIR, "feature_columns.json")
META_PATH = os.path.join(ARTIFACTS_DIR, "meta.json")

TRAIN_FROM_YEAR = 2006
RANDOM_STATE = 42
VALIDATION_YEAR = None  # None means use the newest match year in the training data.


def _label(row: pd.Series, team_a: str, shootout_winners: dict | None = None) -> int:
    """0=Team A loss, 1=draw, 2=Team A win."""
    if row["home_team"] == team_a:
        gf, ga = row["home_score"], row["away_score"]
    else:
        gf, ga = row["away_score"], row["home_score"]

    if gf > ga:
        return 2
    if gf < ga:
        return 0

    if shootout_winners:
        key = (str(row["date"])[:10], row["home_team"], row["away_team"])
        winner = shootout_winners.get(key)
        if winner == team_a:
            return 2
        if winner is not None:
            return 0
    return 1


def _competitive_matches(matches_df: pd.DataFrame) -> pd.DataFrame:
    competitive_keywords = [
        "world cup",
        "copa",
        "euro",
        "afcon",
        "asian cup",
        "qualifier",
        "qualification",
        "gold cup",
        "nations league",
        "confederation",
    ]
    df = matches_df[matches_df["date"].dt.year >= TRAIN_FROM_YEAR].copy()
    mask = df["tournament"].str.lower().str.contains("|".join(competitive_keywords), na=False)
    return df[mask].sort_values("date").reset_index(drop=True)


def _shootout_winners(shootouts_df: pd.DataFrame | None) -> dict:
    if shootouts_df is None:
        return {}
    return {
        (str(row["date"])[:10], row["home_team"], row["away_team"]): row["winner"]
        for _, row in shootouts_df.iterrows()
    }


def _context_row(team_a: str, team_b: str, tournament: str) -> dict:
    return {
        "team_a": team_a,
        "team_b": team_b,
        "tournament": normalize_tournament_name(tournament),
    }


def _sample_matches(df: pd.DataFrame, shootout_winners: dict) -> pd.DataFrame:
    """Optionally cap matches with ML_PRJCT_TRAIN_SAMPLE_SIZE for quick experiments."""
    sample_size = os.getenv("ML_PRJCT_TRAIN_SAMPLE_SIZE")
    if not sample_size:
        return df

    try:
        cap = int(sample_size)
    except ValueError:
        return df

    if cap <= 0 or len(df) <= cap:
        return df

    tmp = df.copy()
    tmp["_tmp_label"] = tmp.apply(lambda row: _label(row, row["home_team"], shootout_winners), axis=1)
    sampled = (
        tmp.groupby("_tmp_label", group_keys=False)
        .apply(lambda group: group.sample(min(len(group), max(1, cap // 3)), random_state=RANDOM_STATE))
        .sort_values("date")
        .drop(columns=["_tmp_label"])
        .reset_index(drop=True)
    )
    return sampled


def build_training_data(
    matches_df: pd.DataFrame,
    goalscorers_df,
    shootouts_df,
    rankings_df,
    substitutions_df=None,
    player_appearances_df=None,
    player_goals_df=None,
    award_winners_df=None,
    copa_data=None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    """Build model-ready rows with no future information leakage."""
    df = _competitive_matches(matches_df)
    shootout_winners = _shootout_winners(shootouts_df)
    df = _sample_matches(df, shootout_winners)

    if verbose:
        print(f"Building features for {len(df)} training matches...")

    rows = []
    labels = []
    dates = []
    t0 = time.time()

    for i, (_, row) in enumerate(df.iterrows()):
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
            skip_goalscoring=False,
            skip_supersub=False,
            skip_player_stats=False,
        )

        feat_a, _ = build_feature_row(team_a=home, team_b=away, **build_kwargs)
        label_a = _label(row, home, shootout_winners)
        rows.append({**_context_row(home, away, row["tournament"]), **feat_a})
        labels.append(label_a)
        dates.append(row["date"])

        feat_b, _ = build_feature_row(team_a=away, team_b=home, **build_kwargs)
        rows.append({**_context_row(away, home, row["tournament"]), **feat_b})
        labels.append(2 - label_a)
        dates.append(row["date"])

        if verbose and (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  {i + 1}/{len(df)} matches - {elapsed:.1f}s elapsed")

    X = pd.DataFrame(rows, columns=MODEL_INPUT_COLUMNS)
    for col in RAW_CONTEXT_COLUMNS:
        X[col] = X[col].fillna("unknown").astype(str)
    for col in FEATURE_COLUMNS:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)

    return X, np.array(labels, dtype=np.int32), pd.to_datetime(pd.Series(dates))


def _time_split(dates: pd.Series) -> tuple[np.ndarray, np.ndarray, int]:
    years = sorted(dates.dt.year.unique())
    validation_year = VALIDATION_YEAR or years[-1]
    train_mask = (dates.dt.year < validation_year).to_numpy()
    test_mask = (dates.dt.year == validation_year).to_numpy()

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        cutoff = dates.quantile(0.8)
        train_mask = (dates < cutoff).to_numpy()
        test_mask = (dates >= cutoff).to_numpy()
        validation_year = int(pd.Timestamp(cutoff).year)

    return train_mask, test_mask, int(validation_year)


def _catboost_model() -> CatBoostClassifier:
    if not HAS_CATBOOST:
        raise RuntimeError("CatBoost is required. Install dependencies with `pip install -r requirements.txt`.")

    return CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="MultiClass",
        iterations=1000,
        depth=6,
        learning_rate=0.03,
        l2_leaf_reg=5,
        random_seed=RANDOM_STATE,
        verbose=False,
    )


def _fit_catboost(model: CatBoostClassifier, X_train: pd.DataFrame, y_train: np.ndarray):
    model.fit(X_train[MODEL_INPUT_COLUMNS], y_train, cat_features=RAW_CONTEXT_COLUMNS)


def _evaluate_catboost(model: CatBoostClassifier, X_test: pd.DataFrame, y_test: np.ndarray) -> dict:
    X_eval = X_test[MODEL_INPUT_COLUMNS]
    pred = model.predict(X_eval)
    if isinstance(pred, np.ndarray) and pred.ndim > 1:
        pred = pred.ravel()
    proba = model.predict_proba(X_eval)
    return {
        "model_name": "CatBoost",
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "log_loss": float(log_loss(y_test, proba, labels=[0, 1, 2])),
    }


def train_and_save(verbose: bool = True) -> dict:
    """
    Full train pipeline. Returns meta dict and saves artifacts to model/artifacts/.
    """
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    if verbose:
        print("Loading data...")
    matches_df = load_matches()
    goalscorers_df = load_goalscorers()
    shootouts_df = load_shootouts()
    rankings_df = load_rankings()
    substitutions_df = load_substitutions()
    player_appearances_df = load_player_appearances()
    player_goals_df = load_player_goals()
    award_winners_df = load_award_winners()
    copa_data = load_copa_america_data()

    if matches_df is None:
        raise RuntimeError("matches.csv is required for training. Drop it into the project root.")

    X, y, dates = build_training_data(
        matches_df,
        goalscorers_df,
        shootouts_df,
        rankings_df,
        substitutions_df,
        player_appearances_df,
        player_goals_df,
        award_winners_df,
        copa_data,
        verbose=verbose,
    )

    train_mask, test_mask, validation_year = _time_split(dates)
    X_train, X_test = X.loc[train_mask], X.loc[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    if verbose:
        print(f"\nValidation year: {validation_year}")
        print("Training model: CatBoost")

    validation_model = _catboost_model()
    _fit_catboost(validation_model, X_train, y_train)
    metrics = _evaluate_catboost(validation_model, X_test, y_test)

    if verbose:
        print(
            f"CatBoost: "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"balanced_accuracy={metrics['balanced_accuracy']:.4f}, "
            f"log_loss={metrics['log_loss']:.4f}"
        )

    if verbose:
        pred = validation_model.predict(X_test[MODEL_INPUT_COLUMNS])
        if isinstance(pred, np.ndarray) and pred.ndim > 1:
            pred = pred.ravel()
        print(f"\nSelected model: CatBoost (accuracy: {metrics['accuracy']:.4f})")
        print(classification_report(y_test, pred, target_names=["Team A Loss", "Draw", "Team A Win"]))
        print("Confusion matrix:")
        print(confusion_matrix(y_test, pred))

    best_model = _catboost_model()
    _fit_catboost(best_model, X, y)

    model_bundle = {
        "model": best_model,
        "model_name": "CatBoost",
        "feature_columns": MODEL_INPUT_COLUMNS,
        "categorical_columns": RAW_CONTEXT_COLUMNS,
        "raw_context_columns": RAW_CONTEXT_COLUMNS,
        "numeric_feature_columns": FEATURE_COLUMNS,
    }
    joblib.dump(model_bundle, MODEL_PATH)
    joblib.dump(None, SCALER_PATH)  # Backwards-compatible placeholder.
    with open(FEATURES_PATH, "w") as f:
        json.dump(MODEL_INPUT_COLUMNS, f, indent=2)

    data_min = dates.min()
    data_max = dates.max()
    meta = {
        "model_name": "CatBoost",
        "accuracy": round(metrics["accuracy"], 4),
        "balanced_accuracy": round(metrics["balanced_accuracy"], 4),
        "log_loss": round(metrics["log_loss"], 4),
        "validation_year": validation_year,
        "training_rows": int(len(X)),
        "train_rows": int(train_mask.sum()),
        "validation_rows": int(test_mask.sum()),
        "features": int(len(MODEL_INPUT_COLUMNS)),
        "numeric_features": int(len(FEATURE_COLUMNS)),
        "categorical_features": int(len(RAW_CONTEXT_COLUMNS)),
        "train_from_year": TRAIN_FROM_YEAR,
        "data_start": str(data_min.date()) if pd.notna(data_min) else None,
        "data_end": str(data_max.date()) if pd.notna(data_max) else None,
        "validation_metrics": {
            "accuracy": round(metrics["accuracy"], 4),
            "balanced_accuracy": round(metrics["balanced_accuracy"], 4),
            "log_loss": round(metrics["log_loss"], 4),
        },
        "model_policy": "fixed_catboost",
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    if verbose:
        print(f"\nArtifacts saved to {ARTIFACTS_DIR}")
        print(f"Model: CatBoost | Accuracy: {metrics['accuracy']:.1%} | Rows: {len(X)}")

    return meta


if __name__ == "__main__":
    train_and_save(verbose=True)
