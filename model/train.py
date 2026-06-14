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
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

from data.ingest import (
    COMPETITIVE_PATTERN,
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
from data.euro_2024 import load_euro_2024_data
from data.international_friendlies import FRIENDLIES_TOURNAMENT, append_friendlies_matches, load_friendlies_data
from data.preprocess import build_feature_row
from model.features import (
    FEATURE_COLUMNS,
    MODEL_INPUT_COLUMNS,
    RAW_CONTEXT_COLUMNS,
    get_tournament_weight,
    normalize_tournament_name,
)

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pkl")
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(ARTIFACTS_DIR, "feature_columns.json")
META_PATH = os.path.join(ARTIFACTS_DIR, "meta.json")

TRAIN_FROM_YEAR = 2006
RANDOM_STATE = 42
VALIDATION_YEAR = 2022  # Hold out the latest completed FIFA World Cup for validation.
CURATED_FRIENDLIES_FROM_YEAR = 2026
DRAW_CLASS_WEIGHT = float(os.getenv("ML_PRJCT_DRAW_CLASS_WEIGHT", "1.6"))
DRAW_PROB_THRESHOLD = float(os.getenv("ML_PRJCT_DRAW_PROB_THRESHOLD", "0.24"))
DRAW_CLOSE_MARGIN = float(os.getenv("ML_PRJCT_DRAW_CLOSE_MARGIN", "0.15"))
USE_TWO_STAGE_RESULT = os.getenv("ML_PRJCT_USE_TWO_STAGE_RESULT", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
DRAW_BINARY_WEIGHT = float(os.getenv("ML_PRJCT_DRAW_BINARY_WEIGHT", "2.5"))
DRAW_THRESHOLD_MIN = float(os.getenv("ML_PRJCT_DRAW_THRESHOLD_MIN", "0.08"))
DRAW_THRESHOLD_MAX = float(os.getenv("ML_PRJCT_DRAW_THRESHOLD_MAX", "0.45"))
DRAW_THRESHOLD_STEP = float(os.getenv("ML_PRJCT_DRAW_THRESHOLD_STEP", "0.01"))
DRAW_CALIBRATION_MIN_ACCURACY = float(os.getenv("ML_PRJCT_DRAW_CALIBRATION_MIN_ACCURACY", "0.55"))
USE_HISTORICAL_WEIGHTED = os.getenv("ML_PRJCT_USE_HISTORICAL_WEIGHTED", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
HISTORICAL_SOURCE_CAP = int(os.getenv("ML_PRJCT_HISTORICAL_SOURCE_CAP", "5000"))
HISTORICAL_FAST_FEATURES = os.getenv("ML_PRJCT_HISTORICAL_FAST_FEATURES", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
RAW_RESULTS_DIR = os.path.join(ROOT_DIR, "data", "raw", "international_results")


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        return series.fillna(False).astype(str).str.strip().str.lower().isin({"1", "true", "yes"})
    return series.fillna(False).astype(bool)


def _prepare_matches_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team"])
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce").fillna(0).astype(int)
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce").fillna(0).astype(int)
    df["tournament"] = df["tournament"].fillna("Unknown").astype(str)
    df["country"] = df["country"].fillna("").astype(str) if "country" in df.columns else ""
    df["city"] = df["city"].fillna("").astype(str) if "city" in df.columns else ""
    df["neutral"] = _parse_bool_series(df["neutral"]) if "neutral" in df.columns else False
    tournament = df["tournament"].str.lower()
    df["_is_competitive"] = tournament.str.contains(COMPETITIVE_PATTERN, na=False)
    df["_is_wc"] = tournament.str.contains("world cup", na=False) & ~tournament.str.contains(
        "qualifier|qualification", na=False
    )
    return df.sort_values("date").reset_index(drop=True)


def _prepare_event_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date", "home_team", "away_team"]).sort_values("date").reset_index(drop=True)


def _first_existing(*paths: str) -> str | None:
    return next((path for path in paths if os.path.exists(path)), None)


def _load_historical_matches() -> pd.DataFrame | None:
    path = _first_existing(
        os.path.join(PROCESSED_DIR, "matches_all.csv"),
        os.path.join(RAW_RESULTS_DIR, "results.csv"),
        os.path.join(ROOT_DIR, "matches.csv"),
    )
    if not path:
        return None
    return _prepare_matches_frame(pd.read_csv(path))


def _load_historical_goalscorers() -> pd.DataFrame | None:
    path = _first_existing(
        os.path.join(PROCESSED_DIR, "goalscorers_all.csv"),
        os.path.join(RAW_RESULTS_DIR, "goalscorers.csv"),
        os.path.join(ROOT_DIR, "goalscorers.csv"),
    )
    if not path:
        return None
    df = _prepare_event_frame(pd.read_csv(path))
    df["minute"] = pd.to_numeric(df.get("minute"), errors="coerce")
    df["own_goal"] = _parse_bool_series(df["own_goal"]) if "own_goal" in df.columns else False
    df["penalty"] = _parse_bool_series(df["penalty"]) if "penalty" in df.columns else False
    return df


def _load_historical_shootouts() -> pd.DataFrame | None:
    path = _first_existing(
        os.path.join(PROCESSED_DIR, "shootouts_all.csv"),
        os.path.join(RAW_RESULTS_DIR, "shootouts.csv"),
        os.path.join(ROOT_DIR, "shootouts.csv"),
    )
    if not path:
        return None
    return _prepare_event_frame(pd.read_csv(path).dropna(subset=["winner"]))


def _era_weight(match_date) -> float:
    year = pd.Timestamp(match_date).year
    if year >= 2018:
        return 1.25
    if year >= 2006:
        return 1.0
    if year >= 2002:
        return 0.45
    return 0.08


def _match_sample_weight(row: pd.Series) -> float:
    return float(_era_weight(row["date"]) * get_tournament_weight(row.get("tournament", "")))


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


def _training_matches(
    matches_df: pd.DataFrame,
    include_curated_friendlies: bool = False,
    train_from_year: int = TRAIN_FROM_YEAR,
) -> pd.DataFrame:
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
    df = matches_df[matches_df["date"].dt.year >= train_from_year].copy()
    if "_is_competitive" in df.columns:
        mask = df["_is_competitive"].fillna(False)
    else:
        mask = df["tournament"].str.lower().str.contains("|".join(competitive_keywords), na=False)
    if include_curated_friendlies:
        friendly_mask = (
            df["tournament"].eq(FRIENDLIES_TOURNAMENT)
            & (df["date"].dt.year >= CURATED_FRIENDLIES_FROM_YEAR)
        )
        mask = mask | friendly_mask
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


def _venue_context(row: pd.Series, team_a: str, team_b: str) -> dict:
    neutral = bool(row.get("neutral", False))
    country = str(row.get("country", "") or "")
    return {
        "team_a_is_home": float((not neutral) and row["home_team"] == team_a),
        "team_b_is_home": float((not neutral) and row["home_team"] == team_b),
        "neutral_venue": float(neutral),
        "team_a_host_country": float(country == team_a),
        "team_b_host_country": float(country == team_b),
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


def _select_historical_training_source(df: pd.DataFrame, validation_year: int = VALIDATION_YEAR) -> pd.DataFrame:
    """Cap broad historical targets while preserving every World Cup row."""
    source = df[df["date"].dt.year != validation_year].copy()
    if HISTORICAL_SOURCE_CAP <= 0 or len(source) <= HISTORICAL_SOURCE_CAP:
        return source.sort_values("date").reset_index(drop=True)

    wc_mask = source["_is_wc"].fillna(False) if "_is_wc" in source.columns else source["tournament"].str.contains(
        "world cup", case=False, na=False
    )
    wc_rows = source[wc_mask]
    other = source[~wc_mask]
    remaining = max(0, HISTORICAL_SOURCE_CAP - len(wc_rows))

    if remaining > 0 and not other.empty:
        recent = other[other["date"].dt.year >= 2006]
        older = other[other["date"].dt.year < 2006]
        recent_n = min(len(recent), int(remaining * 0.75))
        older_n = min(len(older), remaining - recent_n)
        sampled = []
        if recent_n > 0:
            sampled.append(recent.sample(n=recent_n, random_state=RANDOM_STATE))
        if older_n > 0:
            sampled.append(older.sample(n=older_n, random_state=RANDOM_STATE))
        other_sample = pd.concat(sampled, ignore_index=True) if sampled else other.head(0)
    else:
        other_sample = other.head(0)

    return pd.concat([wc_rows, other_sample], ignore_index=True).sort_values("date").reset_index(drop=True)


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
    friendlies_data=None,
    euro_data=None,
    include_curated_friendlies: bool = False,
    source_matches_df: pd.DataFrame | None = None,
    skip_goalscoring: bool = False,
    skip_supersub: bool = False,
    skip_player_stats: bool = False,
    return_weights: bool = False,
    verbose: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, pd.Series] | tuple[pd.DataFrame, np.ndarray, pd.Series, np.ndarray]:
    """Build model-ready rows with no future information leakage."""
    if source_matches_df is None:
        df = _training_matches(matches_df, include_curated_friendlies=include_curated_friendlies)
    else:
        df = source_matches_df.sort_values("date").reset_index(drop=True)
    shootout_winners = _shootout_winners(shootouts_df)
    df = _sample_matches(df, shootout_winners)

    if verbose:
        print(f"Building features for {len(df)} training matches...")

    rows = []
    labels = []
    dates = []
    weights = []
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
            friendlies_data=friendlies_data,
            euro_data=euro_data,
            skip_goalscoring=skip_goalscoring,
            skip_supersub=skip_supersub,
            skip_player_stats=skip_player_stats,
        )

        feat_a, _ = build_feature_row(
            team_a=home,
            team_b=away,
            venue_context=_venue_context(row, home, away),
            **build_kwargs,
        )
        label_a = _label(row, home, shootout_winners)
        rows.append({**_context_row(home, away, row["tournament"]), **feat_a})
        labels.append(label_a)
        dates.append(row["date"])
        weights.append(_match_sample_weight(row))

        feat_b, _ = build_feature_row(
            team_a=away,
            team_b=home,
            venue_context=_venue_context(row, away, home),
            **build_kwargs,
        )
        rows.append({**_context_row(away, home, row["tournament"]), **feat_b})
        labels.append(2 - label_a)
        dates.append(row["date"])
        weights.append(_match_sample_weight(row))

        if verbose and (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  {i + 1}/{len(df)} matches - {elapsed:.1f}s elapsed")

    X = pd.DataFrame(rows, columns=MODEL_INPUT_COLUMNS)
    for col in RAW_CONTEXT_COLUMNS:
        X[col] = X[col].fillna("unknown").astype(str)
    for col in FEATURE_COLUMNS:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)

    if return_weights:
        return (
            X,
            np.array(labels, dtype=np.int32),
            pd.to_datetime(pd.Series(dates)),
            np.array(weights, dtype=np.float32),
        )
    return X, np.array(labels, dtype=np.int32), pd.to_datetime(pd.Series(dates))


def _time_split(dates: pd.Series, validation_year: int | None = None) -> tuple[np.ndarray, np.ndarray, int]:
    years = sorted(dates.dt.year.unique())
    validation_year = validation_year or VALIDATION_YEAR or years[-1]
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
        class_weights=[1.0, DRAW_CLASS_WEIGHT, 1.0],
        random_seed=RANDOM_STATE,
        verbose=False,
    )


def _binary_catboost_model(positive_class_weight: float = 1.0) -> CatBoostClassifier:
    if not HAS_CATBOOST:
        raise RuntimeError("CatBoost is required. Install dependencies with `pip install -r requirements.txt`.")

    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
        iterations=1000,
        depth=6,
        learning_rate=0.03,
        l2_leaf_reg=5,
        class_weights=[1.0, positive_class_weight],
        random_seed=RANDOM_STATE,
        verbose=False,
    )


def _fit_catboost(model: CatBoostClassifier, X_train: pd.DataFrame, y_train: np.ndarray, sample_weight=None):
    model.fit(
        X_train[MODEL_INPUT_COLUMNS],
        y_train,
        cat_features=RAW_CONTEXT_COLUMNS,
        sample_weight=sample_weight,
    )


def _fit_binary_catboost(model: CatBoostClassifier, X_train: pd.DataFrame, y_train: np.ndarray, sample_weight=None):
    model.fit(
        X_train[MODEL_INPUT_COLUMNS],
        y_train,
        cat_features=RAW_CONTEXT_COLUMNS,
        sample_weight=sample_weight,
    )


def _positive_probability(model: CatBoostClassifier, proba: np.ndarray) -> np.ndarray:
    if 1 not in model.classes_:
        return np.zeros(len(proba), dtype=np.float64)
    return proba[:, int(np.where(model.classes_ == 1)[0][0])].astype(np.float64)


def _fit_two_stage_model(X_train: pd.DataFrame, y_train: np.ndarray, sample_weight=None) -> dict:
    draw_model = _binary_catboost_model(DRAW_BINARY_WEIGHT)
    y_draw = (y_train == 1).astype(np.int32)
    _fit_binary_catboost(draw_model, X_train, y_draw, sample_weight=sample_weight)

    decisive_mask = y_train != 1
    if decisive_mask.sum() == 0:
        raise RuntimeError("Two-stage model requires decisive examples.")
    decisive_model = _binary_catboost_model(1.0)
    y_decisive = (y_train[decisive_mask] == 2).astype(np.int32)
    decisive_weights = sample_weight[decisive_mask] if sample_weight is not None else None
    _fit_binary_catboost(decisive_model, X_train.loc[decisive_mask], y_decisive, sample_weight=decisive_weights)

    return {
        "result_model_type": "two_stage",
        "model_name": "TwoStageCatBoost",
        "draw_model": draw_model,
        "decisive_model": decisive_model,
        "draw_threshold": DRAW_PROB_THRESHOLD,
    }


def _fit_multiclass_result_model(X_train: pd.DataFrame, y_train: np.ndarray, sample_weight=None) -> dict:
    model = _catboost_model()
    _fit_catboost(model, X_train, y_train, sample_weight=sample_weight)
    return {
        "result_model_type": "multiclass",
        "model_name": "CatBoost",
        "model": model,
        "draw_threshold": DRAW_PROB_THRESHOLD,
    }


def _fit_result_model(X_train: pd.DataFrame, y_train: np.ndarray, sample_weight=None) -> dict:
    if USE_TWO_STAGE_RESULT:
        return _fit_two_stage_model(X_train, y_train, sample_weight=sample_weight)
    return _fit_multiclass_result_model(X_train, y_train, sample_weight=sample_weight)


def _class_probabilities(model: CatBoostClassifier, proba: np.ndarray) -> np.ndarray:
    out = np.zeros((len(proba), 3), dtype=np.float64)
    for idx, class_label in enumerate(model.classes_):
        out[:, int(class_label)] = proba[:, idx]
    return out


def _normalize_probabilities(probs: np.ndarray) -> np.ndarray:
    probs = np.clip(probs.astype(np.float64), 1e-6, 1.0)
    return probs / probs.sum(axis=1, keepdims=True)


def _predict_two_stage_proba(model_bundle: dict, X_eval: pd.DataFrame) -> np.ndarray:
    draw_model = model_bundle["draw_model"]
    decisive_model = model_bundle["decisive_model"]
    p_draw = _positive_probability(draw_model, draw_model.predict_proba(X_eval[MODEL_INPUT_COLUMNS]))
    p_win_given_decisive = _positive_probability(
        decisive_model,
        decisive_model.predict_proba(X_eval[MODEL_INPUT_COLUMNS]),
    )
    p_decisive = 1.0 - p_draw
    probs = np.column_stack(
        [
            p_decisive * (1.0 - p_win_given_decisive),
            p_draw,
            p_decisive * p_win_given_decisive,
        ]
    )
    return _normalize_probabilities(probs)


def _predict_result_proba(model_bundle: dict, X_eval: pd.DataFrame) -> np.ndarray:
    if model_bundle.get("result_model_type") == "two_stage":
        return _predict_two_stage_proba(model_bundle, X_eval)
    model = model_bundle["model"]
    return _normalize_probabilities(_class_probabilities(model, model.predict_proba(X_eval[MODEL_INPUT_COLUMNS])))


def _predict_labels_from_proba(
    model: CatBoostClassifier,
    proba: np.ndarray,
    draw_threshold: float = DRAW_PROB_THRESHOLD,
) -> np.ndarray:
    probs = _class_probabilities(model, proba)
    labels = probs.argmax(axis=1).astype(np.int32)
    best_non_draw = np.maximum(probs[:, 0], probs[:, 2])
    close_to_best = best_non_draw - probs[:, 1] <= DRAW_CLOSE_MARGIN
    draw_rule = (probs[:, 1] >= draw_threshold) & close_to_best
    labels[draw_rule] = 1
    return labels


def _predict_result_labels(model_bundle: dict, probs: np.ndarray) -> np.ndarray:
    if model_bundle.get("result_model_type") == "two_stage":
        threshold = float(model_bundle.get("draw_threshold", DRAW_PROB_THRESHOLD))
        decisive_labels = np.where(probs[:, 2] >= probs[:, 0], 2, 0)
        return np.where(probs[:, 1] >= threshold, 1, decisive_labels).astype(np.int32)
    return _predict_labels_from_proba(
        model_bundle["model"],
        probs,
        draw_threshold=float(model_bundle.get("draw_threshold", DRAW_PROB_THRESHOLD)),
    )


def _draw_metrics(y_true: np.ndarray, pred: np.ndarray) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true == 1,
        pred == 1,
        average="binary",
        zero_division=0,
    )
    return {
        "draw_precision": float(precision),
        "draw_recall": float(recall),
        "draw_f1": float(f1),
        "predicted_draws": int((pred == 1).sum()),
        "actual_draws": int((y_true == 1).sum()),
    }


def _evaluate_catboost(model: CatBoostClassifier, X_test: pd.DataFrame, y_test: np.ndarray) -> dict:
    X_eval = X_test[MODEL_INPUT_COLUMNS]
    proba = model.predict_proba(X_eval)
    pred = _predict_labels_from_proba(model, proba)
    draw = _draw_metrics(y_test, pred)
    return {
        "model_name": "CatBoost",
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "log_loss": float(log_loss(y_test, proba, labels=[0, 1, 2])),
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
        **draw,
    }


def _metrics_from_predictions(y_test: np.ndarray, probs: np.ndarray, pred: np.ndarray, model_name: str) -> dict:
    draw = _draw_metrics(y_test, pred)
    return {
        "model_name": model_name,
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "log_loss": float(log_loss(y_test, probs, labels=[0, 1, 2])),
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
        **draw,
    }


def _evaluate_result_model(model_bundle: dict, X_test: pd.DataFrame, y_test: np.ndarray) -> dict:
    probs = _predict_result_proba(model_bundle, X_test)
    pred = _predict_result_labels(model_bundle, probs)
    return _metrics_from_predictions(y_test, probs, pred, model_bundle.get("model_name", "CatBoost"))


def _threshold_candidates() -> np.ndarray:
    if DRAW_THRESHOLD_STEP <= 0:
        return np.array([DRAW_PROB_THRESHOLD], dtype=np.float64)
    count = int(round((DRAW_THRESHOLD_MAX - DRAW_THRESHOLD_MIN) / DRAW_THRESHOLD_STEP)) + 1
    return np.round(DRAW_THRESHOLD_MIN + (np.arange(max(count, 1)) * DRAW_THRESHOLD_STEP), 4)


def _calibrate_draw_threshold(y_true: np.ndarray, probs: np.ndarray) -> dict:
    rows = []
    for threshold in _threshold_candidates():
        decisive_labels = np.where(probs[:, 2] >= probs[:, 0], 2, 0)
        pred = np.where(probs[:, 1] >= threshold, 1, decisive_labels).astype(np.int32)
        metrics = _metrics_from_predictions(y_true, probs, pred, "threshold_candidate")
        rows.append(
            {
                "threshold": float(threshold),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "draw_f1": metrics["draw_f1"],
                "draw_recall": metrics["draw_recall"],
                "predicted_draws": metrics["predicted_draws"],
            }
        )

    eligible = [row for row in rows if row["accuracy"] >= DRAW_CALIBRATION_MIN_ACCURACY]
    pool = eligible or rows
    best = max(pool, key=lambda row: (row["macro_f1"], row["draw_f1"], row["accuracy"]))
    return {
        "threshold": float(best["threshold"]),
        "min_accuracy": DRAW_CALIBRATION_MIN_ACCURACY,
        "used_accuracy_guard": bool(eligible),
        "metrics": {
            "accuracy": round(float(best["accuracy"]), 4),
            "macro_f1": round(float(best["macro_f1"]), 4),
            "draw_f1": round(float(best["draw_f1"]), 4),
            "draw_recall": round(float(best["draw_recall"]), 4),
            "predicted_draws": int(best["predicted_draws"]),
        },
    }


def _calibrate_result_threshold_from_training(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    train_dates: pd.Series,
    train_weights: np.ndarray | None,
    validation_year: int,
) -> dict:
    if not USE_TWO_STAGE_RESULT:
        return {
            "threshold": DRAW_PROB_THRESHOLD,
            "calibration_year": None,
            "source": "multiclass_default",
            "metrics": {},
        }

    wc_mask = X_train["tournament"].eq(normalize_tournament_name("FIFA World Cup"))
    years = sorted(int(year) for year in train_dates[wc_mask].dt.year.unique() if int(year) < validation_year)
    if not years:
        return {
            "threshold": DRAW_PROB_THRESHOLD,
            "calibration_year": None,
            "source": "default_no_previous_world_cup",
            "metrics": {},
        }

    calibration_year = years[-1]
    cal_mask = (train_dates.dt.year == calibration_year).to_numpy() & wc_mask.to_numpy()
    inner_train_mask = (train_dates.dt.year < calibration_year).to_numpy()
    if cal_mask.sum() == 0 or inner_train_mask.sum() < 100 or (y_train[cal_mask] == 1).sum() == 0:
        return {
            "threshold": DRAW_PROB_THRESHOLD,
            "calibration_year": calibration_year,
            "source": "default_insufficient_calibration_rows",
            "metrics": {},
        }

    inner_weights = train_weights[inner_train_mask] if train_weights is not None else None
    calibration_model = _fit_two_stage_model(
        X_train.loc[inner_train_mask].reset_index(drop=True),
        y_train[inner_train_mask],
        sample_weight=inner_weights,
    )
    cal_probs = _predict_result_proba(calibration_model, X_train.loc[cal_mask].reset_index(drop=True))
    calibration = _calibrate_draw_threshold(y_train[cal_mask], cal_probs)
    calibration.update(
        {
            "calibration_year": calibration_year,
            "source": "previous_world_cup",
            "calibration_rows": int(cal_mask.sum()),
            "calibration_draws": int((y_train[cal_mask] == 1).sum()),
        }
    )
    return calibration


def train_and_save(verbose: bool = True) -> dict:
    """
    Full train pipeline. Returns meta dict and saves artifacts to model/artifacts/.
    """
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    if verbose:
        print("Loading data...")
    include_copa = _env_bool("ML_PRJCT_INCLUDE_COPA", True)
    include_friendlies = _env_bool("ML_PRJCT_INCLUDE_FRIENDLIES", True)
    include_friendlies_train = _env_bool("ML_PRJCT_INCLUDE_FRIENDLIES_TRAIN", False)

    if USE_HISTORICAL_WEIGHTED:
        matches_df = _load_historical_matches()
        goalscorers_df = _load_historical_goalscorers()
        shootouts_df = _load_historical_shootouts()
    else:
        matches_df = load_matches(include_copa=include_copa, include_friendlies=include_friendlies)
        goalscorers_df = load_goalscorers()
        shootouts_df = load_shootouts()
    rankings_df = load_rankings()
    substitutions_df = None if USE_HISTORICAL_WEIGHTED and HISTORICAL_FAST_FEATURES else load_substitutions()
    player_appearances_df = None if USE_HISTORICAL_WEIGHTED and HISTORICAL_FAST_FEATURES else load_player_appearances()
    player_goals_df = None if USE_HISTORICAL_WEIGHTED and HISTORICAL_FAST_FEATURES else load_player_goals()
    award_winners_df = None if USE_HISTORICAL_WEIGHTED and HISTORICAL_FAST_FEATURES else load_award_winners()
    copa_data = load_copa_america_data()
    euro_data = load_euro_2024_data()
    friendlies_data = load_friendlies_data()
    if not include_copa:
        copa_data = None
    if not include_friendlies:
        friendlies_data = None
    elif USE_HISTORICAL_WEIGHTED:
        matches_df = append_friendlies_matches(matches_df, friendlies_data)

    if matches_df is None:
        raise RuntimeError("matches.csv is required for training. Drop it into the project root.")

    train_from_year = 1872 if USE_HISTORICAL_WEIGHTED else TRAIN_FROM_YEAR
    candidates = _training_matches(
        matches_df,
        include_curated_friendlies=include_friendlies_train,
        train_from_year=train_from_year,
    )
    validation_source = candidates[
        (candidates["date"].dt.year == VALIDATION_YEAR)
        & (
            candidates["_is_wc"].fillna(False)
            if "_is_wc" in candidates.columns
            else candidates["tournament"].str.contains("world cup", case=False, na=False)
        )
    ].copy()
    train_source = candidates[candidates["date"].dt.year < VALIDATION_YEAR].copy()
    if USE_HISTORICAL_WEIGHTED:
        train_source = _select_historical_training_source(train_source, validation_year=VALIDATION_YEAR)
    final_source = _select_historical_training_source(candidates, validation_year=-1) if USE_HISTORICAL_WEIGHTED else candidates

    if validation_source.empty:
        raise RuntimeError(f"No World Cup validation rows found for {VALIDATION_YEAR}.")

    skip_expensive = USE_HISTORICAL_WEIGHTED and HISTORICAL_FAST_FEATURES

    X_train, y_train, train_dates, train_weights = build_training_data(
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
        euro_data,
        include_curated_friendlies=include_friendlies_train,
        source_matches_df=train_source,
        skip_goalscoring=skip_expensive,
        skip_supersub=skip_expensive,
        skip_player_stats=skip_expensive,
        return_weights=True,
        verbose=verbose,
    )
    X_test, y_test, test_dates = build_training_data(
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
        euro_data,
        include_curated_friendlies=include_friendlies_train,
        source_matches_df=validation_source,
        skip_goalscoring=skip_expensive,
        skip_supersub=skip_expensive,
        skip_player_stats=skip_expensive,
        verbose=verbose,
    )
    X_final, y_final, final_dates, final_weights = build_training_data(
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
        euro_data,
        include_curated_friendlies=include_friendlies_train,
        source_matches_df=final_source,
        skip_goalscoring=skip_expensive,
        skip_supersub=skip_expensive,
        skip_player_stats=skip_expensive,
        return_weights=True,
        verbose=verbose,
    )
    validation_year = VALIDATION_YEAR

    if verbose:
        print(f"\nValidation year: {validation_year}")
        print("Training model candidates: CatBoost" + (" + TwoStageCatBoost" if USE_TWO_STAGE_RESULT else ""))

    default_threshold_calibration = {
        "threshold": DRAW_PROB_THRESHOLD,
        "calibration_year": None,
        "source": "default_multiclass_rule",
        "metrics": {},
    }
    candidates_to_compare = []

    multiclass_model = _fit_multiclass_result_model(X_train, y_train, sample_weight=train_weights)
    multiclass_metrics = _evaluate_result_model(multiclass_model, X_test, y_test)
    candidates_to_compare.append((multiclass_model, multiclass_metrics, default_threshold_calibration))

    if USE_TWO_STAGE_RESULT:
        threshold_calibration = _calibrate_result_threshold_from_training(
            X_train,
            y_train,
            train_dates,
            train_weights,
            validation_year,
        )
        two_stage_model = _fit_two_stage_model(X_train, y_train, sample_weight=train_weights)
        two_stage_model["draw_threshold"] = threshold_calibration["threshold"]
        two_stage_metrics = _evaluate_result_model(two_stage_model, X_test, y_test)
        candidates_to_compare.append((two_stage_model, two_stage_metrics, threshold_calibration))

    validation_model, metrics, threshold_calibration = max(
        candidates_to_compare,
        key=lambda item: (item[1]["macro_f1"], item[1]["accuracy"], item[1]["draw_f1"]),
    )

    if verbose:
        for candidate_model, candidate_metrics, candidate_threshold in candidates_to_compare:
            print(
                f"{candidate_model['model_name']}: "
                f"accuracy={candidate_metrics['accuracy']:.4f}, "
                f"macro_f1={candidate_metrics['macro_f1']:.4f}, "
                f"draw_f1={candidate_metrics['draw_f1']:.4f}, "
                f"threshold={candidate_threshold['threshold']:.2f}"
            )
        print(
            f"Selected {validation_model['model_name']}: "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"balanced_accuracy={metrics['balanced_accuracy']:.4f}, "
            f"log_loss={metrics['log_loss']:.4f}, "
            f"macro_f1={metrics['macro_f1']:.4f}, "
            f"draw_f1={metrics['draw_f1']:.4f}"
        )
        print(
            "Draw threshold: "
            f"{threshold_calibration['threshold']:.2f} "
            f"({threshold_calibration.get('source', 'default')}, "
            f"year={threshold_calibration.get('calibration_year')})"
        )

    if verbose:
        probs = _predict_result_proba(validation_model, X_test)
        pred = _predict_result_labels(validation_model, probs)
        print(f"\nSelected model: {validation_model['model_name']} (macro F1: {metrics['macro_f1']:.4f})")
        print(classification_report(y_test, pred, target_names=["Team A Loss", "Draw", "Team A Win"]))
        print("Confusion matrix:")
        print(confusion_matrix(y_test, pred))

    if validation_model["result_model_type"] == "two_stage":
        best_result = _fit_two_stage_model(X_final, y_final, sample_weight=final_weights)
    else:
        best_result = _fit_multiclass_result_model(X_final, y_final, sample_weight=final_weights)
    best_result["draw_threshold"] = threshold_calibration["threshold"]

    model_bundle = {
        "model": best_result.get("model") or best_result.get("draw_model"),
        "model_name": best_result["model_name"],
        "result_model_type": best_result["result_model_type"],
        "draw_model": best_result.get("draw_model"),
        "decisive_model": best_result.get("decisive_model"),
        "draw_threshold": best_result["draw_threshold"],
        "threshold_calibration": threshold_calibration,
        "feature_columns": MODEL_INPUT_COLUMNS,
        "categorical_columns": RAW_CONTEXT_COLUMNS,
        "raw_context_columns": RAW_CONTEXT_COLUMNS,
        "numeric_feature_columns": FEATURE_COLUMNS,
    }
    joblib.dump(model_bundle, MODEL_PATH)
    joblib.dump(None, SCALER_PATH)  # Backwards-compatible placeholder.
    with open(FEATURES_PATH, "w") as f:
        json.dump(MODEL_INPUT_COLUMNS, f, indent=2)

    all_dates = pd.concat([train_dates, test_dates, final_dates], ignore_index=True)
    data_min = all_dates.min()
    data_max = all_dates.max()
    ranking_dates = rankings_df["rank_date"] if rankings_df is not None and "rank_date" in rankings_df.columns else None
    meta = {
        "model_name": best_result["model_name"],
        "accuracy": round(metrics["accuracy"], 4),
        "balanced_accuracy": round(metrics["balanced_accuracy"], 4),
        "log_loss": round(metrics["log_loss"], 4),
        "macro_f1": round(metrics["macro_f1"], 4),
        "weighted_f1": round(metrics["weighted_f1"], 4),
        "validation_year": validation_year,
        "training_rows": int(len(X_final)),
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_test)),
        "features": int(len(MODEL_INPUT_COLUMNS)),
        "numeric_features": int(len(FEATURE_COLUMNS)),
        "categorical_features": int(len(RAW_CONTEXT_COLUMNS)),
        "train_from_year": 1872 if USE_HISTORICAL_WEIGHTED else TRAIN_FROM_YEAR,
        "data_start": str(data_min.date()) if pd.notna(data_min) else None,
        "data_end": str(data_max.date()) if pd.notna(data_max) else None,
        "ranking_rows": int(len(rankings_df)) if rankings_df is not None else 0,
        "ranking_start": str(ranking_dates.min().date()) if ranking_dates is not None and not ranking_dates.empty else None,
        "ranking_end": str(ranking_dates.max().date()) if ranking_dates is not None and not ranking_dates.empty else None,
        "validation_metrics": {
            "accuracy": round(metrics["accuracy"], 4),
            "balanced_accuracy": round(metrics["balanced_accuracy"], 4),
            "log_loss": round(metrics["log_loss"], 4),
            "macro_f1": round(metrics["macro_f1"], 4),
            "weighted_f1": round(metrics["weighted_f1"], 4),
            "draw_precision": round(metrics["draw_precision"], 4),
            "draw_recall": round(metrics["draw_recall"], 4),
            "draw_f1": round(metrics["draw_f1"], 4),
            "predicted_draws": metrics["predicted_draws"],
            "actual_draws": metrics["actual_draws"],
        },
        "model_policy": "fixed_catboost",
        "result_model_type": best_result["result_model_type"],
        "candidate_selection_policy": "highest_macro_f1_then_accuracy_then_draw_f1",
        "validation_target": f"world_cup_{VALIDATION_YEAR}",
        "historical_weighted": USE_HISTORICAL_WEIGHTED,
        "historical_source_cap": HISTORICAL_SOURCE_CAP if USE_HISTORICAL_WEIGHTED else 0,
        "historical_fast_features": skip_expensive,
        "euro_2024_player_tables_loaded": int(euro_data.get("tables_loaded", 0)) if euro_data else 0,
        "euro_2024_player_rows": int(len(euro_data.get("players"))) if euro_data and euro_data.get("players") is not None else 0,
        "euro_2024_match_team_rows": (
            int(len(euro_data.get("match_team_stats")))
            if euro_data and euro_data.get("match_team_stats") is not None
            else 0
        ),
        "euro_2024_goal_assist_team_rows": (
            int(len(euro_data.get("goal_assist_team_stats")))
            if euro_data and euro_data.get("goal_assist_team_stats") is not None
            else 0
        ),
        "sample_weight_policy": "era_weight * tournament_weight" if USE_HISTORICAL_WEIGHTED else "tournament_weight",
        "era_weight_policy": {
            "2018_present": 1.25,
            "2006_2017": 1.0,
            "2002_2005": 0.45,
            "pre_2002": 0.08,
        } if USE_HISTORICAL_WEIGHTED else None,
        "draw_policy": {
            "class_weight": DRAW_CLASS_WEIGHT,
            "binary_class_weight": DRAW_BINARY_WEIGHT,
            "probability_threshold": DRAW_PROB_THRESHOLD,
            "calibrated_threshold": round(float(threshold_calibration["threshold"]), 4),
            "close_margin": DRAW_CLOSE_MARGIN,
            "threshold_calibration": threshold_calibration,
        },
        "venue_policy": "Venue features are trained from home/away/neutral match context.",
        "include_copa": include_copa,
        "include_friendlies": include_friendlies,
        "include_friendlies_train": include_friendlies_train,
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    if verbose:
        print(f"\nArtifacts saved to {ARTIFACTS_DIR}")
        print(
            f"Model: {best_result['model_name']} | "
            f"Accuracy: {metrics['accuracy']:.1%} | "
            f"Macro F1: {metrics['macro_f1']:.1%} | "
            f"Rows: {len(X_final)}"
        )

    return meta


if __name__ == "__main__":
    train_and_save(verbose=True)
