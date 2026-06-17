"""Goal prediction model for expected scorelines.

This module is intentionally separate from the match-result classifier. The
classifier predicts win/draw/loss; this model predicts Team A and Team B goals
from a Team-A-perspective row where Team A is treated as the home side at
inference time.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import defaultdict
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error

try:
    from catboost import CatBoostRegressor

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

from data.ingest import MAX_DATA_DATE, MIN_DATA_YEAR, load_rankings
from data.world_cup_2026 import append_world_cup_2026_matches
from model.features import get_tournament_weight, normalize_tournament_name


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
GOALS_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "goals_model.pkl")
GOALS_META_PATH = os.path.join(ARTIFACTS_DIR, "goals_meta.json")

RANDOM_STATE = 42
GOAL_VALIDATION_YEAR = None
SCORELINE_MAX_GOALS = 10

GOAL_CATEGORICAL_COLUMNS = ["team_a", "team_b", "tournament", "host_country"]
GOAL_NUMERIC_COLUMNS = [
    "team_a_is_home",
    "team_b_is_home",
    "neutral_venue",
    "team_a_host_country",
    "team_b_host_country",
    "tournament_weight",
    "team_a_matches_count",
    "team_a_recent_goals_for",
    "team_a_recent_goals_against",
    "team_a_recent_win_rate",
    "team_a_recent_draw_rate",
    "team_a_recent_clean_sheet_rate",
    "team_a_recent_over_2_5_rate",
    "team_a_home_goals_for",
    "team_a_home_goals_against",
    "team_a_away_goals_for",
    "team_a_away_goals_against",
    "team_b_matches_count",
    "team_b_recent_goals_for",
    "team_b_recent_goals_against",
    "team_b_recent_win_rate",
    "team_b_recent_draw_rate",
    "team_b_recent_clean_sheet_rate",
    "team_b_recent_over_2_5_rate",
    "team_b_home_goals_for",
    "team_b_home_goals_against",
    "team_b_away_goals_for",
    "team_b_away_goals_against",
    "h2h_matches",
    "h2h_team_a_goals_for",
    "h2h_team_b_goals_for",
    "h2h_over_2_5_rate",
    "team_a_rank",
    "team_b_rank",
    "rank_diff",
    "team_a_rank_points",
    "team_b_rank_points",
]
GOAL_INPUT_COLUMNS = GOAL_CATEGORICAL_COLUMNS + GOAL_NUMERIC_COLUMNS


def _goal_matches_path() -> str:
    processed = os.path.join(ROOT, "data", "processed", f"matches_{MIN_DATA_YEAR}.csv")
    return processed if os.path.exists(processed) else os.path.join(ROOT, "matches.csv")


@lru_cache(maxsize=1)
def load_goal_matches() -> pd.DataFrame:
    """Load the broad 2006+ match-score dataset used by the goal model."""
    path = _goal_matches_path()
    if not os.path.exists(path):
        raise RuntimeError("No match-score data found for goal prediction.")

    df = pd.read_csv(path)
    required = ["date", "home_team", "away_team", "home_score", "away_score", "tournament"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise RuntimeError(f"Goal training data is missing columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team"])
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["tournament"] = df["tournament"].fillna("Unknown").astype(str)
    if "country" in df.columns:
        df["country"] = df["country"].fillna("").astype(str)
    else:
        df["country"] = ""
    if "neutral" in df.columns:
        neutral = df["neutral"].fillna(False)
        if neutral.dtype == object:
            df["neutral"] = neutral.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})
        else:
            df["neutral"] = neutral.astype(bool)
    else:
        df["neutral"] = False
    mask = (df["date"].dt.year >= MIN_DATA_YEAR) & (df["date"] <= MAX_DATA_DATE)
    return df[mask].sort_values("date").reset_index(drop=True)


def _load_goal_inference_matches() -> pd.DataFrame:
    matches = append_world_cup_2026_matches(load_goal_matches())
    return matches.sort_values("date").reset_index(drop=True)


def _ranking_index(rankings_df: pd.DataFrame | None) -> dict:
    if rankings_df is None or rankings_df.empty:
        return {}
    indexed = {}
    for team, group in rankings_df.dropna(subset=["rank_date", "country_full"]).groupby("country_full"):
        g = group.sort_values("rank_date")
        indexed[str(team)] = {
            "dates": pd.to_datetime(g["rank_date"]).to_numpy(),
            "ranks": pd.to_numeric(g["rank"], errors="coerce").fillna(0).to_numpy(),
            "points": pd.to_numeric(g["total_points"], errors="coerce").fillna(0).to_numpy(),
        }
    return indexed


def _rank_for(team: str, match_date, rankings: dict) -> tuple[float, float]:
    data = rankings.get(team)
    if not data:
        return 0.0, 0.0
    dates = data["dates"]
    idx = np.searchsorted(dates, np.datetime64(pd.Timestamp(match_date)), side="right") - 1
    if idx < 0:
        return 0.0, 0.0
    return float(data["ranks"][idx]), float(data["points"][idx])


def _avg(values: list[float], default: float = 0.0) -> float:
    return float(np.mean(values)) if values else default


def _team_summary(team: str, history: dict[str, list[dict]], n: int = 10) -> dict:
    rows = history.get(team, [])
    recent = rows[-n:]
    if not recent:
        return {
            "matches_count": 0.0,
            "recent_goals_for": 0.0,
            "recent_goals_against": 0.0,
            "recent_win_rate": 0.0,
            "recent_draw_rate": 0.0,
            "recent_clean_sheet_rate": 0.0,
            "recent_over_2_5_rate": 0.0,
            "home_goals_for": 0.0,
            "home_goals_against": 0.0,
            "away_goals_for": 0.0,
            "away_goals_against": 0.0,
        }

    home_rows = [row for row in rows if row["is_home"] and not row["neutral"]][-n:]
    away_rows = [row for row in rows if (not row["is_home"]) and not row["neutral"]][-n:]
    return {
        "matches_count": float(len(rows)),
        "recent_goals_for": _avg([row["gf"] for row in recent]),
        "recent_goals_against": _avg([row["ga"] for row in recent]),
        "recent_win_rate": _avg([1.0 if row["gf"] > row["ga"] else 0.0 for row in recent]),
        "recent_draw_rate": _avg([1.0 if row["gf"] == row["ga"] else 0.0 for row in recent]),
        "recent_clean_sheet_rate": _avg([1.0 if row["ga"] == 0 else 0.0 for row in recent]),
        "recent_over_2_5_rate": _avg([1.0 if row["gf"] + row["ga"] > 2.5 else 0.0 for row in recent]),
        "home_goals_for": _avg([row["gf"] for row in home_rows]),
        "home_goals_against": _avg([row["ga"] for row in home_rows]),
        "away_goals_for": _avg([row["gf"] for row in away_rows]),
        "away_goals_against": _avg([row["ga"] for row in away_rows]),
    }


def _h2h_summary(team_a: str, team_b: str, h2h: dict[frozenset, list[dict]], n: int = 10) -> dict:
    rows = h2h.get(frozenset([team_a, team_b]), [])[-n:]
    if not rows:
        return {
            "h2h_matches": 0.0,
            "h2h_team_a_goals_for": 0.0,
            "h2h_team_b_goals_for": 0.0,
            "h2h_over_2_5_rate": 0.0,
        }

    goals_a = []
    goals_b = []
    over = []
    for row in rows:
        if row["home_team"] == team_a:
            ga, gb = row["home_score"], row["away_score"]
        elif row["away_team"] == team_a:
            ga, gb = row["away_score"], row["home_score"]
        else:
            continue
        goals_a.append(ga)
        goals_b.append(gb)
        over.append(1.0 if ga + gb > 2.5 else 0.0)

    return {
        "h2h_matches": float(len(goals_a)),
        "h2h_team_a_goals_for": _avg(goals_a),
        "h2h_team_b_goals_for": _avg(goals_b),
        "h2h_over_2_5_rate": _avg(over),
    }


def _venue_context(row: pd.Series, team_a: str, team_b: str) -> dict:
    neutral = bool(row.get("neutral", False))
    country = str(row.get("country", "") or "")
    team_a_is_home = float((not neutral) and row["home_team"] == team_a)
    team_b_is_home = float((not neutral) and row["home_team"] == team_b)
    return {
        "team_a_is_home": team_a_is_home,
        "team_b_is_home": team_b_is_home,
        "neutral_venue": float(neutral),
        "team_a_host_country": float(country == team_a),
        "team_b_host_country": float(country == team_b),
        "host_country": country or "unknown",
    }


def _inference_venue_context(team_a: str, team_b: str, venue_mode: str) -> dict:
    if venue_mode == "team_b_home":
        return {
            "team_a_is_home": 0.0,
            "team_b_is_home": 1.0,
            "neutral_venue": 0.0,
            "team_a_host_country": 0.0,
            "team_b_host_country": 1.0,
            "host_country": team_b,
        }
    if venue_mode == "neutral":
        return {
            "team_a_is_home": 0.0,
            "team_b_is_home": 0.0,
            "neutral_venue": 1.0,
            "team_a_host_country": 0.0,
            "team_b_host_country": 0.0,
            "host_country": "neutral",
        }
    return {
        "team_a_is_home": 1.0,
        "team_b_is_home": 0.0,
        "neutral_venue": 0.0,
        "team_a_host_country": 1.0,
        "team_b_host_country": 0.0,
        "host_country": team_a,
    }


def _build_goal_row(
    team_a: str,
    team_b: str,
    tournament: str,
    match_date,
    venue: dict,
    history: dict[str, list[dict]],
    h2h: dict[frozenset, list[dict]],
    rankings: dict,
) -> dict:
    row = {
        "team_a": team_a,
        "team_b": team_b,
        "tournament": normalize_tournament_name(tournament),
        "host_country": venue.get("host_country") or "unknown",
        "tournament_weight": get_tournament_weight(tournament),
        "team_a_is_home": float(venue.get("team_a_is_home", 0.0)),
        "team_b_is_home": float(venue.get("team_b_is_home", 0.0)),
        "neutral_venue": float(venue.get("neutral_venue", 1.0)),
        "team_a_host_country": float(venue.get("team_a_host_country", 0.0)),
        "team_b_host_country": float(venue.get("team_b_host_country", 0.0)),
    }

    for prefix, team in [("team_a", team_a), ("team_b", team_b)]:
        summary = _team_summary(team, history)
        for key, value in summary.items():
            row[f"{prefix}_{key}"] = value

    row.update(_h2h_summary(team_a, team_b, h2h))
    rank_a, points_a = _rank_for(team_a, match_date, rankings)
    rank_b, points_b = _rank_for(team_b, match_date, rankings)
    row["team_a_rank"] = rank_a
    row["team_b_rank"] = rank_b
    row["rank_diff"] = rank_b - rank_a if rank_a > 0 and rank_b > 0 else 0.0
    row["team_a_rank_points"] = points_a
    row["team_b_rank_points"] = points_b

    for col in GOAL_NUMERIC_COLUMNS:
        row[col] = float(row.get(col, 0.0) or 0.0)
    for col in GOAL_CATEGORICAL_COLUMNS:
        row[col] = str(row.get(col, "unknown") or "unknown")
    return {col: row[col] for col in GOAL_INPUT_COLUMNS}


def _append_match_to_history(row: pd.Series, history: dict[str, list[dict]], h2h: dict[frozenset, list[dict]]) -> None:
    home = row["home_team"]
    away = row["away_team"]
    home_score = int(row["home_score"])
    away_score = int(row["away_score"])
    neutral = bool(row.get("neutral", False))
    history[home].append({"gf": home_score, "ga": away_score, "is_home": True, "neutral": neutral})
    history[away].append({"gf": away_score, "ga": home_score, "is_home": False, "neutral": neutral})
    h2h[frozenset([home, away])].append(
        {
            "home_team": home,
            "away_team": away,
            "home_score": home_score,
            "away_score": away_score,
        }
    )


def build_goal_training_data(
    matches_df: pd.DataFrame,
    rankings_df: pd.DataFrame | None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    rows = []
    targets = []
    dates = []
    history: dict[str, list[dict]] = defaultdict(list)
    h2h: dict[frozenset, list[dict]] = defaultdict(list)
    rankings = _ranking_index(rankings_df)
    t0 = time.time()

    for i, (_, match) in enumerate(matches_df.sort_values("date").iterrows()):
        home = match["home_team"]
        away = match["away_team"]
        tournament = match["tournament"]
        match_date = match["date"]

        rows.append(
            _build_goal_row(
                home,
                away,
                tournament,
                match_date,
                _venue_context(match, home, away),
                history,
                h2h,
                rankings,
            )
        )
        targets.append([int(match["home_score"]), int(match["away_score"])])
        dates.append(match_date)

        rows.append(
            _build_goal_row(
                away,
                home,
                tournament,
                match_date,
                _venue_context(match, away, home),
                history,
                h2h,
                rankings,
            )
        )
        targets.append([int(match["away_score"]), int(match["home_score"])])
        dates.append(match_date)

        _append_match_to_history(match, history, h2h)

        if verbose and (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{len(matches_df)} matches - {time.time() - t0:.1f}s elapsed")

    X = pd.DataFrame(rows, columns=GOAL_INPUT_COLUMNS)
    for col in GOAL_CATEGORICAL_COLUMNS:
        X[col] = X[col].fillna("unknown").astype(str)
    for col in GOAL_NUMERIC_COLUMNS:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)
    return X, np.asarray(targets, dtype=np.float32), pd.to_datetime(pd.Series(dates))


def _time_split(dates: pd.Series, validation_year: int | None = None) -> tuple[np.ndarray, np.ndarray, int]:
    years = sorted(dates.dt.year.unique())
    validation_year = validation_year or GOAL_VALIDATION_YEAR or years[-1]
    train_mask = (dates.dt.year < validation_year).to_numpy()
    test_mask = (dates.dt.year == validation_year).to_numpy()
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        cutoff = dates.quantile(0.8)
        train_mask = (dates < cutoff).to_numpy()
        test_mask = (dates >= cutoff).to_numpy()
        validation_year = int(pd.Timestamp(cutoff).year)
    return train_mask, test_mask, int(validation_year)


def _goal_model() -> CatBoostRegressor:
    if not HAS_CATBOOST:
        raise RuntimeError("CatBoost is required. Install dependencies with `pip install -r requirements.txt`.")
    return CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=600,
        depth=6,
        learning_rate=0.04,
        l2_leaf_reg=6,
        random_seed=RANDOM_STATE,
        verbose=False,
    )


def _fit_model(X: pd.DataFrame, y: np.ndarray) -> CatBoostRegressor:
    model = _goal_model()
    model.fit(X[GOAL_INPUT_COLUMNS], y, cat_features=GOAL_CATEGORICAL_COLUMNS)
    return model


def _scoreline_from_expected(team_a_goals: float, team_b_goals: float, max_goals: int = SCORELINE_MAX_GOALS) -> dict:
    lam_a = max(0.05, float(team_a_goals))
    lam_b = max(0.05, float(team_b_goals))
    rounded_a = int(min(max_goals, max(0, math.floor(lam_a + 0.5))))
    rounded_b = int(min(max_goals, max(0, math.floor(lam_b + 0.5))))
    rounded_probability = 0.0
    best = (0, 0, -1.0)
    best_by_outcome = {
        "team_a_win": (1, 0, -1.0),
        "draw": (0, 0, -1.0),
        "team_b_win": (0, 1, -1.0),
    }
    outcome_probs = {
        "win_prob": 0.0,
        "draw_prob": 0.0,
        "loss_prob": 0.0,
    }
    over_2_5 = 0.0
    grid_mass = 0.0
    for a in range(max_goals + 1):
        pa = math.exp(-lam_a) * (lam_a**a) / math.factorial(a)
        for b in range(max_goals + 1):
            pb = math.exp(-lam_b) * (lam_b**b) / math.factorial(b)
            prob = pa * pb
            grid_mass += prob
            if a == rounded_a and b == rounded_b:
                rounded_probability += prob
            if a + b > 2.5:
                over_2_5 += prob
            if prob > best[2]:
                best = (a, b, prob)

            if a > b:
                outcome = "team_a_win"
                outcome_probs["win_prob"] += prob
            elif b > a:
                outcome = "team_b_win"
                outcome_probs["loss_prob"] += prob
            else:
                outcome = "draw"
                outcome_probs["draw_prob"] += prob
            if prob > best_by_outcome[outcome][2]:
                best_by_outcome[outcome] = (a, b, prob)

    if grid_mass > 0:
        over_2_5 /= grid_mass
        rounded_probability /= grid_mass
        for key in outcome_probs:
            outcome_probs[key] = float(outcome_probs[key] / grid_mass)
        best = (best[0], best[1], best[2] / grid_mass)
        best_by_outcome = {
            key: (score[0], score[1], score[2] / grid_mass)
            for key, score in best_by_outcome.items()
        }

    scorelines_by_outcome = {
        key: {
            "team_a_goals": int(score[0]),
            "team_b_goals": int(score[1]),
            "probability": float(score[2]),
            "scoreline": f"{int(score[0])}-{int(score[1])}",
        }
        for key, score in best_by_outcome.items()
    }
    return {
        "likely_team_a_goals": int(best[0]),
        "likely_team_b_goals": int(best[1]),
        "likely_score_probability": float(best[2]),
        "rounded_expected_team_a_goals": rounded_a,
        "rounded_expected_team_b_goals": rounded_b,
        "rounded_expected_scoreline": f"{rounded_a}-{rounded_b}",
        "rounded_expected_score_probability": float(rounded_probability),
        "over_2_5_probability": float(over_2_5),
        "under_2_5_probability": float(1.0 - over_2_5),
        "goal_outcome_probabilities": outcome_probs,
        "scorelines_by_outcome": scorelines_by_outcome,
        "goal_probability_grid_mass": float(grid_mass),
    }


def _evaluate(
    model_a: CatBoostRegressor,
    model_b: CatBoostRegressor,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> dict:
    pred_a = np.clip(model_a.predict(X_test[GOAL_INPUT_COLUMNS]), 0.0, 10.0)
    pred_b = np.clip(model_b.predict(X_test[GOAL_INPUT_COLUMNS]), 0.0, 10.0)
    pred = np.vstack([pred_a, pred_b]).T
    rounded = np.rint(pred).clip(0, 10).astype(int)
    actual = y_test.astype(int)
    return {
        "team_a_mae": float(mean_absolute_error(actual[:, 0], pred[:, 0])),
        "team_b_mae": float(mean_absolute_error(actual[:, 1], pred[:, 1])),
        "combined_mae": float(mean_absolute_error(actual, pred)),
        "team_a_rmse": float(np.sqrt(mean_squared_error(actual[:, 0], pred[:, 0]))),
        "team_b_rmse": float(np.sqrt(mean_squared_error(actual[:, 1], pred[:, 1]))),
        "exact_score_accuracy": float(np.mean((rounded[:, 0] == actual[:, 0]) & (rounded[:, 1] == actual[:, 1]))),
        "over_2_5_accuracy": float(accuracy_score((actual.sum(axis=1) > 2.5), (pred.sum(axis=1) > 2.5))),
    }


def train_and_save_goals(verbose: bool = True) -> dict:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    if verbose:
        print("Loading goal training data...")
    matches_df = load_goal_matches()
    rankings_df = load_rankings()
    if verbose:
        print(f"Building goal features for {len(matches_df)} matches...")
    X, y, dates = build_goal_training_data(matches_df, rankings_df, verbose=verbose)
    train_mask, test_mask, validation_year = _time_split(dates)
    X_train, X_test = X.loc[train_mask], X.loc[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    if verbose:
        print(f"Goal validation year: {validation_year}")
        print("Training Team A goals model...")
    model_a = _fit_model(X_train, y_train[:, 0])
    if verbose:
        print("Training Team B goals model...")
    model_b = _fit_model(X_train, y_train[:, 1])
    metrics = _evaluate(model_a, model_b, X_test, y_test)

    final_a = _fit_model(X, y[:, 0])
    final_b = _fit_model(X, y[:, 1])
    bundle = {
        "team_a_model": final_a,
        "team_b_model": final_b,
        "feature_columns": GOAL_INPUT_COLUMNS,
        "categorical_columns": GOAL_CATEGORICAL_COLUMNS,
        "numeric_columns": GOAL_NUMERIC_COLUMNS,
    }
    joblib.dump(bundle, GOALS_MODEL_PATH)

    meta = {
        "model_name": "CatBoostRegressor",
        "validation_year": validation_year,
        "training_rows": int(len(X)),
        "train_rows": int(train_mask.sum()),
        "validation_rows": int(test_mask.sum()),
        "features": int(len(GOAL_INPUT_COLUMNS)),
        "train_from_year": MIN_DATA_YEAR,
        "data_start": str(dates.min().date()) if not dates.empty else None,
        "data_end": str(dates.max().date()) if not dates.empty else None,
        "metrics": {key: round(value, 4) for key, value in metrics.items()},
        "venue_policy": "App predictions use the selected Team A home, Team B home, or neutral venue context.",
        "data_source": os.path.relpath(_goal_matches_path(), ROOT),
    }
    with open(GOALS_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    if verbose:
        print(json.dumps(meta, indent=2))
        print(f"Goal artifacts saved to {ARTIFACTS_DIR}")
    return meta


@lru_cache(maxsize=1)
def _load_or_train_bundle() -> tuple[dict, dict]:
    if not os.path.exists(GOALS_MODEL_PATH) or not os.path.exists(GOALS_META_PATH):
        train_and_save_goals(verbose=False)
    bundle = joblib.load(GOALS_MODEL_PATH)
    with open(GOALS_META_PATH) as f:
        meta = json.load(f)
    return bundle, meta


def _history_until(matches_df: pd.DataFrame, as_of_date) -> tuple[dict[str, list[dict]], dict[frozenset, list[dict]]]:
    history: dict[str, list[dict]] = defaultdict(list)
    h2h: dict[frozenset, list[dict]] = defaultdict(list)
    cutoff = pd.Timestamp(as_of_date)
    for _, row in matches_df[matches_df["date"] < cutoff].sort_values("date").iterrows():
        _append_match_to_history(row, history, h2h)
    return history, h2h


def predict_goals(
    team_a: str,
    team_b: str,
    rankings_df: pd.DataFrame | None = None,
    venue_mode: str = "team_a_home",
) -> dict:
    bundle, meta = _load_or_train_bundle()
    matches_df = _load_goal_inference_matches()
    as_of_date = pd.Timestamp.now().normalize() + pd.Timedelta(days=1)
    if "date" in matches_df.columns and not matches_df.empty:
        latest_match_date = pd.to_datetime(matches_df["date"], errors="coerce").max()
        if pd.notna(latest_match_date):
            as_of_date = max(as_of_date, latest_match_date.normalize() + pd.Timedelta(days=1))
    history, h2h = _history_until(matches_df, as_of_date)
    rankings = _ranking_index(rankings_df if rankings_df is not None else load_rankings())

    opposite_venue = {
        "team_a_home": "team_b_home",
        "team_b_home": "team_a_home",
        "neutral": "neutral",
    }.get(venue_mode, "neutral")
    rows = [
        _build_goal_row(
            team_a,
            team_b,
            "FIFA World Cup",
            as_of_date,
            _inference_venue_context(team_a, team_b, venue_mode),
            history,
            h2h,
            rankings,
        ),
        _build_goal_row(
            team_b,
            team_a,
            "FIFA World Cup",
            as_of_date,
            _inference_venue_context(team_b, team_a, opposite_venue),
            history,
            h2h,
            rankings,
        ),
    ]
    X = pd.DataFrame(rows, columns=bundle["feature_columns"])
    for col in bundle["categorical_columns"]:
        X[col] = X[col].fillna("unknown").astype(str)
    for col in bundle["numeric_columns"]:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)

    predicted_a = np.clip(bundle["team_a_model"].predict(X), 0.0, 10.0)
    predicted_b = np.clip(bundle["team_b_model"].predict(X), 0.0, 10.0)
    pred_a = float((predicted_a[0] + predicted_b[1]) / 2.0)
    pred_b = float((predicted_b[0] + predicted_a[1]) / 2.0)
    scoreline = _scoreline_from_expected(pred_a, pred_b)
    displayed_a = round(pred_a, 2)
    displayed_b = round(pred_b, 2)
    return {
        "expected_team_a_goals": displayed_a,
        "expected_team_b_goals": displayed_b,
        "expected_total_goals": round(displayed_a + displayed_b, 2),
        **scoreline,
        "goal_model_meta": meta,
    }


if __name__ == "__main__":
    train_and_save_goals(verbose=True)
