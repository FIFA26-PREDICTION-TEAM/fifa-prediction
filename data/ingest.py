"""
Data ingestion — loads and validates all four CSVs.
All functions decorated with @st.cache_data for Streamlit performance.
Missing files show st.warning() and return None so callers degrade gracefully.
"""

import os
import pandas as pd
import streamlit as st

from data.copa_america import append_copa_matches, load_copa_america_data
from data.euro_2024 import load_euro_2024_data
from data.international_friendlies import append_friendlies_matches, load_friendlies_data
from data.world_cup_2026 import append_world_cup_2026_matches, load_world_cup_2026_data

# CSVs are expected in the project root (one level up from data/)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROCESSED = os.path.join(_ROOT, "data", "processed")
_EXTERNAL = os.path.join(_ROOT, "data", "external")
USE_HISTORICAL_WEIGHTED = os.getenv("ML_PRJCT_USE_HISTORICAL_WEIGHTED", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
MIN_DATA_YEAR = int(os.getenv("ML_PRJCT_MIN_DATA_YEAR", "1872" if USE_HISTORICAL_WEIGHTED else "2006"))
MAX_DATA_DATE = pd.Timestamp(os.getenv("ML_PRJCT_MAX_DATA_DATE", pd.Timestamp.today().date().isoformat()))
USE_PROCESSED_DATA = os.getenv("ML_PRJCT_USE_PROCESSED", "0").strip().lower() in {"1", "true", "yes", "on"}

REQUIRED_COLUMNS = {
    "matches": ["date", "home_team", "away_team", "home_score", "away_score", "tournament"],
    "goalscorers": ["date", "home_team", "away_team", "team", "scorer", "minute"],
    "shootouts": ["date", "home_team", "away_team", "winner"],
    "rankings": ["rank_date", "country_full", "rank", "total_points"],
}
COMPETITIVE_PATTERN = (
    "world cup|copa|euro|afcon|asian cup|qualifier|qualification|"
    "gold cup|nations league|confederation|olympic"
)


def _check_columns(df: pd.DataFrame, required: list[str], filename: str) -> pd.DataFrame:
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.warning(f"⚠️  `{filename}` is missing columns: {missing}. Some features will be unavailable.")
    return df


def _first_existing(*paths: str) -> str | None:
    return next((path for path in paths if os.path.exists(path)), None)


def _filter_date_window(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    if date_col not in df.columns:
        return df
    dates = pd.to_datetime(df[date_col], errors="coerce")
    mask = (dates.dt.year >= MIN_DATA_YEAR) & (dates <= MAX_DATA_DATE)
    return df[mask].reset_index(drop=True)


def _add_tournament_flags(df: pd.DataFrame) -> pd.DataFrame:
    if "tournament" not in df.columns:
        return df
    tournament = df["tournament"].fillna("").astype(str).str.lower()
    df["_is_competitive"] = tournament.str.contains(COMPETITIVE_PATTERN, na=False)
    df["_is_wc"] = tournament.str.contains("world cup", na=False) & ~tournament.str.contains(
        "qualifier|qualification", na=False
    )
    return df


LIVE_DATA_CACHE_TTL_SECONDS = int(os.getenv("ML_PRJCT_LIVE_DATA_CACHE_TTL_SECONDS", "60"))


@st.cache_data(show_spinner=False, ttl=LIVE_DATA_CACHE_TTL_SECONDS)
def load_matches(
    include_copa: bool = True,
    include_friendlies: bool = True,
    include_world_cup_2026: bool = True,
) -> pd.DataFrame | None:
    candidates = [os.path.join(_ROOT, "matches.csv")]
    if USE_HISTORICAL_WEIGHTED:
        candidates.insert(0, os.path.join(_PROCESSED, "matches_all.csv"))
    if USE_PROCESSED_DATA:
        candidates.insert(0, os.path.join(_PROCESSED, f"matches_{MIN_DATA_YEAR}.csv"))
    path = _first_existing(*candidates)
    if not path:
        st.warning("⚠️  `matches.csv` not found. Drop it into the project root to enable predictions.")
        return None
    df = pd.read_csv(path)
    df = _check_columns(df, REQUIRED_COLUMNS["matches"], "matches.csv")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team"])
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce").fillna(0).astype(int)
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce").fillna(0).astype(int)
    df["neutral"] = df.get("neutral", False)
    df["tournament"] = df["tournament"].fillna("Unknown")
    df = df.reset_index(drop=True)
    loaded_historical = os.path.basename(path) == "matches_all.csv"
    if include_copa and not loaded_historical:
        df = append_copa_matches(df)
    if include_friendlies:
        df = append_friendlies_matches(df)
    if include_world_cup_2026:
        df = append_world_cup_2026_matches(df)
    df = _filter_date_window(df).sort_values("date").reset_index(drop=True)
    return _add_tournament_flags(df)


@st.cache_data(show_spinner=False)
def load_goalscorers() -> pd.DataFrame | None:
    candidates = [os.path.join(_ROOT, "goalscorers.csv")]
    if USE_HISTORICAL_WEIGHTED:
        candidates.insert(0, os.path.join(_PROCESSED, "goalscorers_all.csv"))
    if USE_PROCESSED_DATA:
        candidates.insert(0, os.path.join(_PROCESSED, f"goalscorers_{MIN_DATA_YEAR}.csv"))
    path = _first_existing(*candidates)
    if not path:
        st.warning("⚠️  `goalscorers.csv` not found. Super-sub and goalscoring features will be unavailable.")
        return None
    df = pd.read_csv(path)
    df = _check_columns(df, REQUIRED_COLUMNS["goalscorers"], "goalscorers.csv")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team"])
    df["minute"] = pd.to_numeric(df["minute"], errors="coerce")
    df["own_goal"] = df.get("own_goal", False)
    df["penalty"] = df.get("penalty", False)
    return _filter_date_window(df).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_shootouts() -> pd.DataFrame | None:
    candidates = [os.path.join(_ROOT, "shootouts.csv")]
    if USE_HISTORICAL_WEIGHTED:
        candidates.insert(0, os.path.join(_PROCESSED, "shootouts_all.csv"))
    if USE_PROCESSED_DATA:
        candidates.insert(0, os.path.join(_PROCESSED, f"shootouts_{MIN_DATA_YEAR}.csv"))
    path = _first_existing(*candidates)
    if not path:
        st.warning("⚠️  `shootouts.csv` not found. Penalty shootout features will be unavailable.")
        return None
    df = pd.read_csv(path)
    df = _check_columns(df, REQUIRED_COLUMNS["shootouts"], "shootouts.csv")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team", "winner"])
    return _filter_date_window(df).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_rankings() -> pd.DataFrame | None:
    path = os.path.join(_ROOT, "rankings.csv")
    external_path = os.path.join(_EXTERNAL, "copa_archive_2024", "fifa_ranking-2024-04-04.csv")
    if not os.path.exists(path) and not os.path.exists(external_path):
        st.warning("⚠️  `rankings.csv` not found. FIFA ranking features will be unavailable.")
        return None

    frames = []
    for source_path in (external_path, path):
        if os.path.exists(source_path):
            frame = pd.read_csv(source_path)
            frame = _check_columns(frame, REQUIRED_COLUMNS["rankings"], os.path.basename(source_path))
            frames.append(frame[REQUIRED_COLUMNS["rankings"]])
    df = pd.concat(frames, ignore_index=True)
    df["rank_date"] = pd.to_datetime(df["rank_date"], errors="coerce")
    df = df.dropna(subset=["rank_date", "country_full", "rank"])
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df["total_points"] = pd.to_numeric(df["total_points"], errors="coerce")
    df = df.drop_duplicates(["rank_date", "country_full"], keep="last")
    return _filter_date_window(df, "rank_date").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_player_appearances() -> pd.DataFrame | None:
    path = os.path.join(_ROOT, "player_appearances.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "team", "player_id"])
    df["starter"]   = df["starter"].astype(bool)
    df["substitute"] = df["substitute"].astype(bool)
    return _filter_date_window(df).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_player_goals() -> pd.DataFrame | None:
    path = os.path.join(_ROOT, "player_goals.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "team", "player_id"])
    df["own_goal"] = df["own_goal"].astype(bool)
    df["penalty"]  = df["penalty"].astype(bool)
    return _filter_date_window(df).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_award_winners() -> pd.DataFrame | None:
    path = os.path.join(_ROOT, "award_winners.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "team", "player_id"])
    return _filter_date_window(df).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_substitutions() -> pd.DataFrame | None:
    path = os.path.join(_ROOT, "substitutions.csv")
    if not os.path.exists(path):
        return None  # optional — no warning, supersub falls back to minute-proxy
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "team", "player"])
    df["minute"] = pd.to_numeric(df["minute"], errors="coerce")
    return _filter_date_window(df).reset_index(drop=True)


def load_all() -> dict:
    """Load all CSVs."""
    copa_data = load_copa_america_data()
    euro_data = load_euro_2024_data()
    friendlies_data = load_friendlies_data()
    world_cup_2026_data = load_world_cup_2026_data()
    return {
        "matches":             load_matches(),
        "goalscorers":         load_goalscorers(),
        "shootouts":           load_shootouts(),
        "rankings":            load_rankings(),
        "substitutions":       load_substitutions(),
        "player_appearances":  load_player_appearances(),
        "player_goals":        load_player_goals(),
        "award_winners":       load_award_winners(),
        "copa_america":        copa_data,
        "euro_2024":           euro_data,
        "international_friendlies": friendlies_data,
        "world_cup_2026":      world_cup_2026_data,
    }


def get_all_teams(matches_df: pd.DataFrame) -> list[str]:
    """Return sorted list of all team names appearing in matches."""
    if matches_df is None:
        return []
    teams = set(matches_df["home_team"].dropna().tolist()) | set(matches_df["away_team"].dropna().tolist())
    return sorted(teams)


def count_loaded(data: dict) -> int:
    """Count how many of the four primary CSVs loaded successfully."""
    return sum(1 for k in ("matches", "goalscorers", "shootouts", "rankings") if data.get(k) is not None)
