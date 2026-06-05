"""
Data ingestion — loads and validates all four CSVs.
All functions decorated with @st.cache_data for Streamlit performance.
Missing files show st.warning() and return None so callers degrade gracefully.
"""

import os
import pandas as pd
import streamlit as st

from data.copa_america import append_copa_matches, load_copa_america_data

# CSVs are expected in the project root (one level up from data/)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REQUIRED_COLUMNS = {
    "matches": ["date", "home_team", "away_team", "home_score", "away_score", "tournament"],
    "goalscorers": ["date", "home_team", "away_team", "team", "scorer", "minute"],
    "shootouts": ["date", "home_team", "away_team", "winner"],
    "rankings": ["rank_date", "country_full", "rank", "total_points"],
}


def _check_columns(df: pd.DataFrame, required: list[str], filename: str) -> pd.DataFrame:
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.warning(f"⚠️  `{filename}` is missing columns: {missing}. Some features will be unavailable.")
    return df


@st.cache_data(show_spinner=False)
def load_matches() -> pd.DataFrame | None:
    path = os.path.join(_ROOT, "matches.csv")
    if not os.path.exists(path):
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
    return append_copa_matches(df.reset_index(drop=True))


@st.cache_data(show_spinner=False)
def load_goalscorers() -> pd.DataFrame | None:
    path = os.path.join(_ROOT, "goalscorers.csv")
    if not os.path.exists(path):
        st.warning("⚠️  `goalscorers.csv` not found. Super-sub and goalscoring features will be unavailable.")
        return None
    df = pd.read_csv(path)
    df = _check_columns(df, REQUIRED_COLUMNS["goalscorers"], "goalscorers.csv")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team"])
    df["minute"] = pd.to_numeric(df["minute"], errors="coerce")
    df["own_goal"] = df.get("own_goal", False)
    df["penalty"] = df.get("penalty", False)
    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_shootouts() -> pd.DataFrame | None:
    path = os.path.join(_ROOT, "shootouts.csv")
    if not os.path.exists(path):
        st.warning("⚠️  `shootouts.csv` not found. Penalty shootout features will be unavailable.")
        return None
    df = pd.read_csv(path)
    df = _check_columns(df, REQUIRED_COLUMNS["shootouts"], "shootouts.csv")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team", "winner"])
    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_rankings() -> pd.DataFrame | None:
    path = os.path.join(_ROOT, "rankings.csv")
    if not os.path.exists(path):
        st.warning("⚠️  `rankings.csv` not found. FIFA ranking features will be unavailable.")
        return None
    df = pd.read_csv(path)
    df = _check_columns(df, REQUIRED_COLUMNS["rankings"], "rankings.csv")
    df["rank_date"] = pd.to_datetime(df["rank_date"], errors="coerce")
    df = df.dropna(subset=["rank_date", "country_full", "rank"])
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df["total_points"] = pd.to_numeric(df["total_points"], errors="coerce")
    return df.reset_index(drop=True)


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
    return df.reset_index(drop=True)


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
    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_award_winners() -> pd.DataFrame | None:
    path = os.path.join(_ROOT, "award_winners.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "team", "player_id"])
    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_substitutions() -> pd.DataFrame | None:
    path = os.path.join(_ROOT, "substitutions.csv")
    if not os.path.exists(path):
        return None  # optional — no warning, supersub falls back to minute-proxy
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "team", "player"])
    df["minute"] = pd.to_numeric(df["minute"], errors="coerce")
    return df.reset_index(drop=True)


def load_all() -> dict:
    """Load all CSVs."""
    copa_data = load_copa_america_data()
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
