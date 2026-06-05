"""Optional Copa America 2024 data loaders and feature helpers."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import streamlit as st

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXTERNAL = os.path.join(_ROOT, "data", "external")

PLAYERS_PATH = os.path.join(_EXTERNAL, "international-copa-america-players-2024-to-2024-stats.csv")
TEAMS_PATH = os.path.join(_EXTERNAL, "international-copa-america-teams-2024-to-2024-stats.csv")
MATCHES_PATH = os.path.join(_EXTERNAL, "international-copa-america-matches-2024-to-2024-stats.csv")
LEAGUE_PATH = os.path.join(_EXTERNAL, "international-copa-america-league-2024-to-2024-stats.csv")

COPA_TOURNAMENT = "Copa America"
COPA_AGGREGATE_AVAILABLE_FROM = pd.Timestamp("2024-07-16")


def normalize_team_name(team: str) -> str:
    if pd.isna(team):
        return ""
    name = str(team).strip()
    replacements = {
        "USMNT": "United States",
        "USA": "United States",
        "United States Men's National Team": "United States",
    }
    if name in replacements:
        return replacements[name]
    return (
        name.replace(" National Team", "")
        .replace(" Men's National Team", "")
        .replace(" Womens National Team", "")
        .strip()
    )


def _read_csv(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_copa_america_data() -> dict:
    matches = _read_csv(MATCHES_PATH)
    teams = _read_csv(TEAMS_PATH)
    players = _read_csv(PLAYERS_PATH)
    league = _read_csv(LEAGUE_PATH)

    if matches is not None and not matches.empty:
        matches = matches.copy()
        matches["date"] = pd.to_datetime(
            matches["date_GMT"].astype(str).str.strip(),
            format="%b %d %Y - %I:%M%p",
            errors="coerce",
        )
        matches["home_team"] = matches["home_team_name"].map(normalize_team_name)
        matches["away_team"] = matches["away_team_name"].map(normalize_team_name)
        matches["home_score"] = pd.to_numeric(matches["home_team_goal_count"], errors="coerce").fillna(0).astype(int)
        matches["away_score"] = pd.to_numeric(matches["away_team_goal_count"], errors="coerce").fillna(0).astype(int)

    if teams is not None and not teams.empty:
        teams = teams.copy()
        source = "common_name" if "common_name" in teams.columns else "team_name"
        teams["team"] = teams[source].map(normalize_team_name)
        teams.loc[teams["country"].astype(str).str.upper() == "USA", "team"] = "United States"

    if players is not None and not players.empty:
        players = players.copy()
        players["team"] = players["Current Club"].map(normalize_team_name)
        missing_team = players["team"].eq("")
        players.loc[missing_team, "team"] = players.loc[missing_team, "nationality"].map(normalize_team_name)

    return {
        "matches": matches,
        "teams": teams,
        "players": players,
        "league": league,
    }


def append_copa_matches(matches_df: pd.DataFrame, copa_data: dict | None = None) -> pd.DataFrame:
    if matches_df is None:
        return matches_df
    copa_data = copa_data or load_copa_america_data()
    copa_matches = copa_data.get("matches")
    if copa_matches is None or copa_matches.empty:
        return matches_df

    extra = pd.DataFrame(
        {
            "date": copa_matches["date"],
            "home_team": copa_matches["home_team"],
            "away_team": copa_matches["away_team"],
            "home_score": copa_matches["home_score"],
            "away_score": copa_matches["away_score"],
            "tournament": COPA_TOURNAMENT,
            "city": copa_matches.get("stadium_name", ""),
            "country": "United States",
            "neutral": True,
        }
    )
    extra = extra.dropna(subset=["date", "home_team", "away_team"])

    combined = pd.concat([matches_df, extra], ignore_index=True, sort=False)
    combined["_key"] = (
        combined["date"].dt.strftime("%Y-%m-%d")
        + "|"
        + combined["home_team"].astype(str)
        + "|"
        + combined["away_team"].astype(str)
        + "|"
        + combined["tournament"].astype(str)
    )
    combined = combined.drop_duplicates("_key", keep="first").drop(columns=["_key"])
    return combined.sort_values("date").reset_index(drop=True)


def _num(row: pd.Series, col: str, default: float = 0.0) -> float:
    if row is None or col not in row:
        return default
    value = pd.to_numeric(row[col], errors="coerce")
    return default if pd.isna(value) else float(value)


def _team_row(team: str, teams: pd.DataFrame | None) -> pd.Series | None:
    if teams is None or teams.empty:
        return None
    rows = teams[teams["team"] == normalize_team_name(team)]
    if rows.empty:
        return None
    return rows.iloc[0]


def _player_summary(team: str, players: pd.DataFrame | None) -> dict:
    defaults = {
        "player_avg_rating": 0.0,
        "player_goal_rate": 0.0,
        "player_xg_per90": 0.0,
    }
    if players is None or players.empty:
        return defaults

    team_players = players[players["team"] == normalize_team_name(team)].copy()
    if team_players.empty:
        return defaults

    minutes = pd.to_numeric(team_players.get("minutes_played_overall"), errors="coerce").fillna(0)
    regulars = team_players[minutes >= 90]
    if regulars.empty:
        regulars = team_players

    def _mean(col: str, replace_zero: bool = False) -> float:
        values = pd.to_numeric(regulars.get(col), errors="coerce").replace([np.inf, -np.inf], np.nan)
        if replace_zero:
            values = values.replace(0, np.nan)
        mean = values.mean()
        return 0.0 if pd.isna(mean) else float(mean)

    defaults["player_avg_rating"] = _mean("average_rating_overall", replace_zero=True)
    defaults["player_goal_rate"] = _mean("goals_per_90_overall")
    defaults["player_xg_per90"] = _mean("xg_per_90_overall")
    return defaults


def _match_row(team_a: str, team_b: str, as_of_date, matches: pd.DataFrame | None) -> tuple[pd.Series | None, bool]:
    if matches is None or matches.empty:
        return None, False
    date = pd.Timestamp(as_of_date).normalize()
    a = normalize_team_name(team_a)
    b = normalize_team_name(team_b)
    same_date = matches["date"].dt.normalize() == date

    direct = matches[same_date & (matches["home_team"] == a) & (matches["away_team"] == b)]
    if not direct.empty:
        return direct.iloc[0], False

    swapped = matches[same_date & (matches["home_team"] == b) & (matches["away_team"] == a)]
    if not swapped.empty:
        return swapped.iloc[0], True
    return None, False


def build_copa_feature_values(team_a: str, team_b: str, as_of_date, copa_data: dict | None) -> dict:
    """Return Copa-derived features without using post-match match stats."""
    values = {}

    match, swapped = _match_row(team_a, team_b, as_of_date, copa_data.get("matches") if copa_data else None)
    if match is not None:
        if swapped:
            values.update(
                {
                    "team_a_copa_pre_match_ppg": _num(match, "Pre-Match PPG (Away)"),
                    "team_b_copa_pre_match_ppg": _num(match, "Pre-Match PPG (Home)"),
                    "team_a_copa_pre_match_xg": _num(match, "Away Team Pre-Match xG"),
                    "team_b_copa_pre_match_xg": _num(match, "Home Team Pre-Match xG"),
                    "team_a_copa_odds_win": _num(match, "odds_ft_away_team_win"),
                    "draw_copa_odds": _num(match, "odds_ft_draw"),
                    "team_b_copa_odds_win": _num(match, "odds_ft_home_team_win"),
                }
            )
        else:
            values.update(
                {
                    "team_a_copa_pre_match_ppg": _num(match, "Pre-Match PPG (Home)"),
                    "team_b_copa_pre_match_ppg": _num(match, "Pre-Match PPG (Away)"),
                    "team_a_copa_pre_match_xg": _num(match, "Home Team Pre-Match xG"),
                    "team_b_copa_pre_match_xg": _num(match, "Away Team Pre-Match xG"),
                    "team_a_copa_odds_win": _num(match, "odds_ft_home_team_win"),
                    "draw_copa_odds": _num(match, "odds_ft_draw"),
                    "team_b_copa_odds_win": _num(match, "odds_ft_away_team_win"),
                }
            )

    for odds_col, prob_col in [
        ("team_a_copa_odds_win", "team_a_copa_implied_win_prob"),
        ("draw_copa_odds", "draw_copa_implied_prob"),
        ("team_b_copa_odds_win", "team_b_copa_implied_win_prob"),
    ]:
        odds = values.get(odds_col, 0.0)
        values[prob_col] = 1.0 / odds if odds and odds > 0 else 0.0

    # Full tournament aggregates are valid only after the competition completed.
    if pd.Timestamp(as_of_date) >= COPA_AGGREGATE_AVAILABLE_FROM and copa_data:
        teams = copa_data.get("teams")
        players = copa_data.get("players")
        league = copa_data.get("league")
        for prefix, team in [("team_a", team_a), ("team_b", team_b)]:
            team_stats = _team_row(team, teams)
            player_stats = _player_summary(team, players)
            values.update(
                {
                    f"{prefix}_copa_team_ppg": _num(team_stats, "points_per_game"),
                    f"{prefix}_copa_team_xg_for": _num(team_stats, "xg_for_avg_overall"),
                    f"{prefix}_copa_team_xg_against": _num(team_stats, "xg_against_avg_overall"),
                    f"{prefix}_copa_team_goals_per_match": _num(team_stats, "goals_scored_per_match"),
                    f"{prefix}_copa_team_conceded_per_match": _num(team_stats, "goals_conceded_per_match"),
                    f"{prefix}_copa_player_avg_rating": player_stats["player_avg_rating"],
                    f"{prefix}_copa_player_goal_rate": player_stats["player_goal_rate"],
                    f"{prefix}_copa_player_xg_per90": player_stats["player_xg_per90"],
                }
            )
        if league is not None and not league.empty:
            row = league.iloc[0]
            values["copa_league_avg_goals"] = _num(row, "average_goals_per_match")
            values["copa_league_xg_avg"] = _num(row, "xg_avg_per_match")
            values["copa_league_home_advantage"] = _num(row, "home_advantage_percentage")

    return values
