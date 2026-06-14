"""Optional 2026 international friendlies data loaders and feature helpers."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import streamlit as st

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXTERNAL = os.path.join(_ROOT, "data", "external")


def _default_path(filename: str) -> str:
    return os.path.join(os.path.expanduser("~"), "Downloads", filename)


PLAYERS_PATH = os.getenv(
    "ML_PRJCT_FRIENDLIES_PLAYERS_PATH",
    os.path.join(_EXTERNAL, "international-international-friendlies-players-2026-to-2026-stats.csv"),
)
TEAMS_PATH = os.getenv(
    "ML_PRJCT_FRIENDLIES_TEAMS_PATH",
    os.path.join(_EXTERNAL, "international-international-friendlies-teams-2026-to-2026-stats.csv"),
)
MATCHES_PATH = os.getenv(
    "ML_PRJCT_FRIENDLIES_MATCHES_PATH",
    os.path.join(_EXTERNAL, "international-international-friendlies-matches-2026-to-2026-stats.csv"),
)
FIFA_PRE_WC_MATCHES_PATH = os.getenv(
    "ML_PRJCT_FIFA_PRE_WC_FRIENDLIES_PATH",
    os.path.join(_EXTERNAL, "fifa_pre_world_cup_friendlies_2026.csv"),
)
LEAGUE_PATH = os.getenv(
    "ML_PRJCT_FRIENDLIES_LEAGUE_PATH",
    os.path.join(_EXTERNAL, "international-international-friendlies-league-2026-to-2026-stats.csv"),
)

if not os.path.exists(PLAYERS_PATH):
    PLAYERS_PATH = _default_path("international-international-friendlies-players-2026-to-2026-stats.csv")
if not os.path.exists(TEAMS_PATH):
    TEAMS_PATH = _default_path("international-international-friendlies-teams-2026-to-2026-stats.csv")
if not os.path.exists(MATCHES_PATH):
    MATCHES_PATH = _default_path("international-international-friendlies-matches-2026-to-2026-stats.csv")
if not os.path.exists(FIFA_PRE_WC_MATCHES_PATH):
    FIFA_PRE_WC_MATCHES_PATH = _default_path("fifa_pre_world_cup_friendlies_2026.csv")
if not os.path.exists(LEAGUE_PATH):
    LEAGUE_PATH = _default_path("international-international-friendlies-league-2026-to-2026-stats.csv")

FRIENDLIES_TOURNAMENT = "International Friendly"


def normalize_team_name(team: str) -> str:
    if pd.isna(team):
        return ""
    name = str(team).strip()
    replacements = {
        "USMNT": "United States",
        "USA": "United States",
        "United States Men's National Team": "United States",
        "Congo DR": "DR Congo",
        "Ivory Coast": "Cote d'Ivoire",
        "Cape Verde Islands": "Cape Verde",
        "Korea Republic": "South Korea",
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
    return pd.read_csv(path, encoding="utf-8-sig")


def _prepare_stats_matches(matches: pd.DataFrame | None) -> pd.DataFrame | None:
    if matches is None or matches.empty:
        return None

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
    matches["status"] = matches.get("status", "").astype(str).str.lower().str.strip()
    matches = matches[matches["status"] == "complete"].reset_index(drop=True)
    return matches


def _prepare_fifa_pre_wc_matches(matches: pd.DataFrame | None) -> pd.DataFrame | None:
    if matches is None or matches.empty:
        return None

    required = {"date", "team1", "team2", "team1_score", "team2_score"}
    if not required.issubset(matches.columns):
        return None

    prepared = pd.DataFrame(
        {
            "date": pd.to_datetime(matches["date"], format="%d-%m-%Y", errors="coerce"),
            "home_team": matches["team1"].map(normalize_team_name),
            "away_team": matches["team2"].map(normalize_team_name),
            "home_score": pd.to_numeric(matches["team1_score"], errors="coerce").fillna(0).astype(int),
            "away_score": pd.to_numeric(matches["team2_score"], errors="coerce").fillna(0).astype(int),
            "status": "complete",
            "stadium_name": "",
            "source": matches.get("source", "FIFA pre-World Cup friendlies 2026"),
            "match_id": matches.get("match_id", ""),
        }
    )
    return prepared.dropna(subset=["date", "home_team", "away_team"]).reset_index(drop=True)


def _dedupe_matches(matches: pd.DataFrame | None) -> pd.DataFrame | None:
    if matches is None or matches.empty:
        return matches

    matches = matches.copy()
    matches["_key"] = (
        matches["date"].dt.strftime("%Y-%m-%d")
        + "|"
        + matches["home_team"].astype(str)
        + "|"
        + matches["away_team"].astype(str)
    )
    return (
        matches.sort_values("date")
        .drop_duplicates("_key", keep="first")
        .drop(columns=["_key"])
        .reset_index(drop=True)
    )


def _derive_team_stats_from_matches(matches: pd.DataFrame | None) -> pd.DataFrame | None:
    if matches is None or matches.empty:
        return None

    rows = []
    for _, match in matches.iterrows():
        home_score = int(match["home_score"])
        away_score = int(match["away_score"])
        rows.extend(
            [
                {
                    "team": match["home_team"],
                    "goals_for": home_score,
                    "goals_against": away_score,
                    "points": 3 if home_score > away_score else 1 if home_score == away_score else 0,
                },
                {
                    "team": match["away_team"],
                    "goals_for": away_score,
                    "goals_against": home_score,
                    "points": 3 if away_score > home_score else 1 if away_score == home_score else 0,
                },
            ]
        )

    team_matches = pd.DataFrame(rows).dropna(subset=["team"])
    if team_matches.empty:
        return None

    grouped = team_matches.groupby("team", as_index=False).agg(
        matches_played=("team", "size"),
        points=("points", "sum"),
        goals_for=("goals_for", "sum"),
        goals_against=("goals_against", "sum"),
    )
    grouped["points_per_game"] = grouped["points"] / grouped["matches_played"].replace(0, np.nan)
    grouped["goals_scored_per_match"] = grouped["goals_for"] / grouped["matches_played"].replace(0, np.nan)
    grouped["goals_conceded_per_match"] = grouped["goals_against"] / grouped["matches_played"].replace(0, np.nan)
    grouped["xg_for_avg_overall"] = 0.0
    grouped["xg_against_avg_overall"] = 0.0
    grouped["common_name"] = grouped["team"]
    grouped["country"] = ""
    return grouped.fillna(0.0)


def _derive_league_stats_from_matches(matches: pd.DataFrame | None) -> pd.DataFrame | None:
    if matches is None or matches.empty:
        return None

    total_goals = pd.to_numeric(matches["home_score"], errors="coerce").fillna(0) + pd.to_numeric(
        matches["away_score"], errors="coerce"
    ).fillna(0)
    home_wins = (matches["home_score"] > matches["away_score"]).mean()
    return pd.DataFrame(
        [
            {
                "average_goals_per_match": float(total_goals.mean()) if len(total_goals) else 0.0,
                "xg_avg_per_match": 0.0,
                "home_advantage_percentage": float(home_wins * 100) if pd.notna(home_wins) else 0.0,
            }
        ]
    )


def _merge_team_stats(source_teams: pd.DataFrame | None, derived_teams: pd.DataFrame | None) -> pd.DataFrame | None:
    if source_teams is None or source_teams.empty:
        return derived_teams
    if derived_teams is None or derived_teams.empty:
        return source_teams

    known = set(source_teams["team"].dropna().astype(str))
    missing_derived = derived_teams[~derived_teams["team"].astype(str).isin(known)]
    if missing_derived.empty:
        return source_teams
    return pd.concat([source_teams, missing_derived], ignore_index=True, sort=False)


@st.cache_data(show_spinner=False)
def load_friendlies_data() -> dict:
    stats_matches = _prepare_stats_matches(_read_csv(MATCHES_PATH))
    fifa_pre_wc_matches = _prepare_fifa_pre_wc_matches(_read_csv(FIFA_PRE_WC_MATCHES_PATH))
    teams = _read_csv(TEAMS_PATH)
    players = _read_csv(PLAYERS_PATH)
    league = _read_csv(LEAGUE_PATH)

    match_frames = [frame for frame in (stats_matches, fifa_pre_wc_matches) if frame is not None and not frame.empty]
    matches = _dedupe_matches(pd.concat(match_frames, ignore_index=True, sort=False)) if match_frames else None

    if teams is not None and not teams.empty:
        teams = teams.copy()
        source = "common_name" if "common_name" in teams.columns else "team_name"
        teams["team"] = teams[source].map(normalize_team_name)
        teams.loc[teams["country"].astype(str).str.upper() == "USA", "team"] = "United States"
    teams = _merge_team_stats(teams, _derive_team_stats_from_matches(matches))

    if players is not None and not players.empty:
        players = players.copy()
        players["team"] = players["nationality"].map(normalize_team_name)

    if (league is None or league.empty) and matches is not None and not matches.empty:
        league = _derive_league_stats_from_matches(matches)

    return {
        "matches": matches,
        "teams": teams,
        "players": players,
        "league": league,
    }


def append_friendlies_matches(matches_df: pd.DataFrame, friendlies_data: dict | None = None) -> pd.DataFrame:
    if matches_df is None:
        return matches_df
    friendlies_data = friendlies_data or load_friendlies_data()
    friendly_matches = friendlies_data.get("matches")
    if friendly_matches is None or friendly_matches.empty:
        return matches_df

    extra = pd.DataFrame(
        {
            "date": friendly_matches["date"],
            "home_team": friendly_matches["home_team"],
            "away_team": friendly_matches["away_team"],
            "home_score": friendly_matches["home_score"],
            "away_score": friendly_matches["away_score"],
            "tournament": FRIENDLIES_TOURNAMENT,
            "city": friendly_matches.get("stadium_name", ""),
            "country": "",
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


def _aggregate_available_from(friendlies_data: dict | None) -> pd.Timestamp | None:
    if not friendlies_data:
        return None
    matches = friendlies_data.get("matches")
    if matches is None or matches.empty:
        return None
    latest = pd.to_datetime(matches["date"], errors="coerce").max()
    if pd.isna(latest):
        return None
    return pd.Timestamp(latest).normalize() + pd.Timedelta(days=1)


def build_friendlies_feature_values(team_a: str, team_b: str, as_of_date, friendlies_data: dict | None) -> dict:
    """Return curated friendlies features with leakage guards."""
    values = {}

    match, swapped = _match_row(team_a, team_b, as_of_date, friendlies_data.get("matches") if friendlies_data else None)
    if match is not None:
        if swapped:
            values.update(
                {
                    "team_a_friendlies_pre_match_ppg": _num(match, "Pre-Match PPG (Away)"),
                    "team_b_friendlies_pre_match_ppg": _num(match, "Pre-Match PPG (Home)"),
                    "team_a_friendlies_pre_match_xg": _num(match, "Away Team Pre-Match xG"),
                    "team_b_friendlies_pre_match_xg": _num(match, "Home Team Pre-Match xG"),
                    "team_a_friendlies_odds_win": _num(match, "odds_ft_away_team_win"),
                    "draw_friendlies_odds": _num(match, "odds_ft_draw"),
                    "team_b_friendlies_odds_win": _num(match, "odds_ft_home_team_win"),
                }
            )
        else:
            values.update(
                {
                    "team_a_friendlies_pre_match_ppg": _num(match, "Pre-Match PPG (Home)"),
                    "team_b_friendlies_pre_match_ppg": _num(match, "Pre-Match PPG (Away)"),
                    "team_a_friendlies_pre_match_xg": _num(match, "Home Team Pre-Match xG"),
                    "team_b_friendlies_pre_match_xg": _num(match, "Away Team Pre-Match xG"),
                    "team_a_friendlies_odds_win": _num(match, "odds_ft_home_team_win"),
                    "draw_friendlies_odds": _num(match, "odds_ft_draw"),
                    "team_b_friendlies_odds_win": _num(match, "odds_ft_away_team_win"),
                }
            )

    for odds_col, prob_col in [
        ("team_a_friendlies_odds_win", "team_a_friendlies_implied_win_prob"),
        ("draw_friendlies_odds", "draw_friendlies_implied_prob"),
        ("team_b_friendlies_odds_win", "team_b_friendlies_implied_win_prob"),
    ]:
        odds = values.get(odds_col, 0.0)
        values[prob_col] = 1.0 / odds if odds and odds > 0 else 0.0

    aggregate_available_from = _aggregate_available_from(friendlies_data)
    if aggregate_available_from is not None and pd.Timestamp(as_of_date) >= aggregate_available_from and friendlies_data:
        teams = friendlies_data.get("teams")
        players = friendlies_data.get("players")
        league = friendlies_data.get("league")
        for prefix, team in [("team_a", team_a), ("team_b", team_b)]:
            team_stats = _team_row(team, teams)
            player_stats = _player_summary(team, players)
            values.update(
                {
                    f"{prefix}_friendlies_team_ppg": _num(team_stats, "points_per_game"),
                    f"{prefix}_friendlies_team_xg_for": _num(team_stats, "xg_for_avg_overall"),
                    f"{prefix}_friendlies_team_xg_against": _num(team_stats, "xg_against_avg_overall"),
                    f"{prefix}_friendlies_team_goals_per_match": _num(team_stats, "goals_scored_per_match"),
                    f"{prefix}_friendlies_team_conceded_per_match": _num(team_stats, "goals_conceded_per_match"),
                    f"{prefix}_friendlies_player_avg_rating": player_stats["player_avg_rating"],
                    f"{prefix}_friendlies_player_goal_rate": player_stats["player_goal_rate"],
                    f"{prefix}_friendlies_player_xg_per90": player_stats["player_xg_per90"],
                }
            )
        if league is not None and not league.empty:
            row = league.iloc[0]
            values["friendlies_league_avg_goals"] = _num(row, "average_goals_per_match")
            values["friendlies_league_xg_avg"] = _num(row, "xg_avg_per_match")
            values["friendlies_league_home_advantage"] = _num(row, "home_advantage_percentage")

    return values
