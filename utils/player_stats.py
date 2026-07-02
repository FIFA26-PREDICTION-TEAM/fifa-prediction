"""
Player impact engine.

Two public functions:
  compute_player_features(team, as_of_date, ...)  -> dict of 6 model features
  get_key_players(team, as_of_date, ...)          -> list of player dicts for UI display

All data is filtered to dates strictly before as_of_date — no leakage.
"""

import pandas as pd
import numpy as np

from model.features import DATA_FROM_YEAR

# Attacking positions — used to compute attack_rating
ATTACK_POSITIONS = {"FW", "CF", "LW", "RW", "LF", "RF", "SS", "AM"}

_FROM = pd.Timestamp(f"{DATA_FROM_YEAR}-01-01")
_PLAYER_FEATURE_CACHE: dict[tuple[str, int, int, int, int], dict] = {}


def _filter(df: pd.DataFrame, as_of_date, date_col: str = "date", team: str | None = None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    col = df[date_col]
    if not pd.api.types.is_datetime64_any_dtype(col):
        col = pd.to_datetime(col, errors="coerce")
    cutoff = pd.Timestamp(as_of_date)
    mask = (col < cutoff) & (col >= _FROM)
    if team is not None and "team" in df.columns:
        mask = mask & (df["team"] == team)
    return df[mask].copy()


def compute_player_features(
    team: str,
    as_of_date,
    player_appearances_df: pd.DataFrame | None,
    player_goals_df: pd.DataFrame | None,
    award_winners_df: pd.DataFrame | None,
) -> dict:
    """
    Returns 6 player-based features for team as of as_of_date.

    Features:
        top_scorer_wc_goals     — WC career goals of team's all-time top scorer
        attack_rating           — avg goals/appearance for attacking-position players
        golden_ball_count       — Golden Ball (best player) awards won by team players
        golden_boot_count       — Golden Boot (top scorer) awards
        golden_glove_count      — Golden Glove (best keeper) awards
        avg_player_experience   — avg number of WC tournaments per squad member
    """
    defaults = {
        "top_scorer_wc_goals": 0,
        "attack_rating": 0.0,
        "golden_ball_count": 0,
        "golden_boot_count": 0,
        "golden_glove_count": 0,
        "avg_player_experience": 0.0,
    }
    cache_key = (
        team,
        pd.Timestamp(as_of_date).value,
        id(player_appearances_df),
        id(player_goals_df),
        id(award_winners_df),
    )
    cached = _PLAYER_FEATURE_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)

    pa = _filter(player_appearances_df, as_of_date, team=team) if player_appearances_df is not None else pd.DataFrame()
    pg = _filter(player_goals_df, as_of_date, team=team) if player_goals_df is not None else pd.DataFrame()
    aw = _filter(award_winners_df, as_of_date, team=team) if award_winners_df is not None else pd.DataFrame()

    # ── Team slices ───────────────────────────────────────────────────────────
    team_pa = pa
    team_pg = pg[~pg["own_goal"]] if not pg.empty else pd.DataFrame()
    team_aw = aw

    if team_pa.empty and team_pg.empty:
        return defaults

    # ── top_scorer_wc_goals ───────────────────────────────────────────────────
    if not team_pg.empty:
        goals_by_player = team_pg.groupby("player_id").size()
        defaults["top_scorer_wc_goals"] = int(goals_by_player.max())

    # ── attack_rating ─────────────────────────────────────────────────────────
    if not team_pa.empty and not team_pg.empty:
        atk_pa = team_pa[team_pa["position"].isin(ATTACK_POSITIONS)]
        if not atk_pa.empty:
            apps_by_player = atk_pa.groupby("player_id")["match_id"].count()
            goals_by_player = team_pg.groupby("player_id").size()
            # Only players with at least 2 appearances to reduce noise
            eligible = apps_by_player[apps_by_player >= 2].index
            rates = []
            for pid in eligible:
                goals = goals_by_player.get(pid, 0)
                apps = apps_by_player[pid]
                rates.append(goals / apps)
            defaults["attack_rating"] = float(np.mean(rates)) if rates else 0.0

    # ── Award counts ──────────────────────────────────────────────────────────
    if not team_aw.empty:
        award_counts = team_aw["award_name"].value_counts()
        defaults["golden_ball_count"]  = int(award_counts.get("Golden Ball", 0))
        defaults["golden_boot_count"]  = int(award_counts.get("Golden Boot", 0))
        defaults["golden_glove_count"] = int(award_counts.get("Golden Glove", 0))

    # ── avg_player_experience (WC tournaments per player) ─────────────────────
    if not team_pa.empty:
        exp = team_pa.groupby("player_id")["tournament_id"].nunique()
        defaults["avg_player_experience"] = float(exp.mean())

    if len(_PLAYER_FEATURE_CACHE) > 20000:
        _PLAYER_FEATURE_CACHE.clear()
    _PLAYER_FEATURE_CACHE[cache_key] = dict(defaults)
    return defaults


def get_key_players(
    team: str,
    as_of_date,
    player_appearances_df: pd.DataFrame | None,
    player_goals_df: pd.DataFrame | None,
    award_winners_df: pd.DataFrame | None,
    n: int = 5,
) -> list[dict]:
    """
    Returns top-n key players for team for UI display.

    Each dict:
        name, goals, appearances, goal_rate, position, awards (list of award names)
    """
    pa = _filter(player_appearances_df, as_of_date, team=team) if player_appearances_df is not None else pd.DataFrame()
    pg = _filter(player_goals_df, as_of_date, team=team) if player_goals_df is not None else pd.DataFrame()
    aw = _filter(award_winners_df, as_of_date, team=team) if award_winners_df is not None else pd.DataFrame()

    team_pa = pa
    team_pg = pg[~pg["own_goal"]] if not pg.empty else pd.DataFrame()
    team_aw = aw

    if team_pa.empty and team_pg.empty:
        return []

    # Appearances per player
    apps = team_pa.groupby(["player_id", "player_name"])["match_id"].count().reset_index()
    apps.columns = ["player_id", "player_name", "appearances"]

    # Goals per player
    if not team_pg.empty:
        goals = team_pg.groupby("player_id").size().rename("goals").reset_index()
    else:
        goals = pd.DataFrame(columns=["player_id", "goals"])

    merged = apps.merge(goals, on="player_id", how="left").fillna({"goals": 0})
    merged["goals"] = merged["goals"].astype(int)
    merged["goal_rate"] = merged["goals"] / merged["appearances"]

    # Most common position per player
    if not team_pa.empty:
        pos = team_pa.groupby("player_id")["position"].agg(lambda x: x.mode().iloc[0] if len(x) > 0 else "")
        merged = merged.merge(pos.reset_index(), on="player_id", how="left")
    else:
        merged["position"] = ""

    # Awards per player
    award_map: dict[str, list[str]] = {}
    if not team_aw.empty:
        for _, row in team_aw.iterrows():
            pid = row["player_id"]
            award_map.setdefault(pid, [])
            if row["award_name"] not in award_map[pid]:
                award_map[pid].append(row["award_name"])

    # Rank by goals then appearances
    merged = merged.sort_values(["goals", "appearances"], ascending=False)

    result = []
    for _, row in merged.head(n).iterrows():
        result.append({
            "name":        row["player_name"],
            "goals":       int(row["goals"]),
            "appearances": int(row["appearances"]),
            "goal_rate":   round(float(row["goal_rate"]), 3),
            "position":    row.get("position", ""),
            "awards":      award_map.get(row["player_id"], []),
        })
    return result
