"""
Super-sub detection engine.

Primary path (when substitutions_df is available):
  Uses real substitution records — exact who came on and when.
  sub_appearances = matches where player entered as a substitute
  goals_as_sub    = goals scored in those specific matches

Fallback path (no substitutions_df):
  Heuristic proxy — a goal scored after minute 60 with no earlier-minute
  goal in that same match is treated as a probable substitute appearance.
  This is less precise; prefer providing substitutions_df.

Threshold: 3+ substitute appearances, goals-as-sub rate > 0.40.
"""

import pandas as pd

SUPERSUB_MIN_APPEARANCES = 3
SUPERSUB_MIN_RATE = 0.40
SUB_ENTRY_MINUTE_PROXY = 60


def detect_supersub(
    team: str,
    goalscorers_df: pd.DataFrame | None,
    matches_df: pd.DataFrame | None,
    substitutions_df: pd.DataFrame | None = None,
) -> dict | None:
    """
    Returns the top super-sub candidate for `team`, or None if no one qualifies.

    Return shape:
        {
            "name": str,
            "goals_as_sub": int,
            "sub_appearances": int,
            "goal_rate": float,
            "team": str,
        }
    """
    if goalscorers_df is None:
        return None

    team_goals = goalscorers_df[
        (goalscorers_df["team"] == team) &
        (goalscorers_df["own_goal"] == False)  # noqa: E712
    ].copy()

    if team_goals.empty:
        return None

    if substitutions_df is not None:
        return _detect_with_real_subs(team, team_goals, substitutions_df)
    elif matches_df is not None:
        return _detect_with_proxy(team, team_goals)
    return None


def _detect_with_real_subs(
    team: str,
    team_goals: pd.DataFrame,
    substitutions_df: pd.DataFrame,
) -> dict | None:
    """Use real substitution records to identify sub appearances."""
    # substitutions_df already contains only coming_on==1 rows
    team_subs = substitutions_df[substitutions_df["team"] == team].copy()
    if team_subs.empty:
        return None

    # Ensure date columns are comparable
    team_subs = team_subs.copy()
    team_goals = team_goals.copy()
    if not pd.api.types.is_datetime64_any_dtype(team_subs["date"]):
        team_subs["date"] = pd.to_datetime(team_subs["date"], errors="coerce")
    if not pd.api.types.is_datetime64_any_dtype(team_goals["date"]):
        team_goals["date"] = pd.to_datetime(team_goals["date"], errors="coerce")

    # Build a set of (date, home_team, away_team) match keys for goals
    goals_keys = set(
        zip(
            team_goals["date"].dt.date,
            team_goals["home_team"],
            team_goals["away_team"],
        )
    )

    player_stats: dict[str, dict] = {}

    for player, subs_grp in team_subs.groupby("player"):
        sub_appearances = 0
        goals_as_sub = 0

        for _, sub_row in subs_grp.iterrows():
            sub_date = sub_row["date"].date() if pd.notna(sub_row["date"]) else None
            if sub_date is None:
                continue
            sub_appearances += 1

            # Count goals this player scored in this match
            match_goals = team_goals[
                (team_goals["date"].dt.date == sub_date) &
                (team_goals["home_team"] == sub_row["home_team"]) &
                (team_goals["away_team"] == sub_row["away_team"]) &
                (team_goals["scorer"] == player)
            ]
            goals_as_sub += len(match_goals)

        if sub_appearances >= SUPERSUB_MIN_APPEARANCES:
            rate = goals_as_sub / sub_appearances
            player_stats[player] = {
                "goals_as_sub": goals_as_sub,
                "sub_appearances": sub_appearances,
                "goal_rate": rate,
            }

    if not player_stats:
        return None

    qualified = {p: s for p, s in player_stats.items() if s["goal_rate"] > SUPERSUB_MIN_RATE}
    if not qualified:
        return None

    best_player = max(qualified, key=lambda p: qualified[p]["goal_rate"])
    stats = qualified[best_player]
    return {
        "name": best_player,
        "goals_as_sub": stats["goals_as_sub"],
        "sub_appearances": stats["sub_appearances"],
        "goal_rate": stats["goal_rate"],
        "team": team,
    }


def _detect_with_proxy(team: str, team_goals: pd.DataFrame) -> dict | None:
    """Minute-proxy fallback when real substitution records are unavailable."""
    player_stats: dict[str, dict] = {}

    for player, grp in team_goals.groupby("scorer"):
        match_keys = grp[["date", "home_team", "away_team"]].drop_duplicates()
        total_sub_appearances = 0
        total_sub_goals = 0

        for _, row in match_keys.iterrows():
            match_goals = grp[
                (grp["date"] == row["date"]) &
                (grp["home_team"] == row["home_team"]) &
                (grp["away_team"] == row["away_team"])
            ]
            if match_goals["minute"].notna().any():
                late_goals = match_goals[match_goals["minute"] > SUB_ENTRY_MINUTE_PROXY]
                early_goals = match_goals[match_goals["minute"] <= SUB_ENTRY_MINUTE_PROXY]
                if not late_goals.empty and early_goals.empty:
                    total_sub_appearances += 1
                    total_sub_goals += len(late_goals)

        if total_sub_appearances >= SUPERSUB_MIN_APPEARANCES:
            rate = total_sub_goals / total_sub_appearances
            player_stats[player] = {
                "goals_as_sub": total_sub_goals,
                "sub_appearances": total_sub_appearances,
                "goal_rate": rate,
            }

    if not player_stats:
        return None

    qualified = {p: s for p, s in player_stats.items() if s["goal_rate"] > SUPERSUB_MIN_RATE}
    if not qualified:
        return None

    best_player = max(qualified, key=lambda p: qualified[p]["goal_rate"])
    stats = qualified[best_player]
    return {
        "name": best_player,
        "goals_as_sub": stats["goals_as_sub"],
        "sub_appearances": stats["sub_appearances"],
        "goal_rate": stats["goal_rate"],
        "team": team,
    }
