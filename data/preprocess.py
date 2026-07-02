"""
Feature engineering — the keystone module.

build_feature_row(team_a, team_b, as_of_date, ...) is the single function
called by both train.py (per historical match) and predict.py (for inference).
All data is filtered to `date < as_of_date` inside this function — that
one line is the no-leakage guarantee for the rolling training window.
"""

import pandas as pd
import numpy as np

from model.features import (
    FEATURE_COLUMNS,
    DATA_FROM_YEAR,
    get_tournament_weight,
    recency_weight,
    WC_TITLES,
)
from utils.supersub import detect_supersub
from utils.player_stats import compute_player_features
from data.copa_america import build_copa_feature_values
from data.euro_2024 import build_euro_feature_values
from data.international_friendlies import build_friendlies_feature_values

# Tournament strings that count as competitive (not friendly)
_COMPETITIVE_KEYWORDS = [
    "world cup", "copa", "euro", "afcon", "asian cup",
    "qualifier", "qualification", "gold cup",
    "nations league", "confederation", "olympic",
]


def _is_competitive(tournament: str) -> bool:
    t = tournament.lower()
    return any(k in t for k in _COMPETITIVE_KEYWORDS)


def _is_wc(tournament: str) -> bool:
    t = tournament.lower()
    return "world cup" in t and "qualifier" not in t and "qualification" not in t


_COMPETITIVE_RE = "|".join(_COMPETITIVE_KEYWORDS)
_WINDOW_CACHE: dict[tuple[str, int, int], pd.DataFrame] = {}
_GOALSCORING_CACHE: dict[tuple[str, int, int], dict] = {}
_WC_MASK_CACHE: dict[int, pd.Series] = {}
_FIRST_GOAL_CACHE: dict[int, pd.DataFrame] = {}
_RANKING_GROUP_CACHE: dict[int, dict[str, pd.DataFrame]] = {}
_RANKING_FEATURE_CACHE: dict[tuple[str, int, int], dict] = {}


def _window_before(name: str, df: pd.DataFrame | None, cutoff: pd.Timestamp) -> pd.DataFrame | None:
    if df is None:
        return None
    key = (name, id(df), cutoff.value)
    cached = _WINDOW_CACHE.get(key)
    if cached is not None:
        return cached
    if len(_WINDOW_CACHE) > 24:
        _WINDOW_CACHE.clear()
        _WC_MASK_CACHE.clear()
        _FIRST_GOAL_CACHE.clear()
    out = df[df["date"] < cutoff]
    _WINDOW_CACHE[key] = out
    return out


def _wc_mask_for(matches: pd.DataFrame) -> pd.Series:
    key = id(matches)
    cached = _WC_MASK_CACHE.get(key)
    if cached is not None:
        return cached
    if "_is_wc" in matches.columns:
        mask = matches["_is_wc"].fillna(False).astype(bool)
    else:
        mask = matches["tournament"].apply(_is_wc)
    if len(_WC_MASK_CACHE) > 10000:
        _WC_MASK_CACHE.clear()
    _WC_MASK_CACHE[key] = mask
    return mask


def _first_goal_lookup(goalscorers: pd.DataFrame) -> pd.DataFrame:
    key = id(goalscorers)
    cached = _FIRST_GOAL_CACHE.get(key)
    if cached is not None:
        return cached
    first_goals = (
        goalscorers[goalscorers["own_goal"] == False]  # noqa: E712
        .sort_values(["date", "home_team", "away_team", "minute"])
        .drop_duplicates(["date", "home_team", "away_team"], keep="first")
        [["date", "home_team", "away_team", "team"]]
        .rename(columns={"team": "first_scorer_team"})
    )
    if len(_FIRST_GOAL_CACHE) > 10000:
        _FIRST_GOAL_CACHE.clear()
    _FIRST_GOAL_CACHE[key] = first_goals
    return first_goals


def _team_form(team: str, matches: pd.DataFrame, n: int = 10) -> dict:
    """Recent form stats for a team (last n *competitive* matches only)."""
    _zero = {
        "form_wins": 0, "form_draws": 0, "form_losses": 0,
        "goals_scored_avg": 0.0, "goals_conceded_avg": 0.0,
        "clean_sheet_rate": 0.0, "win_streak": 0, "unbeaten_streak": 0,
    }
    if matches.empty or "tournament" not in matches.columns:
        return _zero
    if "_is_competitive" in matches.columns:
        mask = matches["_is_competitive"].fillna(False).astype(bool)
    else:
        # str.contains avoids the zero-column apply() bug on empty DataFrames
        mask = matches["tournament"].str.lower().str.contains(_COMPETITIVE_RE, na=False)
    competitive = matches[mask]
    if competitive.empty:
        return _zero
    tm = competitive[
        (competitive["home_team"] == team) | (competitive["away_team"] == team)
    ].sort_values("date").tail(n)

    wins = draws = losses = 0
    goals_scored = goals_conceded = clean_sheets = 0
    win_streak = unbeaten_streak = 0
    streak_broken = False
    ub_broken = False

    results_rev = []
    for _, row in tm.iterrows():
        if row["home_team"] == team:
            gf, ga = row["home_score"], row["away_score"]
        else:
            gf, ga = row["away_score"], row["home_score"]

        goals_scored += gf
        goals_conceded += ga
        if ga == 0:
            clean_sheets += 1

        if gf > ga:
            wins += 1
            results_rev.append("W")
        elif gf == ga:
            draws += 1
            results_rev.append("D")
        else:
            losses += 1
            results_rev.append("L")

    # Compute streaks from most recent backwards
    for r in reversed(results_rev):
        if not streak_broken:
            if r == "W":
                win_streak += 1
            else:
                streak_broken = True
        if not ub_broken:
            if r in ("W", "D"):
                unbeaten_streak += 1
            else:
                ub_broken = True

    n_matches = len(tm)
    return {
        "form_wins": wins,
        "form_draws": draws,
        "form_losses": losses,
        "goals_scored_avg": goals_scored / n_matches if n_matches > 0 else 0.0,
        "goals_conceded_avg": goals_conceded / n_matches if n_matches > 0 else 0.0,
        "clean_sheet_rate": clean_sheets / n_matches if n_matches > 0 else 0.0,
        "win_streak": win_streak,
        "unbeaten_streak": unbeaten_streak,
    }


def _h2h_features(team_a: str, team_b: str, matches: pd.DataFrame, ref_date) -> dict:
    """Head-to-head features between team_a and team_b."""
    h2h = matches[
        ((matches["home_team"] == team_a) & (matches["away_team"] == team_b)) |
        ((matches["home_team"] == team_b) & (matches["away_team"] == team_a))
    ].sort_values("date")

    total = len(h2h)
    wins_a = draws = wins_b = 0
    wc_wins_a = wc_wins_b = 0
    weighted_wins_a = weighted_total = 0.0

    for _, row in h2h.iterrows():
        tw = get_tournament_weight(row["tournament"])
        rw = recency_weight(row["date"], ref_date)
        w = tw * rw

        if row["home_team"] == team_a:
            gf, ga = row["home_score"], row["away_score"]
        else:
            gf, ga = row["away_score"], row["home_score"]

        weighted_total += w
        if gf > ga:
            wins_a += 1
            weighted_wins_a += w
            if bool(row.get("_is_wc", _is_wc(row["tournament"]))):
                wc_wins_a += 1
        elif gf < ga:
            wins_b += 1
            if bool(row.get("_is_wc", _is_wc(row["tournament"]))):
                wc_wins_b += 1
        else:
            draws += 1

    # Last 5 meetings
    last5 = h2h.tail(5)
    last5_wins_a = last5_goals_a = last5_goals_b = 0
    for _, row in last5.iterrows():
        if row["home_team"] == team_a:
            gf, ga = row["home_score"], row["away_score"]
        else:
            gf, ga = row["away_score"], row["home_score"]
        last5_goals_a += gf
        last5_goals_b += ga
        if gf > ga:
            last5_wins_a += 1

    weighted_win_ratio = weighted_wins_a / weighted_total if weighted_total > 0 else 0.5

    return {
        "h2h_total_meetings": total,
        "h2h_team_a_wins": wins_a,
        "h2h_team_b_wins": wins_b,
        "h2h_draws": draws,
        "h2h_wc_only_team_a_wins": wc_wins_a,
        "h2h_wc_only_team_b_wins": wc_wins_b,
        "h2h_last5_team_a_wins": last5_wins_a,
        "h2h_last5_goals_a": last5_goals_a,
        "h2h_last5_goals_b": last5_goals_b,
        "h2h_weighted_win_ratio_a": weighted_win_ratio,
    }


def _tournament_features(team: str, matches: pd.DataFrame, shootouts: pd.DataFrame | None) -> dict:
    """WC tournament stats for a team.

    stage_name/knockout_stage/extra_time come straight from matches.csv. Older
    appended sources (Copa, friendlies, live 2026 feed rows) don't carry these
    columns, so their rows show up as NaN and are simply excluded from the
    stage-aware calculations below; wc_finals_reached and wc_win_rate_knockouts
    fall back to the WC_TITLES-based proxy only when no stage data exists at all.
    """
    wc = matches[
        ((matches["home_team"] == team) | (matches["away_team"] == team)) &
        (
            matches["_is_wc"].fillna(False).astype(bool)
            if "_is_wc" in matches.columns else matches["tournament"].apply(_is_wc)
        )
    ]

    titles = WC_TITLES.get(team, 0)

    has_stage_col = "stage_name" in wc.columns and wc["stage_name"].notna().any()
    if has_stage_col:
        finals_reached = int((wc["stage_name"].fillna("") == "final").sum())
    else:
        finals_reached = titles  # lower bound; exact count not derivable without stage data

    has_knockout_col = "knockout_stage" in wc.columns and wc["knockout_stage"].notna().any()
    if has_knockout_col:
        ko_matches = wc[pd.to_numeric(wc["knockout_stage"], errors="coerce").fillna(0) == 1]
    else:
        ko_matches = wc  # proxy: overall WC record when no stage data exists

    ko_wins = ko_total = et_matches = 0
    for _, row in ko_matches.iterrows():
        if row["home_team"] == team:
            gf, ga = row["home_score"], row["away_score"]
        else:
            gf, ga = row["away_score"], row["home_score"]
        ko_total += 1
        if gf > ga:
            ko_wins += 1
        if has_knockout_col and float(row.get("extra_time", 0) or 0) == 1:
            et_matches += 1

    ko_win_rate = ko_wins / ko_total if ko_total > 0 else 0.0
    et_rate = et_matches / ko_total if (has_knockout_col and ko_total > 0) else 0.0

    # Penalty shootout record (from shootouts.csv)
    so_wins = so_losses = 0
    if shootouts is not None:
        team_so = shootouts[
            (shootouts["home_team"] == team) | (shootouts["away_team"] == team)
        ]
        for _, row in team_so.iterrows():
            if row["winner"] == team:
                so_wins += 1
            else:
                so_losses += 1

    return {
        "wc_titles": titles,
        "wc_finals_reached": finals_reached,
        "wc_win_rate_knockouts": ko_win_rate,
        "wc_extra_time_rate": et_rate,
        "penalty_shootout_wins": so_wins,
        "penalty_shootout_losses": so_losses,
    }


def _ranking_features(team: str, rankings: pd.DataFrame | None, ref_date) -> dict:
    """Most recent FIFA ranking for team as of ref_date."""
    if rankings is None:
        return {"rank": 0, "rank_points": 0.0}
    cutoff = pd.Timestamp(ref_date)
    cache_key = (team, cutoff.value, id(rankings))
    cached = _RANKING_FEATURE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    grouped = _RANKING_GROUP_CACHE.get(id(rankings))
    if grouped is None:
        grouped = {
            country: frame.sort_values("rank_date").reset_index(drop=True)
            for country, frame in rankings.groupby("country_full", dropna=False)
        }
        _RANKING_GROUP_CACHE[id(rankings)] = grouped

    team_rankings = grouped.get(team)
    if team_rankings is None or team_rankings.empty:
        return {"rank": 0, "rank_points": 0.0}
    pos = team_rankings["rank_date"].searchsorted(cutoff, side="right") - 1
    if pos < 0:
        return {"rank": 0, "rank_points": 0.0}
    latest = team_rankings.iloc[int(pos)]
    result = {
        "rank": int(latest["rank"]),
        "rank_points": float(latest["total_points"]) if pd.notna(latest["total_points"]) else 0.0,
    }
    if len(_RANKING_FEATURE_CACHE) > 50000:
        _RANKING_FEATURE_CACHE.clear()
    _RANKING_FEATURE_CACHE[cache_key] = result
    return result


def _goalscoring_features(
    team: str,
    matches: pd.DataFrame,
    goalscorers: pd.DataFrame | None,
) -> dict:
    """WC avg goals and scoring-first win rate for a team."""
    cache_key = (team, id(matches), id(goalscorers) if goalscorers is not None else 0)
    cached = _GOALSCORING_CACHE.get(cache_key)
    if cached is not None:
        return cached

    team_mask = (matches["home_team"] == team) | (matches["away_team"] == team)
    wc_mask = _wc_mask_for(matches)
    wc_home = matches[team_mask & wc_mask & (matches["home_team"] == team)]
    wc_away = matches[team_mask & wc_mask & (matches["away_team"] == team)]
    wc_n = int(len(wc_home) + len(wc_away))
    wc_goals = int(wc_home["home_score"].sum() + wc_away["away_score"].sum())
    avg_wc = wc_goals / wc_n if wc_n > 0 else 0.0

    # Scoring-first win rate
    scoring_first_wins = scoring_first_total = 0
    if goalscorers is not None:
        team_matches = matches[team_mask]
        first_goals = _first_goal_lookup(goalscorers)
        scored_first = team_matches.merge(first_goals, on=["date", "home_team", "away_team"], how="inner")
        scored_first = scored_first[scored_first["first_scorer_team"] == team]
        scoring_first_total = int(len(scored_first))
        if scoring_first_total:
            home_wins = (
                (scored_first["home_team"] == team)
                & (scored_first["home_score"] > scored_first["away_score"])
            )
            away_wins = (
                (scored_first["away_team"] == team)
                & (scored_first["away_score"] > scored_first["home_score"])
            )
            scoring_first_wins = int((home_wins | away_wins).sum())

    # Bayesian smoothing toward 0.5 prior — prevents extreme values from tiny samples
    sf_win_rate = (scoring_first_wins + 2) / (scoring_first_total + 4)

    result = {
        "avg_goals_wc": avg_wc,
        "scoring_first_win_rate": sf_win_rate,
    }
    if len(_GOALSCORING_CACHE) > 20000:
        _GOALSCORING_CACHE.clear()
    _GOALSCORING_CACHE[cache_key] = result
    return result


def build_feature_row(
    team_a: str,
    team_b: str,
    as_of_date,
    matches_df: pd.DataFrame,
    goalscorers_df: pd.DataFrame | None,
    shootouts_df: pd.DataFrame | None,
    rankings_df: pd.DataFrame | None,
    substitutions_df: pd.DataFrame | None = None,
    player_appearances_df: pd.DataFrame | None = None,
    player_goals_df: pd.DataFrame | None = None,
    award_winners_df: pd.DataFrame | None = None,
    copa_data: dict | None = None,
    friendlies_data: dict | None = None,
    euro_data: dict | None = None,
    venue_context: dict | None = None,
    skip_goalscoring: bool = False,
    skip_supersub: bool = False,
    skip_player_stats: bool = False,
) -> tuple[dict, int]:
    """
    Compute all features for a team_a vs team_b match as of as_of_date.
    Returns (feature_dict, n_data_sources_active).

    All data is filtered to dates strictly before as_of_date — no leakage.
    skip_goalscoring / skip_supersub are training-time perf flags.
    """
    # Leakage guard: only use data available before this match.
    # Note: DATA_FROM_YEAR applies only to player stats (handled in player_stats.py).
    # Match history (form, H2H, WC stats) uses the full historical record so that
    # long-term WC pedigree (e.g. France's 1998 title, Brazil's dominance) is captured.
    _cutoff = pd.Timestamp(as_of_date)
    matches = _window_before("matches", matches_df, _cutoff)
    goalscorers = _window_before("goalscorers", goalscorers_df, _cutoff)
    shootouts = _window_before("shootouts", shootouts_df, _cutoff)
    substitutions = _window_before("substitutions", substitutions_df, _cutoff)
    rankings = rankings_df  # rankings are snapshot-based

    ref_date = pd.Timestamp(as_of_date)

    feat = {}
    venue = venue_context or {}
    feat["team_a_is_home"] = float(venue.get("team_a_is_home", 0.0))
    feat["team_b_is_home"] = float(venue.get("team_b_is_home", 0.0))
    feat["neutral_venue"] = float(venue.get("neutral_venue", 1.0))
    feat["team_a_host_country"] = float(venue.get("team_a_host_country", 0.0))
    feat["team_b_host_country"] = float(venue.get("team_b_host_country", 0.0))

    # Track which feature groups had source data — used for honest data_coverage
    active_groups = 0
    total_groups = 5  # h2h/form, tournament, rankings, goalscoring, supersub

    # H2H + Form
    h2h = _h2h_features(team_a, team_b, matches, ref_date)
    feat.update(h2h)
    form_a = _team_form(team_a, matches)
    for k, v in form_a.items():
        feat[f"team_a_{k}"] = v
    form_b = _team_form(team_b, matches)
    for k, v in form_b.items():
        feat[f"team_b_{k}"] = v
    if not matches.empty:
        active_groups += 1

    # Tournament
    tourn_a = _tournament_features(team_a, matches, shootouts)
    for k, v in tourn_a.items():
        feat[f"team_a_{k}"] = v
    tourn_b = _tournament_features(team_b, matches, shootouts)
    for k, v in tourn_b.items():
        feat[f"team_b_{k}"] = v
    if shootouts is not None:
        active_groups += 1

    # Rankings
    rank_a = _ranking_features(team_a, rankings, ref_date)
    feat["team_a_rank"] = rank_a["rank"]
    feat["team_a_rank_points"] = rank_a["rank_points"]
    rank_b = _ranking_features(team_b, rankings, ref_date)
    feat["team_b_rank"] = rank_b["rank"]
    feat["team_b_rank_points"] = rank_b["rank_points"]
    # rank_diff is only meaningful when both ranks are known (non-zero sentinel)
    if feat["team_a_rank"] > 0 and feat["team_b_rank"] > 0:
        feat["rank_diff"] = feat["team_b_rank"] - feat["team_a_rank"]
        feat["rank_abs_diff"] = abs(feat["rank_diff"])
        feat["rank_closeness"] = 1.0 / (1.0 + (feat["rank_abs_diff"] / 25.0))
    else:
        feat["rank_diff"] = 0.0
        feat["rank_abs_diff"] = 0.0
        feat["rank_closeness"] = 0.0
    if rankings is not None:
        active_groups += 1

    # Draw-tendency and team-similarity signals.
    h2h_total = float(feat.get("h2h_total_meetings", 0.0))
    h2h_draws = float(feat.get("h2h_draws", 0.0))
    feat["h2h_draw_rate"] = h2h_draws / h2h_total if h2h_total > 0 else 0.0
    feat["h2h_balance_score"] = 1.0 - min(abs(float(feat.get("h2h_weighted_win_ratio_a", 0.5)) - 0.5) * 2.0, 1.0)

    team_a_form_total = max(
        float(feat.get("team_a_form_wins", 0.0))
        + float(feat.get("team_a_form_draws", 0.0))
        + float(feat.get("team_a_form_losses", 0.0)),
        1.0,
    )
    team_b_form_total = max(
        float(feat.get("team_b_form_wins", 0.0))
        + float(feat.get("team_b_form_draws", 0.0))
        + float(feat.get("team_b_form_losses", 0.0)),
        1.0,
    )
    team_a_draw_rate = float(feat.get("team_a_form_draws", 0.0)) / team_a_form_total
    team_b_draw_rate = float(feat.get("team_b_form_draws", 0.0)) / team_b_form_total
    feat["combined_form_draw_rate"] = (team_a_draw_rate + team_b_draw_rate) / 2.0
    feat["form_draw_rate_diff"] = abs(team_a_draw_rate - team_b_draw_rate)

    team_a_attack = float(feat.get("team_a_goals_scored_avg", 0.0))
    team_b_attack = float(feat.get("team_b_goals_scored_avg", 0.0))
    team_a_defense = float(feat.get("team_a_goals_conceded_avg", 0.0))
    team_b_defense = float(feat.get("team_b_goals_conceded_avg", 0.0))
    feat["attack_balance_abs_diff"] = abs(team_a_attack - team_b_attack)
    feat["defense_balance_abs_diff"] = abs(team_a_defense - team_b_defense)
    feat["combined_clean_sheet_rate"] = (
        float(feat.get("team_a_clean_sheet_rate", 0.0)) + float(feat.get("team_b_clean_sheet_rate", 0.0))
    ) / 2.0
    avg_total_goal_tendency = (team_a_attack + team_a_defense + team_b_attack + team_b_defense) / 2.0
    feat["low_scoring_tendency"] = max(0.0, min((2.5 - avg_total_goal_tendency) / 2.5, 1.0))
    closeness_parts = [
        float(feat.get("rank_closeness", 0.0)),
        1.0 - min(feat["attack_balance_abs_diff"] / 3.0, 1.0),
        1.0 - min(feat["defense_balance_abs_diff"] / 3.0, 1.0),
        feat["h2h_balance_score"],
        feat["combined_form_draw_rate"],
        feat["low_scoring_tendency"],
    ]
    feat["draw_similarity_score"] = float(np.mean(closeness_parts))

    # Goalscoring
    if not skip_goalscoring:
        gs_a = _goalscoring_features(team_a, matches, goalscorers)
        feat["team_a_avg_goals_wc"] = gs_a["avg_goals_wc"]
        feat["team_a_scoring_first_win_rate"] = gs_a["scoring_first_win_rate"]
        gs_b = _goalscoring_features(team_b, matches, goalscorers)
        feat["team_b_avg_goals_wc"] = gs_b["avg_goals_wc"]
        feat["team_b_scoring_first_win_rate"] = gs_b["scoring_first_win_rate"]
        if goalscorers is not None:
            active_groups += 1
    else:
        feat["team_a_avg_goals_wc"] = 0.0
        feat["team_a_scoring_first_win_rate"] = 0.5
        feat["team_b_avg_goals_wc"] = 0.0
        feat["team_b_scoring_first_win_rate"] = 0.5

    # Super-sub
    if not skip_supersub:
        ss_a = detect_supersub(team_a, goalscorers, matches, substitutions)
        feat["team_a_has_supersub"] = 1 if ss_a else 0
        ss_b = detect_supersub(team_b, goalscorers, matches, substitutions)
        feat["team_b_has_supersub"] = 1 if ss_b else 0
        if goalscorers is not None:
            active_groups += 1
    else:
        feat["team_a_has_supersub"] = 0
        feat["team_b_has_supersub"] = 0

    # Copa America 2024 optional context.
    copa_features = build_copa_feature_values(team_a, team_b, as_of_date, copa_data)
    feat.update(copa_features)
    if copa_data and (
        copa_data.get("matches") is not None
        or copa_data.get("teams") is not None
        or copa_data.get("players") is not None
        or copa_data.get("league") is not None
    ):
        active_groups += 1

    # International friendlies 2026 optional context.
    friendly_features = build_friendlies_feature_values(team_a, team_b, as_of_date, friendlies_data)
    feat.update(friendly_features)
    if friendlies_data and (
        friendlies_data.get("matches") is not None
        or friendlies_data.get("teams") is not None
        or friendlies_data.get("players") is not None
        or friendlies_data.get("league") is not None
    ):
        active_groups += 1

    # EURO 2024 player-strength snapshot.
    euro_features = build_euro_feature_values(team_a, team_b, as_of_date, euro_data)
    feat.update(euro_features)
    if euro_data and euro_data.get("players") is not None:
        active_groups += 1

    # Player impact features
    if not skip_player_stats:
        pf_a = compute_player_features(
            team_a, as_of_date,
            player_appearances_df, player_goals_df, award_winners_df
        )
        for k, v in pf_a.items():
            feat[f"team_a_{k}"] = v

        pf_b = compute_player_features(
            team_b, as_of_date,
            player_appearances_df, player_goals_df, award_winners_df
        )
        for k, v in pf_b.items():
            feat[f"team_b_{k}"] = v

        if player_appearances_df is not None:
            active_groups += 1
    else:
        for prefix in ("team_a_", "team_b_"):
            feat[f"{prefix}top_scorer_wc_goals"]   = 0
            feat[f"{prefix}attack_rating"]          = 0.0
            feat[f"{prefix}golden_ball_count"]      = 0
            feat[f"{prefix}golden_boot_count"]      = 0
            feat[f"{prefix}golden_glove_count"]     = 0
            feat[f"{prefix}avg_player_experience"]  = 0.0

    # Align to canonical feature column order, fill missing with 0
    aligned = {}
    for col in FEATURE_COLUMNS:
        val = feat.get(col)
        aligned[col] = 0.0 if val is None else float(val)

    return aligned, active_groups
