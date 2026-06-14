"""
Single source of truth for feature definitions, tournament weights,
and recency decay. Imported by preprocess.py, train.py, and predict.py.
"""

# Tournament tier weights
TOURNAMENT_WEIGHTS = {
    "fifa world cup": 1.0,
    "copa america": 0.85,
    "uefa euro": 0.85,
    "africa cup of nations": 0.85,
    "afcon": 0.85,
    "afc asian cup": 0.85,
    "asian cup": 0.85,
    "gold cup": 0.75,
    "world cup qualification": 0.7,
    "world cup qualifier": 0.7,
    "wc qualification": 0.7,
    "friendly": 0.3,
    "international friendly": 0.3,
    "uefa nations league": 0.65,
    "copa america centenario": 0.85,
    "confederation cup": 0.75,
    "fifa confederations cup": 0.75,
}

DEFAULT_TOURNAMENT_WEIGHT = 0.5  # for unrecognized tournaments

# Only use data from this year onwards — removes retired players and stale history
DATA_FROM_YEAR = 2006

# Raw context features are especially useful for CatBoost when larger datasets
# include many teams, venues, tournament types, stages, and squad contexts.
RAW_CONTEXT_COLUMNS = [
    "team_a",
    "team_b",
    "tournament",
]


def get_tournament_weight(tournament_name: str) -> float:
    if not tournament_name:
        return DEFAULT_TOURNAMENT_WEIGHT
    normalized = normalize_tournament_name(tournament_name)
    return TOURNAMENT_WEIGHTS.get(normalized, DEFAULT_TOURNAMENT_WEIGHT)


def normalize_tournament_name(tournament_name: str) -> str:
    """Map source tournament strings to stable categories."""
    if not tournament_name:
        return "unknown"

    name = tournament_name.lower().strip()
    if name in TOURNAMENT_WEIGHTS:
        return name

    # More specific checks first so qualifiers don't become plain World Cup.
    if "world cup" in name and ("qualifier" in name or "qualification" in name):
        return "world cup qualification"
    if "world cup" in name:
        return "fifa world cup"
    if "copa america" in name:
        return "copa america"
    if "uefa euro" in name or "european championship" in name:
        return "uefa euro"
    if "africa cup" in name or "afcon" in name:
        return "afcon"
    if "asian cup" in name:
        return "asian cup"
    if "gold cup" in name:
        return "gold cup"
    if "nations league" in name:
        return "uefa nations league"
    if "confederation" in name:
        return "confederation cup"
    if "friendly" in name:
        return "international friendly"
    return name


def recency_weight(match_date, reference_date) -> float:
    """Linear decay: 1.0 for matches within 2 years, 0.4 for 10+ years old."""
    days_old = (reference_date - match_date).days
    years_old = days_old / 365.25
    if years_old <= 2:
        return 1.0
    elif years_old >= 10:
        return 0.4
    else:
        # linear interpolation between 2y (1.0) and 10y (0.4)
        return 1.0 - (years_old - 2) / 8 * 0.6


# Canonical feature column names — defines model input shape
FEATURE_COLUMNS = [
    # Venue context
    "team_a_is_home",
    "team_b_is_home",
    "neutral_venue",
    "team_a_host_country",
    "team_b_host_country",
    # H2H
    "h2h_total_meetings",
    "h2h_team_a_wins",
    "h2h_team_b_wins",
    "h2h_draws",
    "h2h_wc_only_team_a_wins",
    "h2h_wc_only_team_b_wins",
    "h2h_last5_team_a_wins",
    "h2h_last5_goals_a",
    "h2h_last5_goals_b",
    "h2h_weighted_win_ratio_a",
    # Team A form
    "team_a_form_wins",
    "team_a_form_draws",
    "team_a_form_losses",
    "team_a_goals_scored_avg",
    "team_a_goals_conceded_avg",
    "team_a_clean_sheet_rate",
    "team_a_win_streak",
    "team_a_unbeaten_streak",
    # Team B form
    "team_b_form_wins",
    "team_b_form_draws",
    "team_b_form_losses",
    "team_b_goals_scored_avg",
    "team_b_goals_conceded_avg",
    "team_b_clean_sheet_rate",
    "team_b_win_streak",
    "team_b_unbeaten_streak",
    # Tournament
    "team_a_wc_titles",
    "team_a_wc_finals_reached",
    "team_a_wc_win_rate_knockouts",
    "team_a_penalty_shootout_wins",
    "team_a_penalty_shootout_losses",
    "team_b_wc_titles",
    "team_b_wc_finals_reached",
    "team_b_wc_win_rate_knockouts",
    "team_b_penalty_shootout_wins",
    "team_b_penalty_shootout_losses",
    # Rankings
    "team_a_rank",
    "team_b_rank",
    "rank_diff",
    "rank_abs_diff",
    "rank_closeness",
    "team_a_rank_points",
    "team_b_rank_points",
    # Draw tendency / team similarity
    "h2h_draw_rate",
    "h2h_balance_score",
    "combined_form_draw_rate",
    "form_draw_rate_diff",
    "attack_balance_abs_diff",
    "defense_balance_abs_diff",
    "combined_clean_sheet_rate",
    "low_scoring_tendency",
    "draw_similarity_score",
    # Goalscoring
    "team_a_avg_goals_wc",
    "team_b_avg_goals_wc",
    "team_a_scoring_first_win_rate",
    "team_b_scoring_first_win_rate",
    # Super-sub (binary)
    "team_a_has_supersub",
    "team_b_has_supersub",
    # Copa America 2024 match context (pre-match only)
    "team_a_copa_pre_match_ppg",
    "team_b_copa_pre_match_ppg",
    "team_a_copa_pre_match_xg",
    "team_b_copa_pre_match_xg",
    "team_a_copa_odds_win",
    "draw_copa_odds",
    "team_b_copa_odds_win",
    "team_a_copa_implied_win_prob",
    "draw_copa_implied_prob",
    "team_b_copa_implied_win_prob",
    # Copa America 2024 aggregate context (available after tournament completion)
    "team_a_copa_team_ppg",
    "team_a_copa_team_xg_for",
    "team_a_copa_team_xg_against",
    "team_a_copa_team_goals_per_match",
    "team_a_copa_team_conceded_per_match",
    "team_a_copa_player_avg_rating",
    "team_a_copa_player_goal_rate",
    "team_a_copa_player_xg_per90",
    "team_b_copa_team_ppg",
    "team_b_copa_team_xg_for",
    "team_b_copa_team_xg_against",
    "team_b_copa_team_goals_per_match",
    "team_b_copa_team_conceded_per_match",
    "team_b_copa_player_avg_rating",
    "team_b_copa_player_goal_rate",
    "team_b_copa_player_xg_per90",
    "copa_league_avg_goals",
    "copa_league_xg_avg",
    "copa_league_home_advantage",
    # International friendlies 2026 match context (pre-match only)
    "team_a_friendlies_pre_match_ppg",
    "team_b_friendlies_pre_match_ppg",
    "team_a_friendlies_pre_match_xg",
    "team_b_friendlies_pre_match_xg",
    "team_a_friendlies_odds_win",
    "draw_friendlies_odds",
    "team_b_friendlies_odds_win",
    "team_a_friendlies_implied_win_prob",
    "draw_friendlies_implied_prob",
    "team_b_friendlies_implied_win_prob",
    # International friendlies 2026 aggregate context (available after snapshot end)
    "team_a_friendlies_team_ppg",
    "team_a_friendlies_team_xg_for",
    "team_a_friendlies_team_xg_against",
    "team_a_friendlies_team_goals_per_match",
    "team_a_friendlies_team_conceded_per_match",
    "team_a_friendlies_player_avg_rating",
    "team_a_friendlies_player_goal_rate",
    "team_a_friendlies_player_xg_per90",
    "team_b_friendlies_team_ppg",
    "team_b_friendlies_team_xg_for",
    "team_b_friendlies_team_xg_against",
    "team_b_friendlies_team_goals_per_match",
    "team_b_friendlies_team_conceded_per_match",
    "team_b_friendlies_player_avg_rating",
    "team_b_friendlies_player_goal_rate",
    "team_b_friendlies_player_xg_per90",
    "friendlies_league_avg_goals",
    "friendlies_league_xg_avg",
    "friendlies_league_home_advantage",
    # EURO 2024 player-strength snapshot (available after EURO 2024 completed)
    "team_a_euro_squad_impact",
    "team_a_euro_top5_impact",
    "team_a_euro_attack_impact",
    "team_a_euro_midfield_impact",
    "team_a_euro_defense_impact",
    "team_a_euro_goalkeeping_impact",
    "team_a_euro_minutes_coverage",
    "team_a_euro_team_ppg",
    "team_a_euro_team_goal_diff_per_match",
    "team_a_euro_team_goals_per_match",
    "team_a_euro_team_conceded_per_match",
    "team_a_euro_team_shots_per_match",
    "team_a_euro_team_shots_on_target_rate",
    "team_a_euro_team_pass_accuracy",
    "team_a_euro_team_defensive_actions_per_match",
    "team_a_euro_team_set_piece_pressure",
    "team_a_euro_team_discipline_risk",
    "team_a_euro_leader_goals_per_match",
    "team_a_euro_leader_assists_per_match",
    "team_a_euro_leader_goal_contrib_per_match",
    "team_b_euro_squad_impact",
    "team_b_euro_top5_impact",
    "team_b_euro_attack_impact",
    "team_b_euro_midfield_impact",
    "team_b_euro_defense_impact",
    "team_b_euro_goalkeeping_impact",
    "team_b_euro_minutes_coverage",
    "team_b_euro_team_ppg",
    "team_b_euro_team_goal_diff_per_match",
    "team_b_euro_team_goals_per_match",
    "team_b_euro_team_conceded_per_match",
    "team_b_euro_team_shots_per_match",
    "team_b_euro_team_shots_on_target_rate",
    "team_b_euro_team_pass_accuracy",
    "team_b_euro_team_defensive_actions_per_match",
    "team_b_euro_team_set_piece_pressure",
    "team_b_euro_team_discipline_risk",
    "team_b_euro_leader_goals_per_match",
    "team_b_euro_leader_assists_per_match",
    "team_b_euro_leader_goal_contrib_per_match",
    # Player impact
    "team_a_top_scorer_wc_goals",
    "team_a_attack_rating",
    "team_a_golden_ball_count",
    "team_a_golden_boot_count",
    "team_a_golden_glove_count",
    "team_a_avg_player_experience",
    "team_b_top_scorer_wc_goals",
    "team_b_attack_rating",
    "team_b_golden_ball_count",
    "team_b_golden_boot_count",
    "team_b_golden_glove_count",
    "team_b_avg_player_experience",
]

MODEL_INPUT_COLUMNS = RAW_CONTEXT_COLUMNS + FEATURE_COLUMNS

# Known World Cup winners and their title counts (used as fallback baseline)
WC_TITLES = {
    "Brazil": 5,
    "Germany": 4,
    "West Germany": 4,
    "Italy": 4,
    "Argentina": 3,
    "France": 2,
    "Uruguay": 2,
    "England": 1,
    "Spain": 1,
}
