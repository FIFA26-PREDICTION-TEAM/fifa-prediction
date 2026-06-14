"""2026 World Cup team, squad, and key-player context."""

from __future__ import annotations

import os
import re
import unicodedata

import numpy as np
import pandas as pd
import streamlit as st

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXTERNAL = os.path.join(_ROOT, "data", "external", "world_cup_2026")
TEAMS_PATH = os.path.join(_EXTERNAL, "FIFA_World_Cup_2026_Teams.csv")
SQUADS_PATH = os.path.join(_EXTERNAL, "fifa_world_cup_2026_squads.csv")
NON_WC_SQUADS_PATH = os.path.join(_EXTERNAL, "non_world_cup_2026_squads.csv")
WORLD_CUP_2026_MATCHES_REPO_PATH = os.path.join(_EXTERNAL, "wc2026_matches.csv")
FIFA_RANKING_SNAPSHOT_DATE = "2026-04-01"
FIFA_RANKING_SOURCE = "FIFA/Coca-Cola Men's World Ranking"
LIVE_DATA_CACHE_TTL_SECONDS = int(os.getenv("ML_PRJCT_LIVE_DATA_CACHE_TTL_SECONDS", "60"))

TEAM_REPLACEMENTS = {
    "Bosnia And Herzegovina": "Bosnia & Herzegovina",
    "Bosnia and Herzegovina": "Bosnia & Herzegovina",
    "Cabo Verde": "Cape Verde",
    "Cape Verde Islands": "Cape Verde",
    "Congo DR": "DR Congo",
    "Czechia": "Czech Republic",
    "Cote D'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Côte D'Ivoire": "Ivory Coast",
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "USA": "United States",
    "United States of America": "United States",
}


def _default_path(filename: str) -> str:
    return os.path.join(os.path.expanduser("~"), "Downloads", filename)


def _first_existing(*paths: str) -> str | None:
    return next((path for path in paths if path and os.path.exists(path)), None)


WORLD_CUP_2026_MATCHES_PATH = os.getenv(
    "ML_PRJCT_WORLD_CUP_2026_MATCHES_PATH",
) or _first_existing(_default_path("wc2026_matches.csv"), WORLD_CUP_2026_MATCHES_REPO_PATH) or WORLD_CUP_2026_MATCHES_REPO_PATH

REQUIRED_COLUMNS = [
    "Group",
    "Team",
    "Confederation",
    "FIFA_Ranking",
    "Key_Player_1",
    "KP1_Position",
    "KP1_Club",
    "KP1_Notable_Achievement",
    "Key_Player_2",
    "KP2_Position",
    "KP2_Club",
    "KP2_Notable_Achievement",
    "Team_Best_WC_Finish",
    "WC_Debut",
    "Notes",
]

# Optional extra key-player slots (KP3–KP6); filled only when a team has more curated players.
OPTIONAL_KP_SLOTS = [
    ("Key_Player_3", "KP3_Position", "KP3_Club", "KP3_Notable_Achievement"),
    ("Key_Player_4", "KP4_Position", "KP4_Club", "KP4_Notable_Achievement"),
    ("Key_Player_5", "KP5_Position", "KP5_Club", "KP5_Notable_Achievement"),
    ("Key_Player_6", "KP6_Position", "KP6_Club", "KP6_Notable_Achievement"),
]

CONFEDERATION_PRIOR = {
    "UEFA": 0.68,
    "CONMEBOL": 0.66,
    "CAF": 0.52,
    "CONCACAF": 0.50,
    "AFC": 0.48,
    "OFC": 0.36,
}

TOP_LEAGUE_COUNTRIES = {"ENG", "ESP", "GER", "ITA", "FRA"}
CLUB_COUNTRY_PRIOR = {
    "ENG": 0.96,
    "ESP": 0.94,
    "GER": 0.91,
    "ITA": 0.89,
    "FRA": 0.86,
    "NED": 0.80,
    "POR": 0.79,
    "BEL": 0.73,
    "TUR": 0.70,
    "USA": 0.68,
    "BRA": 0.67,
    "ARG": 0.67,
    "MEX": 0.64,
    "KSA": 0.61,
    "QAT": 0.56,
}


def normalize_team_name(team: str) -> str:
    if pd.isna(team):
        return ""
    value = str(team).strip()
    return TEAM_REPLACEMENTS.get(value, value)


def _name_key(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return " ".join(sorted(tokens))


def _title_name(name: str) -> str:
    """Convert squad-CSV ALL-CAPS names ('MBAPPE Kylian') to title case ('Mbappe Kylian')."""
    return " ".join(w.capitalize() for w in str(name).split())


def _clean_club(club: str) -> str:
    """Strip the '(ENG)'-style country suffix from club names."""
    return str(club).rsplit("(", 1)[0].strip()


def _finish_score(value: str) -> float:
    text = str(value or "").strip().lower()
    if "winner" in text:
        return 1.0
    if "runner-up" in text or "runner up" in text:
        return 0.86
    if "third place" in text or "semi-final" in text or "semifinal" in text:
        return 0.74
    if "quarter" in text:
        return 0.62
    if "round of 16" in text:
        return 0.46
    if "group" in text:
        return 0.28
    if "debut" in text or "never" in text:
        return 0.14
    return 0.30


def _achievement_score(*texts: str) -> float:
    joined = " ".join(str(text or "") for text in texts).lower()
    score = 0.45
    keyword_weights = {
        r"\bballon": 0.16,
        r"champions league|ucl": 0.14,
        r"world cup|wc ": 0.13,
        r"golden boot|golden ball|player of year": 0.12,
        r"premier league|serie a|bundesliga|la liga|ligue 1": 0.08,
        r"\b100\+|\b120\+|\b200\+|\b500\+": 0.06,
        r"captain|all-time|top scorer|leading scorer": 0.06,
        r"youngest|debutant": 0.03,
    }
    for pattern, weight in keyword_weights.items():
        if re.search(pattern, joined):
            score += weight
    return float(np.clip(score, 0.0, 1.0))


def _age_balance_score(avg_age: float) -> float:
    if not np.isfinite(avg_age):
        return 0.5
    return float(np.clip(1.0 - abs(avg_age - 27.4) / 8.0, 0.0, 1.0))


def _club_strength_score(club_country_code: str) -> float:
    return float(CLUB_COUNTRY_PRIOR.get(str(club_country_code or "").strip().upper(), 0.52))


def _prepare_squads(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    squads = df.copy()
    squads["team"] = squads["team"].map(normalize_team_name)
    squads["shirt_number"] = pd.to_numeric(squads["shirt_number"], errors="coerce").fillna(0).astype(int)
    squads["height_cm"] = pd.to_numeric(squads["height_cm"], errors="coerce").fillna(0).astype(float)
    squads["age_on_2026_06_11"] = pd.to_numeric(
        squads["age_on_2026_06_11"], errors="coerce"
    ).fillna(0).astype(float)
    squads["club_country_code"] = squads["club_country_code"].fillna("").astype(str).str.upper()
    squads["position"] = squads["position"].fillna("").astype(str)
    squads["club_strength_score"] = squads["club_country_code"].map(_club_strength_score)
    squads["is_top_league"] = squads["club_country_code"].isin(TOP_LEAGUE_COUNTRIES).astype(int)
    squads["is_domestic_club"] = (squads["club_country_code"] == squads["team_code"]).astype(int)

    grouped = squads.groupby("team", dropna=False)
    summaries = grouped.agg(
        team_code=("team_code", "first"),
        squad_players=("player_name", "count"),
        squad_avg_age=("age_on_2026_06_11", "mean"),
        squad_avg_height_cm=("height_cm", "mean"),
        squad_club_strength=("club_strength_score", "mean"),
        squad_top_league_share=("is_top_league", "mean"),
        squad_domestic_share=("is_domestic_club", "mean"),
        head_coach=("head_coach", "first"),
        coach_nationality=("coach_nationality", "first"),
        source_generated_utc=("source_generated_utc", "first"),
    ).reset_index()

    position_counts = (
        squads.pivot_table(index="team", columns="position", values="player_name", aggfunc="count", fill_value=0)
        .rename(columns={"GK": "squad_goalkeepers", "DF": "squad_defenders", "MF": "squad_midfielders", "FW": "squad_forwards"})
        .reset_index()
    )
    summaries = summaries.merge(position_counts, on="team", how="left")
    for col in ["squad_goalkeepers", "squad_defenders", "squad_midfielders", "squad_forwards"]:
        summaries[col] = pd.to_numeric(summaries.get(col, 0), errors="coerce").fillna(0).astype(int)

    summaries["squad_age_balance"] = summaries["squad_avg_age"].map(_age_balance_score)
    summaries["squad_depth_score"] = (
        (summaries["squad_club_strength"] * 0.46)
        + (summaries["squad_top_league_share"] * 0.24)
        + (summaries["squad_age_balance"] * 0.18)
        + ((summaries["squad_players"].clip(upper=26) / 26.0) * 0.12)
    ).clip(0.0, 1.0)
    return squads.reset_index(drop=True), summaries.reset_index(drop=True)


def _parse_match_dates(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, format="%d-%m-%Y", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(values[missing], errors="coerce", dayfirst=True)
    return parsed


def _prepare_current_matches(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    required = {"date", "team_a", "team_b", "team_a_score", "team_b_score"}
    missing = sorted(required - set(df.columns))
    if missing:
        st.warning(f"`{os.path.basename(WORLD_CUP_2026_MATCHES_PATH)}` is missing columns: {missing}")
        return pd.DataFrame()

    matches = df.copy()
    matches["date"] = _parse_match_dates(matches["date"])
    matches["team_a"] = matches["team_a"].map(normalize_team_name)
    matches["team_b"] = matches["team_b"].map(normalize_team_name)
    matches["team_a_score"] = pd.to_numeric(matches["team_a_score"], errors="coerce")
    matches["team_b_score"] = pd.to_numeric(matches["team_b_score"], errors="coerce")
    matches = matches.dropna(subset=["date", "team_a", "team_b", "team_a_score", "team_b_score"])
    matches["team_a_score"] = matches["team_a_score"].astype(int)
    matches["team_b_score"] = matches["team_b_score"].astype(int)
    matches["tournament"] = matches.get("tournament", "FIFA World Cup 2026")
    matches["stage"] = matches.get("stage", "")
    matches["group"] = matches.get("group", "")
    matches["city"] = matches.get("city", "")
    matches["venue"] = matches.get("venue", "")
    if "neutral_venue" in matches.columns:
        neutral = pd.to_numeric(matches["neutral_venue"], errors="coerce").fillna(1)
    else:
        neutral = pd.Series(1, index=matches.index)
    matches["neutral_venue"] = neutral.astype(bool)
    if "match_number" in matches.columns:
        matches["match_number"] = pd.to_numeric(matches["match_number"], errors="coerce")
        matches = matches.sort_values(["date", "match_number"], na_position="last")
    else:
        matches = matches.sort_values("date")
    key_cols = ["date", "team_a", "team_b", "tournament"]
    if "match_number" in matches.columns:
        key_cols = ["match_number"]
    return matches.drop_duplicates(key_cols, keep="last").reset_index(drop=True)


def _stat(row: pd.Series, prefix: str, stat: str, default: float = 0.0) -> float:
    value = pd.to_numeric(row.get(f"{prefix}_{stat}", default), errors="coerce")
    return 0.0 if pd.isna(value) else float(value)


def _derive_current_team_stats(matches: pd.DataFrame) -> pd.DataFrame:
    if matches is None or matches.empty:
        return pd.DataFrame()

    rows = []
    for _, match in matches.iterrows():
        a_score = int(match["team_a_score"])
        b_score = int(match["team_b_score"])
        for prefix, opponent_prefix, team_col, opponent_col, gf, ga in (
            ("team_a", "team_b", "team_a", "team_b", a_score, b_score),
            ("team_b", "team_a", "team_b", "team_a", b_score, a_score),
        ):
            rows.append(
                {
                    "team": match[team_col],
                    "opponent": match[opponent_col],
                    "played": 1,
                    "wins": int(gf > ga),
                    "draws": int(gf == ga),
                    "losses": int(gf < ga),
                    "points": 3 if gf > ga else 1 if gf == ga else 0,
                    "goals_for": gf,
                    "goals_against": ga,
                    "clean_sheets": int(ga == 0),
                    "shots_total": _stat(match, prefix, "shots_total"),
                    "shots_on_target": _stat(match, prefix, "shots_on_target"),
                    "shots_against": _stat(match, opponent_prefix, "shots_total"),
                    "shots_on_target_against": _stat(match, opponent_prefix, "shots_on_target"),
                    "possession": _stat(match, prefix, "possession", np.nan),
                    "corners": _stat(match, prefix, "corners"),
                    "yellow_cards": _stat(match, prefix, "yellow_cards"),
                    "red_cards": _stat(match, prefix, "red_cards"),
                }
            )

    team_rows = pd.DataFrame(rows)
    if team_rows.empty:
        return pd.DataFrame()

    grouped = team_rows.groupby("team", as_index=False).agg(
        played=("played", "sum"),
        wins=("wins", "sum"),
        draws=("draws", "sum"),
        losses=("losses", "sum"),
        points=("points", "sum"),
        goals_for=("goals_for", "sum"),
        goals_against=("goals_against", "sum"),
        clean_sheets=("clean_sheets", "sum"),
        shots_total=("shots_total", "sum"),
        shots_on_target=("shots_on_target", "sum"),
        shots_against=("shots_against", "sum"),
        shots_on_target_against=("shots_on_target_against", "sum"),
        possession_avg=("possession", "mean"),
        corners=("corners", "sum"),
        yellow_cards=("yellow_cards", "sum"),
        red_cards=("red_cards", "sum"),
    )
    played = grouped["played"].replace(0, np.nan)
    grouped["goal_difference"] = grouped["goals_for"] - grouped["goals_against"]
    grouped["points_per_match"] = grouped["points"] / played
    grouped["goals_for_per_match"] = grouped["goals_for"] / played
    grouped["goals_against_per_match"] = grouped["goals_against"] / played
    grouped["shot_diff_per_match"] = (grouped["shots_total"] - grouped["shots_against"]) / played
    grouped["shots_on_target_diff_per_match"] = (
        grouped["shots_on_target"] - grouped["shots_on_target_against"]
    ) / played
    grouped["clean_sheet_rate"] = grouped["clean_sheets"] / played
    grouped["performance_score"] = grouped.apply(lambda row: _current_tournament_score(row.to_dict()), axis=1)
    return grouped.fillna(0.0).reset_index(drop=True)


def _current_tournament_score(summary: dict | None) -> float:
    if not summary or float(summary.get("played", 0.0)) <= 0:
        return 0.5

    played = max(float(summary.get("played", 0.0)), 1.0)
    ppg = float(summary.get("points_per_match", 0.0))
    goal_diff_per_match = float(summary.get("goal_difference", 0.0)) / played
    goals_for = float(summary.get("goals_for_per_match", 0.0))
    goals_against = float(summary.get("goals_against_per_match", 0.0))
    shot_diff = float(summary.get("shots_on_target_diff_per_match", summary.get("shot_diff_per_match", 0.0)))
    possession = float(summary.get("possession_avg", 50.0))
    score = (
        np.clip(ppg / 3.0, 0.0, 1.0) * 0.38
        + np.clip((goal_diff_per_match + 2.0) / 4.0, 0.0, 1.0) * 0.22
        + np.clip(goals_for / 3.0, 0.0, 1.0) * 0.16
        + np.clip(1.0 - goals_against / 3.0, 0.0, 1.0) * 0.12
        + np.clip((shot_diff + 6.0) / 12.0, 0.0, 1.0) * 0.07
        + np.clip(possession / 100.0, 0.0, 1.0) * 0.05
    )
    return float(np.clip(score, 0.0, 1.0))


def append_world_cup_2026_matches(matches_df: pd.DataFrame, world_cup_2026_data: dict | None = None) -> pd.DataFrame:
    if matches_df is None:
        return matches_df
    world_cup_2026_data = world_cup_2026_data or load_world_cup_2026_data()
    current_matches = world_cup_2026_data.get("matches") if world_cup_2026_data else None
    if current_matches is None or current_matches.empty:
        return matches_df

    extra = pd.DataFrame(
        {
            "date": current_matches["date"],
            "home_team": current_matches["team_a"],
            "away_team": current_matches["team_b"],
            "home_score": current_matches["team_a_score"],
            "away_score": current_matches["team_b_score"],
            "tournament": current_matches["tournament"].fillna("FIFA World Cup 2026"),
            "city": current_matches.get("city", ""),
            "country": "",
            "neutral": current_matches.get("neutral_venue", True),
        }
    ).dropna(subset=["date", "home_team", "away_team"])

    combined = pd.concat([matches_df, extra], ignore_index=True, sort=False)
    combined["_key"] = (
        pd.to_datetime(combined["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        + "|"
        + combined["home_team"].astype(str)
        + "|"
        + combined["away_team"].astype(str)
        + "|"
        + combined["tournament"].astype(str)
    )
    combined = combined.drop_duplicates("_key", keep="last").drop(columns=["_key"])
    return combined.sort_values("date").reset_index(drop=True)


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in REQUIRED_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    for slot in OPTIONAL_KP_SLOTS:
        for col in slot:
            if col not in out.columns:
                out[col] = ""
    keep = REQUIRED_COLUMNS + [c for slot in OPTIONAL_KP_SLOTS for c in slot]
    out = out[keep].copy()
    for col in [c for slot in OPTIONAL_KP_SLOTS for c in slot]:
        out[col] = out[col].fillna("").astype(str).replace("nan", "")
    out["Team"] = out["Team"].map(normalize_team_name)
    out["FIFA_Ranking"] = pd.to_numeric(out["FIFA_Ranking"], errors="coerce").fillna(999).astype(int)
    out["rank_score"] = ((211 - out["FIFA_Ranking"].clip(lower=1, upper=211)) / 210).astype(float)
    out["history_score"] = out["Team_Best_WC_Finish"].map(_finish_score).astype(float)
    out["confederation_score"] = out["Confederation"].map(CONFEDERATION_PRIOR).fillna(0.45).astype(float)
    out["key_player_score"] = out.apply(
        lambda row: _achievement_score(
            row["Key_Player_1"],
            row["KP1_Notable_Achievement"],
            row["Key_Player_2"],
            row["KP2_Notable_Achievement"],
        ),
        axis=1,
    )
    debut_penalty = out["WC_Debut"].astype(str).str.strip().str.lower().isin({"yes", "true", "1"}).astype(float) * 0.06
    out["squad_strength_score"] = (
        (out["rank_score"] * 0.52)
        + (out["history_score"] * 0.22)
        + (out["key_player_score"] * 0.18)
        + (out["confederation_score"] * 0.08)
        - debut_penalty
    ).clip(0.0, 1.0)
    return out.drop_duplicates("Team", keep="first").reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=LIVE_DATA_CACHE_TTL_SECONDS)
def load_world_cup_2026_data() -> dict:
    teams = pd.DataFrame()
    squads = pd.DataFrame()
    squad_summary = pd.DataFrame()
    non_wc_squads = pd.DataFrame()
    matches = pd.DataFrame()
    current_team_stats = pd.DataFrame()

    if os.path.exists(TEAMS_PATH):
        df = pd.read_csv(TEAMS_PATH)
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            st.warning(f"`{os.path.basename(TEAMS_PATH)}` is missing columns: {missing}")
        teams = _prepare(df)

    if os.path.exists(SQUADS_PATH):
        squad_df = pd.read_csv(SQUADS_PATH)
        required = {
            "team", "team_code", "shirt_number", "position", "player_name",
            "date_of_birth", "age_on_2026_06_11", "club", "club_country_code", "height_cm",
        }
        missing = sorted(required - set(squad_df.columns))
        if missing:
            st.warning(f"`{os.path.basename(SQUADS_PATH)}` is missing columns: {missing}")
        else:
            squads, squad_summary = _prepare_squads(squad_df)

    if os.path.exists(NON_WC_SQUADS_PATH):
        non_wc_squads = pd.read_csv(NON_WC_SQUADS_PATH)
        required = {"team", "shirt_number", "player_name", "squad_role", "source_label", "source_match"}
        missing = sorted(required - set(non_wc_squads.columns))
        if missing:
            st.warning(f"`{os.path.basename(NON_WC_SQUADS_PATH)}` is missing columns: {missing}")
            non_wc_squads = pd.DataFrame()
        else:
            non_wc_squads = non_wc_squads.copy()
            non_wc_squads["team"] = non_wc_squads["team"].map(normalize_team_name)
            non_wc_squads["shirt_number"] = pd.to_numeric(
                non_wc_squads["shirt_number"], errors="coerce"
            ).fillna(0).astype(int)
            non_wc_squads = non_wc_squads.sort_values(["team", "squad_role", "shirt_number"]).reset_index(drop=True)

    if os.path.exists(WORLD_CUP_2026_MATCHES_PATH):
        matches = _prepare_current_matches(pd.read_csv(WORLD_CUP_2026_MATCHES_PATH, encoding="utf-8-sig"))
        current_team_stats = _derive_current_team_stats(matches)

    return {
        "teams": teams,
        "squads": squads,
        "squad_summary": squad_summary,
        "non_wc_squads": non_wc_squads,
        "matches": matches,
        "current_team_stats": current_team_stats,
        "path": TEAMS_PATH,
        "squads_path": SQUADS_PATH,
        "non_wc_squads_path": NON_WC_SQUADS_PATH,
        "matches_path": WORLD_CUP_2026_MATCHES_PATH,
        "rows": int(len(teams)),
        "squad_rows": int(len(squads)),
        "non_wc_squad_rows": int(len(non_wc_squads)),
        "match_rows": int(len(matches)),
        "ranking_snapshot_date": FIFA_RANKING_SNAPSHOT_DATE,
        "ranking_source": FIFA_RANKING_SOURCE,
    }


def _team_squad_summary(team: str, world_cup_2026_data: dict | None) -> dict | None:
    if not world_cup_2026_data:
        return None
    summaries = world_cup_2026_data.get("squad_summary")
    if summaries is None or summaries.empty:
        return None
    normalized = normalize_team_name(team)
    match = summaries[summaries["team"] == normalized]
    if match.empty:
        return None
    row = match.iloc[0].to_dict()
    return {
        "players": int(row.get("squad_players", 0)),
        "avg_age": float(row.get("squad_avg_age", 0.0)),
        "avg_height_cm": float(row.get("squad_avg_height_cm", 0.0)),
        "club_strength": float(row.get("squad_club_strength", 0.0)),
        "top_league_share": float(row.get("squad_top_league_share", 0.0)),
        "domestic_share": float(row.get("squad_domestic_share", 0.0)),
        "age_balance": float(row.get("squad_age_balance", 0.0)),
        "depth_score": float(row.get("squad_depth_score", 0.0)),
        "goalkeepers": int(row.get("squad_goalkeepers", 0)),
        "defenders": int(row.get("squad_defenders", 0)),
        "midfielders": int(row.get("squad_midfielders", 0)),
        "forwards": int(row.get("squad_forwards", 0)),
        "head_coach": row.get("head_coach", ""),
        "coach_nationality": row.get("coach_nationality", ""),
        "source_generated_utc": row.get("source_generated_utc", ""),
    }


def _team_current_tournament_summary(team: str, world_cup_2026_data: dict | None) -> dict | None:
    if not world_cup_2026_data:
        return None
    stats = world_cup_2026_data.get("current_team_stats")
    if stats is None or stats.empty:
        return None
    normalized = normalize_team_name(team)
    match = stats[stats["team"] == normalized]
    if match.empty:
        return None
    row = match.iloc[0].to_dict()
    numeric_fields = [
        "played",
        "wins",
        "draws",
        "losses",
        "points",
        "goals_for",
        "goals_against",
        "clean_sheets",
        "goal_difference",
        "points_per_match",
        "goals_for_per_match",
        "goals_against_per_match",
        "shot_diff_per_match",
        "shots_on_target_diff_per_match",
        "possession_avg",
        "clean_sheet_rate",
    ]
    summary = {"team": row.get("team", normalized)}
    for field in numeric_fields:
        value = pd.to_numeric(row.get(field, 0.0), errors="coerce")
        summary[field] = 0.0 if pd.isna(value) else float(value)
    summary["performance_score"] = _current_tournament_score(summary)
    return summary


def is_qualified_team(team: str, world_cup_2026_data: dict | None) -> bool:
    if not world_cup_2026_data:
        return False
    teams = world_cup_2026_data.get("teams")
    if teams is None or teams.empty:
        return False
    normalized = normalize_team_name(team)
    return bool((teams["Team"] == normalized).any())


def get_non_world_cup_2026_squad(team: str, world_cup_2026_data: dict | None) -> list[dict]:
    if not world_cup_2026_data:
        return []
    squads = world_cup_2026_data.get("non_wc_squads")
    if squads is None or squads.empty:
        return []
    normalized = normalize_team_name(team)
    rows = squads[squads["team"] == normalized]
    if rows.empty:
        return []
    return rows.sort_values(["squad_role", "shirt_number"]).to_dict("records")


def get_team_profile(team: str, world_cup_2026_data: dict | None) -> dict | None:
    if not world_cup_2026_data:
        return None
    teams = world_cup_2026_data.get("teams")
    if teams is None or teams.empty:
        return None
    normalized = normalize_team_name(team)
    match = teams[teams["Team"] == normalized]
    squad = _team_squad_summary(team, world_cup_2026_data)
    if match.empty and not squad:
        return None
    row = match.iloc[0].to_dict() if not match.empty else {}
    team_name = row.get("Team") or normalize_team_name(team)
    base_strength = float(row.get("squad_strength_score", 0.45))
    squad_depth = float(squad.get("depth_score", 0.45)) if squad else 0.45
    strength = (base_strength * 0.70) + (squad_depth * 0.30) if squad else base_strength
    current_tournament = _team_current_tournament_summary(team, world_cup_2026_data)
    return {
        "team": team_name,
        "group": row.get("Group", ""),
        "confederation": row.get("Confederation", ""),
        "fifa_ranking": int(row.get("FIFA_Ranking", 999)),
        "best_wc_finish": row.get("Team_Best_WC_Finish", ""),
        "wc_debut": str(row.get("WC_Debut", "")).strip().lower() in {"yes", "true", "1"},
        "key_players": [
            p for p in [
                {
                    "name": row.get("Key_Player_1", ""),
                    "position": row.get("KP1_Position", ""),
                    "club": row.get("KP1_Club", ""),
                    "achievement": row.get("KP1_Notable_Achievement", ""),
                },
                {
                    "name": row.get("Key_Player_2", ""),
                    "position": row.get("KP2_Position", ""),
                    "club": row.get("KP2_Club", ""),
                    "achievement": row.get("KP2_Notable_Achievement", ""),
                },
            ] + [
                {
                    "name": row.get(kp_col, ""),
                    "position": row.get(pos_col, ""),
                    "club": row.get(club_col, ""),
                    "achievement": row.get(ach_col, ""),
                }
                for kp_col, pos_col, club_col, ach_col in OPTIONAL_KP_SLOTS
            ]
            if str(p.get("name", "")).strip()
        ],
        "notes": row.get("Notes", ""),
        "strength": float(np.clip(strength, 0.0, 1.0)),
        "base_strength": base_strength,
        "official_squad": squad,
        "current_tournament": current_tournament,
    }


def get_world_cup_2026_key_players(team: str, world_cup_2026_data: dict | None, limit: int = 5) -> list[dict]:
    if not world_cup_2026_data:
        return []
    squads = world_cup_2026_data.get("squads")
    if squads is None or squads.empty:
        return []

    normalized = normalize_team_name(team)
    team_squad = squads[squads["team"] == normalized].copy()
    if team_squad.empty:
        return []

    profile = get_team_profile(team, world_cup_2026_data)
    curated = {
        _name_key(p.get("name", "")): p
        for p in (profile or {}).get("key_players", [])
        if str(p.get("name", "")).strip()
    }
    # If a team has explicitly curated more than 2 players (KP3+), show only those
    # curated players — no algorithmic fill on top. For teams with 1–2 curated
    # players the algorithm fills the remaining spots up to `limit`.
    # No restriction on curated players — show all of them; only fall back to `limit` when
    # no curated players exist so the algorithm still fills a sensible number of spots.
    effective_limit = len(curated) if curated else limit
    position_priority = {"FW": 0.08, "MF": 0.06, "GK": 0.04, "DF": 0.03}
    team_squad["curated_bonus"] = team_squad["player_name"].map(_name_key).map(
        lambda name: 0.35 if name in curated else 0.0
    )
    team_squad["position_bonus"] = team_squad["position"].map(position_priority).fillna(0.0)
    team_squad["age_bonus"] = team_squad["age_on_2026_06_11"].map(lambda age: _age_balance_score(float(age)) * 0.08)
    team_squad["key_player_score"] = (
        team_squad["club_strength_score"] * 0.49
        + team_squad["is_top_league"] * 0.10
        + team_squad["curated_bonus"]
        + team_squad["position_bonus"]
        + team_squad["age_bonus"]
    )
    # Sort curated players first (guaranteed appearance), then by score within each tier.
    selected = team_squad.sort_values(
        ["curated_bonus", "key_player_score", "shirt_number"],
        ascending=[False, False, True],
    ).head(effective_limit)

    players = []
    for _, row in selected.iterrows():
        curated_info = curated.get(_name_key(row["player_name"]), {})
        players.append(
            {
                "name": curated_info.get("name") or _title_name(row["player_name"]),
                "team": row["team"],
                "shirt_number": int(row["shirt_number"]),
                "position": row["position"],
                "club": _clean_club(row["club"]),
                "age": int(row["age_on_2026_06_11"]),
                "height_cm": int(row["height_cm"]),
                "club_country_code": row["club_country_code"],
                "impact_score": float(row["key_player_score"]),
                "source": "Official 2026 squad",
                "achievement": curated_info.get("achievement", ""),
                "goals": 0,
                "assists": 0,
                "appearances": 0,
                "goal_rate": 0.0,
                "awards": [],
            }
        )
    return players


def adjust_probabilities_for_2026_context(
    team_a: str,
    team_b: str,
    win_prob: float,
    draw_prob: float,
    loss_prob: float,
    world_cup_2026_data: dict | None,
) -> tuple[float, float, float, dict]:
    profile_a = get_team_profile(team_a, world_cup_2026_data)
    profile_b = get_team_profile(team_b, world_cup_2026_data)
    context = {
        "applied": False,
        "team_a_profile": profile_a,
        "team_b_profile": profile_b,
        "strength_delta": 0.0,
    }
    if not profile_a or not profile_b:
        return win_prob, draw_prob, loss_prob, context

    squad_a = profile_a.get("official_squad") or {}
    squad_b = profile_b.get("official_squad") or {}
    delta = float(profile_a["strength"] - profile_b["strength"])
    depth_delta = float(squad_a.get("depth_score", 0.45) - squad_b.get("depth_score", 0.45))
    age_delta = float(squad_a.get("age_balance", 0.5) - squad_b.get("age_balance", 0.5))
    top_league_delta = float(squad_a.get("top_league_share", 0.0) - squad_b.get("top_league_share", 0.0))
    combined_delta = (delta * 0.72) + (depth_delta * 0.18) + (age_delta * 0.05) + (top_league_delta * 0.05)
    squad_win_shift = float(np.clip(combined_delta * 0.11, -0.085, 0.085))

    current_a = profile_a.get("current_tournament") or {}
    current_b = profile_b.get("current_tournament") or {}
    played_a = float(current_a.get("played", 0.0) or 0.0)
    played_b = float(current_b.get("played", 0.0) or 0.0)
    current_score_delta = float(
        current_a.get("performance_score", 0.5) - current_b.get("performance_score", 0.5)
    )
    current_sample_weight = min(max(played_a, played_b) / 3.0, 1.0)
    current_win_shift = float(np.clip(current_score_delta * 0.10 * current_sample_weight, -0.055, 0.055))
    win_shift = float(np.clip(squad_win_shift + current_win_shift, -0.12, 0.12))
    strength_closeness = 1.0 - min(abs(delta) / 0.45, 1.0)
    current_closeness = 1.0 - min(abs(current_score_delta) / 0.45, 1.0)
    closeness = (strength_closeness * 0.75) + (current_closeness * 0.25 if current_sample_weight else 0.0)
    if current_sample_weight == 0:
        closeness = strength_closeness
    draw_shift = ((closeness - 0.5) * 0.025) - (abs(win_shift) * 0.20)

    adjusted = np.array(
        [
            loss_prob - win_shift,
            draw_prob + draw_shift,
            win_prob + win_shift,
        ],
        dtype=np.float64,
    )
    adjusted = np.clip(adjusted, 1e-6, 1.0)
    adjusted = adjusted / adjusted.sum()
    context.update(
        {
            "applied": True,
            "strength_delta": round(delta, 4),
            "squad_depth_delta": round(depth_delta, 4),
            "squad_top_league_delta": round(top_league_delta, 4),
            "combined_squad_parameter_delta": round(combined_delta, 4),
            "squad_probability_shift": round(squad_win_shift, 4),
            "current_tournament_applied": bool(played_a > 0 or played_b > 0),
            "team_a_current_tournament": current_a,
            "team_b_current_tournament": current_b,
            "current_tournament_score_delta": round(current_score_delta, 4),
            "current_tournament_sample_weight": round(current_sample_weight, 4),
            "current_tournament_probability_shift": round(current_win_shift, 4),
            "win_probability_shift": round(win_shift, 4),
            "draw_probability_shift": round(float(draw_shift), 4),
            "probability_changed": bool(abs(win_shift) > 0.0001 or abs(draw_shift) > 0.0001),
        }
    )
    return float(adjusted[2]), float(adjusted[1]), float(adjusted[0]), context
