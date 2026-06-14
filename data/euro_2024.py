"""EURO 2024 player strength and impact features."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import streamlit as st

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXTERNAL = os.path.join(_ROOT, "data", "external", "euro2024")

EURO_AGGREGATE_AVAILABLE_FROM = pd.Timestamp("2024-07-15")

FILES = {
    "standard": "player_standard_stats.csv",
    "shooting": "player_shooting.csv",
    "possession": "player_possession.csv",
    "playing_time": "player_playing_time.csv",
    "passing": "player_passing.csv",
    "pass_type": "player_pass_type.csv",
    "misc": "player_miscellaneous_stats.csv",
    "goalkeeping": "player_goalkeeping.csv",
    "advanced_goalkeeping": "player_advanced_goalkeeping.csv",
    "goal_creation": "player_goal_and_shot_creation.csv",
    "defense": "player_defense_actions.csv",
    "match_stats": "euro_matches.csv",
    "goals_leaders": "euro_goals.csv",
    "assists_leaders": "euro_assists.csv",
}

TEAM_REPLACEMENTS = {
    "Czechia": "Czech Republic",
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
}

FEATURE_DEFAULTS = {
    "euro_squad_impact": 0.0,
    "euro_top5_impact": 0.0,
    "euro_attack_impact": 0.0,
    "euro_midfield_impact": 0.0,
    "euro_defense_impact": 0.0,
    "euro_goalkeeping_impact": 0.0,
    "euro_minutes_coverage": 0.0,
    "euro_team_ppg": 0.0,
    "euro_team_goal_diff_per_match": 0.0,
    "euro_team_goals_per_match": 0.0,
    "euro_team_conceded_per_match": 0.0,
    "euro_team_shots_per_match": 0.0,
    "euro_team_shots_on_target_rate": 0.0,
    "euro_team_pass_accuracy": 0.0,
    "euro_team_defensive_actions_per_match": 0.0,
    "euro_team_set_piece_pressure": 0.0,
    "euro_team_discipline_risk": 0.0,
    "euro_leader_goals_per_match": 0.0,
    "euro_leader_assists_per_match": 0.0,
    "euro_leader_goal_contrib_per_match": 0.0,
}


def normalize_team_name(team: str) -> str:
    if pd.isna(team):
        return ""
    name = str(team).strip()
    parts = name.split(maxsplit=1)
    if len(parts) == 2 and parts[0].islower() and len(parts[0]) <= 3:
        name = parts[1]
    return TEAM_REPLACEMENTS.get(name, name)


def _read_csv(filename: str) -> pd.DataFrame | None:
    path = os.path.join(_EXTERNAL, filename)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def _base_key(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["player"] = out["Player"].astype(str).str.strip()
    out["team"] = out["Squad"].map(normalize_team_name)
    out["position"] = out.get("Pos", "").astype(str).str.split(",", n=1).str[0].str.strip()
    return out


def _select(df: pd.DataFrame | None, cols: dict[str, str]) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None
    out = _base_key(df)
    keep = ["player", "team", "position"]
    rename = {}
    for source, target in cols.items():
        if source in out.columns:
            keep.append(source)
            rename[source] = target
    return out[keep].rename(columns=rename)


def _merge(base: pd.DataFrame, extra: pd.DataFrame | None) -> pd.DataFrame:
    if extra is None or extra.empty:
        return base
    return base.merge(extra, on=["player", "team", "position"], how="left")


def _numeric(df: pd.DataFrame, cols: list[str]) -> None:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _num_series(series: pd.Series) -> pd.Series:
    values = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.extract(r"([-+]?\d*\.?\d+)", expand=False)
    )
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _pct(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if not higher_is_better:
        values = -values
    if values.nunique(dropna=True) <= 1:
        return pd.Series(0.5, index=values.index)
    return values.rank(pct=True).fillna(0.5)


def _mean_cols(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    existing = [col for col in cols if col in df.columns]
    if not existing:
        return pd.Series(0.0, index=df.index)
    return df[existing].mean(axis=1)


def _prepare_players(tables: dict[str, pd.DataFrame | None]) -> pd.DataFrame:
    standard = _select(
        tables.get("standard"),
        {
            "MP": "apps",
            "Starts": "starts",
            "Min": "minutes",
            "90s": "nineties",
            "Gls": "goals",
            "Ast": "assists",
            "xG": "xg",
            "xAG": "xag",
            "PrgC": "prog_carries",
            "PrgP": "prog_passes",
            "PrgR": "prog_receives",
        },
    )
    if standard is None:
        return pd.DataFrame()

    players = standard
    players = _merge(
        players,
        _select(
            tables.get("shooting"),
            {
                "Sh/90": "shots_per90",
                "SoT/90": "sot_per90",
                "G-xG": "goals_minus_xg",
            },
        ),
    )
    players = _merge(
        players,
        _select(
            tables.get("passing"),
            {
                "KP": "key_passes",
                "1/3": "passes_final_third",
                "PPA": "passes_penalty_area",
                "xA": "xa",
            },
        ),
    )
    players = _merge(
        players,
        _select(
            tables.get("possession"),
            {
                "Touches": "touches",
                "Att Pen": "att_pen_touches",
                "Carries": "carries",
                "Mis": "miscontrols",
                "Dis": "dispossessed",
            },
        ),
    )
    players = _merge(
        players,
        _select(
            tables.get("goal_creation"),
            {
                "SCA90": "sca_per90",
                "GCA90": "gca_per90",
            },
        ),
    )
    players = _merge(
        players,
        _select(
            tables.get("defense"),
            {
                "Tkl+Int": "tackles_interceptions",
                "Blocks": "blocks",
                "Clr": "clearances",
                "Err": "errors",
            },
        ),
    )
    players = _merge(
        players,
        _select(
            tables.get("misc"),
            {
                "CrdY": "yellow_cards",
                "CrdR": "red_cards",
                "Recov": "recoveries",
                "Won%": "aerial_win_pct",
            },
        ),
    )
    players = _merge(
        players,
        _select(
            tables.get("goalkeeping"),
            {
                "Save%": "save_pct",
                "CS%": "clean_sheet_pct",
                "GA90": "goals_against_per90",
            },
        ),
    )
    players = _merge(
        players,
        _select(
            tables.get("advanced_goalkeeping"),
            {
                "PSxG+/-": "psxg_plus_minus",
                "/90": "psxg_plus_minus_per90",
                "#OPA/90": "keeper_sweeps_per90",
            },
        ),
    )
    players = _merge(
        players,
        _select(
            tables.get("playing_time"),
            {
                "PPM": "points_per_match_on_pitch",
                "On-Off": "goal_on_off",
                "On-Off.1": "xg_on_off",
            },
        ),
    )

    numeric_cols = [col for col in players.columns if col not in {"player", "team", "position"}]
    _numeric(players, numeric_cols)
    players["nineties"] = players["nineties"].clip(lower=0.1)

    for total_col in [
        "goals",
        "assists",
        "xg",
        "xag",
        "prog_carries",
        "prog_passes",
        "prog_receives",
        "key_passes",
        "passes_final_third",
        "passes_penalty_area",
        "xa",
        "touches",
        "att_pen_touches",
        "carries",
        "tackles_interceptions",
        "blocks",
        "clearances",
        "recoveries",
    ]:
        if total_col in players.columns:
            players[f"{total_col}_per90"] = players[total_col] / players["nineties"]

    players["attack_score"] = _mean_cols(
        pd.DataFrame(
            {
                "goals": _pct(players.get("goals_per90", 0.0)),
                "assists": _pct(players.get("assists_per90", 0.0)),
                "xg": _pct(players.get("xg_per90", 0.0)),
                "xag": _pct(players.get("xag_per90", 0.0)),
                "sot": _pct(players.get("sot_per90", 0.0)),
                "sca": _pct(players.get("sca_per90", 0.0)),
                "gca": _pct(players.get("gca_per90", 0.0)),
            }
        ),
        ["goals", "assists", "xg", "xag", "sot", "sca", "gca"],
    )
    players["creative_score"] = _mean_cols(
        pd.DataFrame(
            {
                "key_passes": _pct(players.get("key_passes_per90", 0.0)),
                "ppa": _pct(players.get("passes_penalty_area_per90", 0.0)),
                "final_third": _pct(players.get("passes_final_third_per90", 0.0)),
                "prog_passes": _pct(players.get("prog_passes_per90", 0.0)),
                "xa": _pct(players.get("xa_per90", 0.0)),
                "sca": _pct(players.get("sca_per90", 0.0)),
            }
        ),
        ["key_passes", "ppa", "final_third", "prog_passes", "xa", "sca"],
    )
    players["possession_score"] = _mean_cols(
        pd.DataFrame(
            {
                "touches": _pct(players.get("touches_per90", 0.0)),
                "carries": _pct(players.get("carries_per90", 0.0)),
                "prog_carries": _pct(players.get("prog_carries_per90", 0.0)),
                "prog_receives": _pct(players.get("prog_receives_per90", 0.0)),
                "security": _pct(
                    players.get("miscontrols", 0.0) + players.get("dispossessed", 0.0),
                    higher_is_better=False,
                ),
            }
        ),
        ["touches", "carries", "prog_carries", "prog_receives", "security"],
    )
    players["defense_score"] = _mean_cols(
        pd.DataFrame(
            {
                "actions": _pct(players.get("tackles_interceptions_per90", 0.0)),
                "blocks": _pct(players.get("blocks_per90", 0.0)),
                "clearances": _pct(players.get("clearances_per90", 0.0)),
                "recoveries": _pct(players.get("recoveries_per90", 0.0)),
                "aerial": _pct(players.get("aerial_win_pct", 0.0)),
                "errors": _pct(players.get("errors", 0.0), higher_is_better=False),
            }
        ),
        ["actions", "blocks", "clearances", "recoveries", "aerial", "errors"],
    )
    players["goalkeeping_score"] = _mean_cols(
        pd.DataFrame(
            {
                "save": _pct(players.get("save_pct", 0.0)),
                "clean_sheet": _pct(players.get("clean_sheet_pct", 0.0)),
                "psxg": _pct(players.get("psxg_plus_minus_per90", 0.0)),
                "sweeper": _pct(players.get("keeper_sweeps_per90", 0.0)),
                "goals_against": _pct(players.get("goals_against_per90", 0.0), higher_is_better=False),
            }
        ),
        ["save", "clean_sheet", "psxg", "sweeper", "goals_against"],
    )

    pos = players["position"].str.upper()
    players["impact_score"] = (
        np.where(
            pos.eq("GK"),
            (players["goalkeeping_score"] * 0.75) + (players["possession_score"] * 0.25),
            np.where(
                pos.str.startswith("DF"),
                (players["defense_score"] * 0.45)
                + (players["possession_score"] * 0.25)
                + (players["creative_score"] * 0.20)
                + (players["attack_score"] * 0.10),
                np.where(
                    pos.str.startswith("MF"),
                    (players["creative_score"] * 0.35)
                    + (players["possession_score"] * 0.25)
                    + (players["defense_score"] * 0.20)
                    + (players["attack_score"] * 0.20),
                    (players["attack_score"] * 0.55)
                    + (players["creative_score"] * 0.25)
                    + (players["possession_score"] * 0.15)
                    + (players["defense_score"] * 0.05),
                ),
            ),
        )
        * np.sqrt(np.clip(players["minutes"] / 270.0, 0.15, 1.0))
    )
    return players.sort_values(["team", "impact_score"], ascending=[True, False]).reset_index(drop=True)


def _side_rows(matches: pd.DataFrame, side: int) -> pd.DataFrame:
    opponent = 2 if side == 1 else 1
    fouls_col = f"team{side}_fouls" if side == 1 else "team2_fouls_committed"
    yellow_col = f"team{side}_yellowcards" if side == 1 else "team2_yellow_cards"
    return pd.DataFrame(
        {
            "team": matches[f"team{side}"].map(normalize_team_name),
            "opponent": matches[f"team{opponent}"].map(normalize_team_name),
            "goals_for": _num_series(matches[f"team{side}_goals"]),
            "goals_against": _num_series(matches[f"team{opponent}_goals"]),
            "shots": _num_series(matches[f"team{side}_total_shots"]),
            "shots_on_target": _num_series(matches[f"team{side}_shots_on_target"]),
            "fouls": _num_series(matches.get(fouls_col, pd.Series(0, index=matches.index))),
            "corners": _num_series(matches[f"team{side}_corners"]),
            "blocked_shots": _num_series(matches[f"team{side}_blocked_shots"]),
            "passes": _num_series(matches[f"team{side}_passes"]),
            "accurate_passes": _num_series(matches[f"team{side}_accurate_passes"]),
            "accurate_crosses": _num_series(matches[f"team{side}_accurate_crosses"]),
            "yellow_cards": _num_series(matches.get(yellow_col, pd.Series(0, index=matches.index))),
            "red_cards": _num_series(matches[f"team{side}_red_cards"]),
            "tackles_won": _num_series(matches[f"team{side}_tackles_won"]),
            "interceptions": _num_series(matches[f"team{side}_interceptions"]),
            "blocks": _num_series(matches[f"team{side}_blocks"]),
            "clearances": _num_series(matches[f"team{side}_clearances"]),
            "keeper_saves": _num_series(matches[f"team{side}_keeper_saves"]),
            "duels_won": _num_series(matches[f"team{side}_duels_won"]),
            "successful_dribbles": _num_series(matches[f"team{side}_successful_dribbles"]),
        }
    )


def _prepare_match_team_stats(matches: pd.DataFrame | None) -> pd.DataFrame:
    if matches is None or matches.empty:
        return pd.DataFrame()
    required = {"team1", "team2", "team1_goals", "team2_goals", "team1_total_shots", "team2_total_shots"}
    if not required.issubset(matches.columns):
        return pd.DataFrame()

    side_rows = pd.concat([_side_rows(matches, 1), _side_rows(matches, 2)], ignore_index=True)
    side_rows["points"] = np.select(
        [
            side_rows["goals_for"] > side_rows["goals_against"],
            side_rows["goals_for"] == side_rows["goals_against"],
        ],
        [3.0, 1.0],
        default=0.0,
    )
    side_rows["goal_diff"] = side_rows["goals_for"] - side_rows["goals_against"]
    side_rows["defensive_actions"] = (
        side_rows["tackles_won"]
        + side_rows["interceptions"]
        + side_rows["blocks"]
        + side_rows["clearances"]
        + side_rows["keeper_saves"]
    )
    side_rows["set_piece_pressure"] = side_rows["corners"] + side_rows["accurate_crosses"]
    side_rows["discipline_risk"] = side_rows["yellow_cards"] + (side_rows["red_cards"] * 2.0)

    grouped = side_rows.groupby("team", as_index=False).agg(
        matches=("team", "size"),
        points=("points", "sum"),
        goals_for=("goals_for", "sum"),
        goals_against=("goals_against", "sum"),
        goal_diff=("goal_diff", "sum"),
        shots=("shots", "sum"),
        shots_on_target=("shots_on_target", "sum"),
        passes=("passes", "sum"),
        accurate_passes=("accurate_passes", "sum"),
        defensive_actions=("defensive_actions", "sum"),
        set_piece_pressure=("set_piece_pressure", "sum"),
        discipline_risk=("discipline_risk", "sum"),
    )
    grouped["ppg"] = grouped["points"] / grouped["matches"].clip(lower=1)
    grouped["goal_diff_per_match"] = grouped["goal_diff"] / grouped["matches"].clip(lower=1)
    grouped["goals_per_match"] = grouped["goals_for"] / grouped["matches"].clip(lower=1)
    grouped["conceded_per_match"] = grouped["goals_against"] / grouped["matches"].clip(lower=1)
    grouped["shots_per_match"] = grouped["shots"] / grouped["matches"].clip(lower=1)
    grouped["shots_on_target_rate"] = grouped["shots_on_target"] / grouped["shots"].clip(lower=1)
    grouped["pass_accuracy"] = grouped["accurate_passes"] / grouped["passes"].clip(lower=1)
    grouped["defensive_actions_per_match"] = grouped["defensive_actions"] / grouped["matches"].clip(lower=1)
    grouped["set_piece_pressure"] = grouped["set_piece_pressure"] / grouped["matches"].clip(lower=1)
    grouped["discipline_risk"] = grouped["discipline_risk"] / grouped["matches"].clip(lower=1)
    return grouped


def _prepare_goal_assist_team_stats(goals: pd.DataFrame | None, assists: pd.DataFrame | None) -> pd.DataFrame:
    pieces = []
    if goals is not None and not goals.empty and {"Team ", "Goals"}.issubset(goals.columns):
        g = goals.copy()
        g["team"] = g["Team "].map(normalize_team_name)
        g["goals"] = _num_series(g["Goals"])
        pieces.append(g.groupby("team", as_index=False)["goals"].sum())
    if assists is not None and not assists.empty and {"Team", "Assists"}.issubset(assists.columns):
        a = assists.copy()
        a["team"] = a["Team"].map(normalize_team_name)
        a["assists"] = _num_series(a["Assists"])
        pieces.append(a.groupby("team", as_index=False)["assists"].sum())
    if not pieces:
        return pd.DataFrame()

    out = pieces[0]
    for piece in pieces[1:]:
        out = out.merge(piece, on="team", how="outer")
    out = out.fillna(0.0)
    out["goal_contrib"] = out.get("goals", 0.0) + out.get("assists", 0.0)
    return out


@st.cache_data(show_spinner=False)
def load_euro_2024_data() -> dict:
    tables = {name: _read_csv(filename) for name, filename in FILES.items()}
    players = _prepare_players(tables)
    match_team_stats = _prepare_match_team_stats(tables.get("match_stats"))
    goal_assist_team_stats = _prepare_goal_assist_team_stats(
        tables.get("goals_leaders"),
        tables.get("assists_leaders"),
    )
    return {
        "players": players,
        "match_team_stats": match_team_stats,
        "goal_assist_team_stats": goal_assist_team_stats,
        "tables_loaded": sum(df is not None for df in tables.values()),
    }


def _team_players(team: str, euro_data: dict | None) -> pd.DataFrame:
    if not euro_data:
        return pd.DataFrame()
    players = euro_data.get("players")
    if players is None or players.empty:
        return pd.DataFrame()
    return players[players["team"] == normalize_team_name(team)].copy()


def _team_summary(team: str, euro_data: dict | None) -> dict:
    defaults = FEATURE_DEFAULTS.copy()
    players = _team_players(team, euro_data)
    if not players.empty:
        regulars = players[players["minutes"] >= 90].copy()
        if regulars.empty:
            regulars = players.copy()
        weights = regulars["minutes"].clip(lower=1.0)

        def _weighted_mean(col: str, frame: pd.DataFrame = regulars) -> float:
            if frame.empty or col not in frame.columns:
                return 0.0
            local_weights = frame["minutes"].clip(lower=1.0)
            return float(np.average(frame[col].fillna(0.0), weights=local_weights))

        attackers = regulars[regulars["position"].str.upper().str.startswith(("FW", "AM", "LW", "RW"))]
        midfielders = regulars[regulars["position"].str.upper().str.startswith("MF")]
        defenders = regulars[regulars["position"].str.upper().str.startswith("DF")]
        keepers = regulars[regulars["position"].str.upper().eq("GK")]

        defaults["euro_squad_impact"] = float(np.average(regulars["impact_score"], weights=weights))
        defaults["euro_top5_impact"] = float(regulars.nlargest(min(5, len(regulars)), "impact_score")["impact_score"].mean())
        defaults["euro_attack_impact"] = _weighted_mean("attack_score", attackers if not attackers.empty else regulars)
        defaults["euro_midfield_impact"] = _weighted_mean("creative_score", midfielders if not midfielders.empty else regulars)
        defaults["euro_defense_impact"] = _weighted_mean("defense_score", defenders if not defenders.empty else regulars)
        defaults["euro_goalkeeping_impact"] = _weighted_mean("goalkeeping_score", keepers if not keepers.empty else regulars)
        defaults["euro_minutes_coverage"] = float(min(players["minutes"].sum() / 4500.0, 1.0))

    team_name = normalize_team_name(team)
    match_stats = euro_data.get("match_team_stats") if euro_data else None
    if match_stats is not None and not match_stats.empty:
        row = match_stats[match_stats["team"] == team_name]
        if not row.empty:
            row = row.iloc[0]
            defaults["euro_team_ppg"] = float(row.get("ppg", 0.0))
            defaults["euro_team_goal_diff_per_match"] = float(row.get("goal_diff_per_match", 0.0))
            defaults["euro_team_goals_per_match"] = float(row.get("goals_per_match", 0.0))
            defaults["euro_team_conceded_per_match"] = float(row.get("conceded_per_match", 0.0))
            defaults["euro_team_shots_per_match"] = float(row.get("shots_per_match", 0.0))
            defaults["euro_team_shots_on_target_rate"] = float(row.get("shots_on_target_rate", 0.0))
            defaults["euro_team_pass_accuracy"] = float(row.get("pass_accuracy", 0.0))
            defaults["euro_team_defensive_actions_per_match"] = float(row.get("defensive_actions_per_match", 0.0))
            defaults["euro_team_set_piece_pressure"] = float(row.get("set_piece_pressure", 0.0))
            defaults["euro_team_discipline_risk"] = float(row.get("discipline_risk", 0.0))

    goal_assist = euro_data.get("goal_assist_team_stats") if euro_data else None
    if goal_assist is not None and not goal_assist.empty:
        row = goal_assist[goal_assist["team"] == team_name]
        if not row.empty:
            row = row.iloc[0]
            matches = 1.0
            if match_stats is not None and not match_stats.empty:
                match_row = match_stats[match_stats["team"] == team_name]
                if not match_row.empty:
                    matches = max(float(match_row.iloc[0].get("matches", 1.0)), 1.0)
            goals = float(row.get("goals", 0.0))
            assists = float(row.get("assists", 0.0))
            defaults["euro_leader_goals_per_match"] = goals / matches
            defaults["euro_leader_assists_per_match"] = assists / matches
            defaults["euro_leader_goal_contrib_per_match"] = float(row.get("goal_contrib", goals + assists)) / matches
    return defaults


def build_euro_feature_values(team_a: str, team_b: str, as_of_date, euro_data: dict | None) -> dict:
    if pd.Timestamp(as_of_date) < EURO_AGGREGATE_AVAILABLE_FROM:
        return {
            f"{prefix}_{feature}": 0.0
            for prefix in ("team_a", "team_b")
            for feature in FEATURE_DEFAULTS
        }

    values = {}
    for prefix, team in [("team_a", team_a), ("team_b", team_b)]:
        summary = _team_summary(team, euro_data)
        values.update({f"{prefix}_{key}": value for key, value in summary.items()})
    return values


def get_euro_key_players(team: str, euro_data: dict | None, n: int = 5) -> list[dict]:
    players = _team_players(team, euro_data)
    if players.empty:
        return []

    def _best_trait(row: pd.Series) -> str:
        traits = {
            "attack": float(row.get("attack_score", 0.0)),
            "creation": float(row.get("creative_score", 0.0)),
            "defending": float(row.get("defense_score", 0.0)),
            "goalkeeping": float(row.get("goalkeeping_score", 0.0)),
        }
        return max(traits, key=traits.get)

    result = []
    for _, row in players.sort_values("impact_score", ascending=False).head(n).iterrows():
        apps = int(row.get("apps", 0))
        goals = int(row.get("goals", 0))
        assists = int(row.get("assists", 0))
        impact = round(float(row.get("impact_score", 0.0)), 3)
        trait = _best_trait(row)
        result.append(
            {
                "name": row["player"],
                "goals": goals,
                "assists": assists,
                "appearances": apps,
                "goal_rate": round(goals / apps, 3) if apps else 0.0,
                "position": row.get("position", ""),
                "awards": [],
                "impact_score": impact,
                "attack_score": round(float(row.get("attack_score", 0.0)), 3),
                "creative_score": round(float(row.get("creative_score", 0.0)), 3),
                "defense_score": round(float(row.get("defense_score", 0.0)), 3),
                "goalkeeping_score": round(float(row.get("goalkeeping_score", 0.0)), 3),
                "euro_summary": (
                    f"EURO 2024: {goals}G, {assists}A in {apps} apps; "
                    f"impact {impact:.2f}, strongest in {trait}."
                ),
                "source": "EURO 2024",
            }
        )
    return result
