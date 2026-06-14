"""Audit prediction probability, label, and scoreline consistency."""

from __future__ import annotations

import argparse
import itertools
import logging
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("STREAMLIT_LOG_LEVEL", "error")
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.caching.cache_data_api").disabled = True

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.ingest import get_all_teams, load_all
from model.predict import predict_match


LABEL_NAME = {
    0: "team_b_win",
    1: "draw",
    2: "team_a_win",
}


def _score_label(result: dict) -> int | None:
    goals_a = result.get("likely_team_a_goals")
    goals_b = result.get("likely_team_b_goals")
    if goals_a is None or goals_b is None:
        return None
    goals_a = int(round(float(goals_a)))
    goals_b = int(round(float(goals_b)))
    if goals_a > goals_b:
        return 2
    if goals_b > goals_a:
        return 0
    return 1


def _teams(data: dict, scope: str) -> list[str]:
    if scope == "2026":
        world_cup = data.get("world_cup_2026") or {}
        teams = world_cup.get("teams")
        if teams is not None and not teams.empty:
            available = set(get_all_teams(data["matches"]))
            return sorted(set(teams["Team"].dropna().astype(str)) & available)
    return get_all_teams(data["matches"])


def _check_result(team_a: str, team_b: str, venue_mode: str, result: dict) -> list[str]:
    issues = []
    probs = np.array([result["loss_prob"], result["draw_prob"], result["win_prob"]], dtype=np.float64)
    if not np.all(np.isfinite(probs)):
        issues.append("non-finite probability")
    if np.any(probs < -1e-9) or np.any(probs > 1.0 + 1e-9):
        issues.append("probability outside [0, 1]")
    if not np.isclose(probs.sum(), 1.0, atol=1e-6):
        issues.append(f"probabilities sum to {probs.sum():.8f}")
    expected_label = int(probs.argmax())
    if int(result["predicted_label"]) != expected_label:
        issues.append(
            f"label {LABEL_NAME.get(result['predicted_label'])} does not match max probability {LABEL_NAME[expected_label]}"
        )
    score_label = _score_label(result)
    if score_label is not None and int(result["predicted_label"]) != score_label:
        issues.append(
            f"scoreline {result.get('likely_team_a_goals')}-{result.get('likely_team_b_goals')} "
            f"does not match label {LABEL_NAME.get(result['predicted_label'])}"
        )
    if issues:
        return [f"{team_a} vs {team_b} [{venue_mode}]: {issue}" for issue in issues]
    return []


def audit(scope: str, venues: list[str], max_pairs: int | None = None) -> list[str]:
    data = load_all()
    teams = _teams(data, scope)
    pairs = list(itertools.permutations(teams, 2))
    if max_pairs is not None:
        pairs = pairs[:max_pairs]

    issues = []
    checked = 0
    for venue_mode in venues:
        for team_a, team_b in pairs:
            result = predict_match(
                team_a=team_a,
                team_b=team_b,
                matches_df=data["matches"],
                goalscorers_df=data["goalscorers"],
                shootouts_df=data["shootouts"],
                rankings_df=data["rankings"],
                substitutions_df=data.get("substitutions"),
                player_appearances_df=data.get("player_appearances"),
                player_goals_df=data.get("player_goals"),
                award_winners_df=data.get("award_winners"),
                copa_data=data.get("copa_america"),
                euro_data=data.get("euro_2024"),
                friendlies_data=data.get("international_friendlies"),
                world_cup_2026_data=data.get("world_cup_2026"),
                venue_mode=venue_mode,
            )
            checked += 1
            issues.extend(_check_result(team_a, team_b, venue_mode, result))

    print(f"Checked {checked} predictions across {len(teams)} teams and {len(venues)} venue mode(s).")
    if issues:
        print(f"Found {len(issues)} consistency issue(s):")
        for issue in issues[:50]:
            print(f"- {issue}")
        if len(issues) > 50:
            print(f"... {len(issues) - 50} more")
    else:
        print("No consistency issues found.")
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=["2026", "all"], default="2026")
    parser.add_argument("--all-venues", action="store_true", help="Audit Team A home, neutral, and Team B home.")
    parser.add_argument("--max-pairs", type=int, default=None, help="Optional cap for quick local smoke runs.")
    args = parser.parse_args()

    venues = ["neutral"]
    if args.all_venues:
        venues = ["team_a_home", "neutral", "team_b_home"]
    issues = audit(args.scope, venues, args.max_pairs)
    raise SystemExit(1 if issues else 0)


if __name__ == "__main__":
    main()
