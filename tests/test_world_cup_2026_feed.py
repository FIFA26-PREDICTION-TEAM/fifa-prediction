import os
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

import data.world_cup_2026 as wc2026
import model.goals as goals


def _team_row(team: str) -> dict:
    return {
        "Group": "A",
        "Team": team,
        "Confederation": "UEFA",
        "FIFA_Ranking": 25,
        "Key_Player_1": f"{team} Forward",
        "KP1_Position": "Forward",
        "KP1_Club": "Test FC",
        "KP1_Notable_Achievement": "Leading scorer",
        "Key_Player_2": f"{team} Midfielder",
        "KP2_Position": "Midfielder",
        "KP2_Club": "Test FC",
        "KP2_Notable_Achievement": "Captain",
        "Team_Best_WC_Finish": "Group stage",
        "WC_Debut": "No",
        "Notes": "",
    }


class WorldCup2026FeedTests(unittest.TestCase):
    def test_match_feed_reader_falls_back_for_latin1_files(self):
        content = (
            b"date,team_a,team_b,team_a_score,team_b_score,referee\n"
            b"15-06-2026,Alpha,Beta,1,0,Fran\xe7ois Letexier\n"
        )
        handle = tempfile.NamedTemporaryFile(delete=False)
        try:
            handle.write(content)
            handle.close()

            df = wc2026._read_match_feed_csv(handle.name)

            self.assertEqual(df.loc[0, "referee"], "Fran\u00e7ois Letexier")
        finally:
            os.unlink(handle.name)

    def test_repo_match_feed_wins_over_downloads_when_no_env_override(self):
        download_path = wc2026._default_path("wc2026_matches_12thjune.csv")

        def exists(path):
            return path in {wc2026.WORLD_CUP_2026_MATCHES_REPO_PATH, download_path}

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ML_PRJCT_WORLD_CUP_2026_MATCHES_PATH", None)
            with patch("data.world_cup_2026.os.path.exists", side_effect=exists):
                self.assertEqual(
                    wc2026._resolve_world_cup_2026_matches_path(),
                    wc2026.WORLD_CUP_2026_MATCHES_REPO_PATH,
                )

    def test_env_override_still_controls_match_feed_path(self):
        custom_path = "/tmp/custom_wc2026.csv"

        with patch.dict(os.environ, {"ML_PRJCT_WORLD_CUP_2026_MATCHES_PATH": custom_path}):
            self.assertEqual(wc2026._resolve_world_cup_2026_matches_path(), custom_path)

    def test_current_scoring_impact_is_derived_from_match_stats(self):
        raw = pd.DataFrame(
            [
                {
                    "date": "15-06-2026",
                    "team_a": "Alpha",
                    "team_b": "Beta",
                    "team_a_score": 4,
                    "team_b_score": 0,
                    "team_a_shots_total": 10,
                    "team_b_shots_total": 2,
                    "team_a_shots_on_target": 6,
                    "team_b_shots_on_target": 0,
                    "team_a_assists": 3,
                    "team_b_assists": 0,
                    "team_a_penalties_scored": 1,
                    "team_b_penalties_scored": 0,
                    "team_a_goals_inside_box": 3,
                    "team_b_goals_inside_box": 0,
                    "team_a_goals_outside_box": 1,
                    "team_b_goals_outside_box": 0,
                }
            ]
        )
        stats = wc2026._derive_current_team_stats(wc2026._prepare_current_matches(raw))
        alpha = stats[stats["team"] == "Alpha"].iloc[0]
        beta = stats[stats["team"] == "Beta"].iloc[0]

        self.assertGreater(alpha["scoring_impact_score"], beta["scoring_impact_score"])
        self.assertGreaterEqual(alpha["scoring_impact_score"], 0.0)
        self.assertLessEqual(alpha["scoring_impact_score"], 1.0)
        self.assertEqual(alpha["open_play_goals"], 3)

    def test_scoring_impact_reaches_2026_probability_context(self):
        raw = pd.DataFrame(
            [
                {
                    "date": "15-06-2026",
                    "team_a": "Alpha",
                    "team_b": "Beta",
                    "team_a_score": 4,
                    "team_b_score": 0,
                    "team_a_shots_total": 10,
                    "team_b_shots_total": 2,
                    "team_a_shots_on_target": 6,
                    "team_b_shots_on_target": 0,
                    "team_a_assists": 3,
                    "team_b_assists": 0,
                    "team_a_penalties_scored": 1,
                    "team_b_penalties_scored": 0,
                }
            ]
        )
        current_stats = wc2026._derive_current_team_stats(wc2026._prepare_current_matches(raw))
        world_cup_data = {
            "teams": wc2026._prepare(pd.DataFrame([_team_row("Alpha"), _team_row("Beta")])),
            "squad_summary": pd.DataFrame(),
            "current_team_stats": current_stats,
        }

        _, _, _, context = wc2026.adjust_probabilities_for_2026_context(
            "Alpha",
            "Beta",
            0.33,
            0.34,
            0.33,
            world_cup_data,
        )

        self.assertTrue(context["applied"])
        self.assertTrue(context["current_tournament_applied"])
        self.assertGreater(context["current_scoring_impact_delta"], 0.0)
        self.assertGreater(context["current_tournament_score_delta"], 0.0)

    def test_goal_inference_matches_append_live_world_cup_feed(self):
        base = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2026-06-01"),
                    "home_team": "Base A",
                    "away_team": "Base B",
                    "home_score": 1,
                    "away_score": 1,
                    "tournament": "Friendly",
                    "country": "",
                    "neutral": True,
                }
            ]
        )
        appended = pd.concat(
            [
                base,
                pd.DataFrame(
                    [
                        {
                            "date": pd.Timestamp("2026-06-15"),
                            "home_team": "Live A",
                            "away_team": "Live B",
                            "home_score": 2,
                            "away_score": 0,
                            "tournament": "FIFA World Cup 2026",
                            "country": "",
                            "neutral": True,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

        with patch("model.goals.load_goal_matches", return_value=base):
            with patch("model.goals.append_world_cup_2026_matches", return_value=appended) as append:
                result = goals._load_goal_inference_matches()

        append.assert_called_once()
        self.assertIn("Live A", result["home_team"].tolist())


if __name__ == "__main__":
    unittest.main()
