import math
import unittest
from unittest.mock import patch

from data.ingest import load_all
from model.goals import predict_goals
from model.predict import predict_match


MATCHUPS = [
    ("Argentina", "France"),
    ("Spain", "Brazil"),
    ("England", "Germany"),
    ("Morocco", "Japan"),
    ("Canada", "Mexico"),
    ("Algeria", "Australia"),
    ("New Zealand", "Portugal"),
    ("Norway", "Colombia"),
]


class PredictionMatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_all()
        cls.result_cache = {}

    def _predict(self, team_a, team_b, venue_mode="neutral"):
        cache_key = (team_a, team_b, venue_mode)
        if cache_key in self.result_cache:
            return self.result_cache[cache_key]
        with patch("model.goals.predict_goals", return_value={}):
            result = predict_match(
                team_a=team_a,
                team_b=team_b,
                matches_df=self.data["matches"],
                goalscorers_df=self.data["goalscorers"],
                shootouts_df=self.data["shootouts"],
                rankings_df=self.data["rankings"],
                substitutions_df=self.data.get("substitutions"),
                player_appearances_df=self.data.get("player_appearances"),
                player_goals_df=self.data.get("player_goals"),
                award_winners_df=self.data.get("award_winners"),
                copa_data=self.data.get("copa_america"),
                euro_data=self.data.get("euro_2024"),
                friendlies_data=self.data.get("international_friendlies"),
                world_cup_2026_data=self.data.get("world_cup_2026"),
                venue_mode=venue_mode,
            )
        self.result_cache[cache_key] = result
        return result

    def assert_valid_result(self, result):
        probabilities = [
            result["win_prob"],
            result["draw_prob"],
            result["loss_prob"],
        ]
        self.assertTrue(all(math.isfinite(value) for value in probabilities))
        self.assertTrue(all(0.0 <= value <= 1.0 for value in probabilities))
        self.assertAlmostEqual(sum(probabilities), 1.0, places=9)
        self.assertIn(result["predicted_label"], {0, 1, 2})
        self.assertTrue(result["verdict"])
        self.assertTrue(result["decision_policy"])
        outcome_probabilities = [
            result["loss_prob"],
            result["draw_prob"],
            result["win_prob"],
        ]
        self.assertEqual(
            result["predicted_label"],
            max(range(3), key=lambda idx: outcome_probabilities[idx]),
        )
        if "likely_team_a_goals" in result and "likely_team_b_goals" in result:
            goals_a = int(round(float(result["likely_team_a_goals"])))
            goals_b = int(round(float(result["likely_team_b_goals"])))
            score_label = 2 if goals_a > goals_b else 0 if goals_b > goals_a else 1
            self.assertEqual(result["predicted_label"], score_label)

        base = result["base_probabilities"]
        self.assertAlmostEqual(
            base["win_prob"] + base["draw_prob"] + base["loss_prob"],
            1.0,
            places=3,
        )

        context = result["world_cup_2026_context"]
        if context.get("applied"):
            shift_total = sum(
                context[key]
                for key in (
                    "team_a_probability_shift",
                    "draw_probability_shift",
                    "team_b_probability_shift",
                )
            )
            self.assertAlmostEqual(shift_total, 0.0, places=3)
            self.assertAlmostEqual(
                context["team_a_squad_edge_shift"] + context["team_b_squad_edge_shift"],
                0.0,
                places=9,
            )

    def test_prediction_matrix_has_valid_outputs(self):
        for team_a, team_b in MATCHUPS:
            with self.subTest(team_a=team_a, team_b=team_b):
                self.assert_valid_result(self._predict(team_a, team_b))

    def test_france_squad_edge_is_visible_against_argentina(self):
        result = self._predict("Argentina", "France")
        context = result["world_cup_2026_context"]

        self.assertLess(context["team_a_squad_edge_shift"], 0.0)
        self.assertGreater(context["team_b_squad_edge_shift"], 0.0)

    def test_neutral_predictions_are_symmetric_when_teams_are_reversed(self):
        for team_a, team_b in MATCHUPS:
            with self.subTest(team_a=team_a, team_b=team_b):
                direct = self._predict(team_a, team_b)
                reverse = self._predict(team_b, team_a)
                self.assertAlmostEqual(direct["win_prob"], reverse["loss_prob"], places=9)
                self.assertAlmostEqual(direct["draw_prob"], reverse["draw_prob"], places=9)
                self.assertAlmostEqual(direct["loss_prob"], reverse["win_prob"], places=9)

    def test_home_context_is_symmetric_when_fixture_is_reversed(self):
        for team_a, team_b in MATCHUPS[:4]:
            with self.subTest(team_a=team_a, team_b=team_b):
                direct = self._predict(team_a, team_b, "team_a_home")
                reverse = self._predict(team_b, team_a, "team_b_home")
                self.assertAlmostEqual(direct["win_prob"], reverse["loss_prob"], places=9)
                self.assertAlmostEqual(direct["draw_prob"], reverse["draw_prob"], places=9)
                self.assertAlmostEqual(direct["loss_prob"], reverse["win_prob"], places=9)

    def test_goal_predictions_are_valid_and_symmetric(self):
        rankings = self.data["rankings"]
        for team_a, team_b in MATCHUPS[:3]:
            with self.subTest(team_a=team_a, team_b=team_b):
                direct = predict_goals(team_a, team_b, rankings, "neutral")
                reverse = predict_goals(team_b, team_a, rankings, "neutral")
                self.assertGreaterEqual(direct["expected_team_a_goals"], 0.0)
                self.assertGreaterEqual(direct["expected_team_b_goals"], 0.0)
                self.assertAlmostEqual(
                    direct["expected_total_goals"],
                    direct["expected_team_a_goals"] + direct["expected_team_b_goals"],
                    places=2,
                )
                self.assertAlmostEqual(
                    direct["expected_team_a_goals"],
                    reverse["expected_team_b_goals"],
                    places=2,
                )
                self.assertAlmostEqual(
                    direct["expected_team_b_goals"],
                    reverse["expected_team_a_goals"],
                    places=2,
                )
                self.assertAlmostEqual(
                    direct["over_2_5_probability"] + direct["under_2_5_probability"],
                    1.0,
                    places=9,
                )


if __name__ == "__main__":
    unittest.main()
