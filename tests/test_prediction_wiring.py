import unittest

import numpy as np
import pandas as pd

from model.goals import _scoreline_from_expected
from model.predict import (
    _blend_outcome_probabilities,
    _build_key_factors,
    _label_from_probs,
    _opposite_venue_mode,
    _predict_proba,
    _prediction_label,
    _resolve_prediction_label,
)


class _FixedBinaryModel:
    classes_ = np.array([0, 1])

    def __init__(self, positive_probability: float):
        self.positive_probability = positive_probability

    def predict_proba(self, rows):
        positive = np.full(len(rows), self.positive_probability, dtype=np.float64)
        return np.column_stack([1.0 - positive, positive])


class PredictionWiringTests(unittest.TestCase):
    def test_two_stage_probabilities_are_composed_into_three_classes(self):
        bundle = {
            "result_model_type": "two_stage",
            "draw_model": _FixedBinaryModel(0.20),
            "decisive_model": _FixedBinaryModel(0.75),
            "feature_columns": ["signal"],
        }

        probabilities = _predict_proba(bundle, pd.DataFrame({"signal": [1.0]}))

        np.testing.assert_allclose(probabilities, [0.20, 0.20, 0.60])

    def test_two_stage_label_uses_calibrated_draw_threshold(self):
        bundle = {"result_model_type": "two_stage", "draw_threshold": 0.37}

        label, policy = _prediction_label(bundle, np.array([0.30, 0.38, 0.32]))

        self.assertEqual(label, 1)
        self.assertIn("37%", policy)

    def test_mirrored_venue_preserves_home_team(self):
        self.assertEqual(_opposite_venue_mode("team_a_home"), "team_b_home")
        self.assertEqual(_opposite_venue_mode("team_b_home"), "team_a_home")
        self.assertEqual(_opposite_venue_mode("neutral"), "neutral")

    def test_goal_model_exposes_aggregate_outcome_probabilities(self):
        scoreline = _scoreline_from_expected(0.87, 1.67)
        outcome = scoreline["goal_outcome_probabilities"]

        self.assertAlmostEqual(sum(outcome.values()), 1.0, places=9)
        self.assertGreater(outcome["loss_prob"], outcome["win_prob"])
        self.assertEqual(scoreline["scorelines_by_outcome"]["team_a_win"]["scoreline"], "1-0")
        self.assertEqual(scoreline["scorelines_by_outcome"]["draw"]["scoreline"], "1-1")
        self.assertEqual(scoreline["scorelines_by_outcome"]["team_b_win"]["scoreline"], "0-1")

    def test_blended_prediction_uses_max_ensemble_probability(self):
        classifier_probs = np.array([0.34, 0.24, 0.42])
        goal_prediction = _scoreline_from_expected(0.87, 1.67)
        label, policy, source, scoreline, blended_probs, goal_probs = _resolve_prediction_label(
            classifier_probs,
            goal_prediction,
            {"classifier_weight": 0.75, "goal_weight": 0.25},
            classifier_label=2,
            classifier_policy="classifier policy",
        )

        self.assertEqual(source, "calibrated_ensemble")
        self.assertEqual(label, _label_from_probs(blended_probs))
        self.assertEqual(label, 0)
        self.assertEqual(scoreline, "0-1")
        self.assertIn("75%", policy)
        self.assertIsNotNone(goal_probs)
        self.assertAlmostEqual(blended_probs.sum(), 1.0, places=9)

    def test_scoreline_is_constrained_to_ensemble_label(self):
        classifier_probs = np.array([0.15, 0.20, 0.65])
        goal_prediction = _scoreline_from_expected(0.87, 1.67)

        label, _, _, scoreline, _, _ = _resolve_prediction_label(
            classifier_probs,
            goal_prediction,
            {"classifier_weight": 0.95, "goal_weight": 0.05},
        )

        self.assertEqual(label, 2)
        self.assertEqual(scoreline, "1-0")
        self.assertEqual(goal_prediction["likely_team_a_goals"], 1)
        self.assertEqual(goal_prediction["likely_team_b_goals"], 0)

    def test_blend_probability_helper_falls_back_to_classifier(self):
        classifier_probs = np.array([0.30, 0.38, 0.32])
        blended, source = _blend_outcome_probabilities(classifier_probs, None, 0.75)

        np.testing.assert_allclose(blended, classifier_probs)
        self.assertEqual(source, "result_classifier")

    def test_final_prediction_falls_back_to_classifier_without_scoreline(self):
        label, policy, source, scoreline, blended_probs, goal_probs = _resolve_prediction_label(
            np.array([0.30, 0.38, 0.32]),
            {},
            {"classifier_weight": 0.75, "goal_weight": 0.25},
            classifier_label=1,
            classifier_policy="calibrated draw threshold",
        )

        self.assertEqual(label, 1)
        self.assertIn("highest displayed probability", policy)
        self.assertEqual(source, "result_classifier")
        self.assertIsNone(scoreline)
        self.assertIsNone(goal_probs)
        np.testing.assert_allclose(blended_probs, [0.30, 0.38, 0.32])

    def test_classifier_fallback_label_matches_displayed_probability(self):
        label, policy, source, _, blended_probs, _ = _resolve_prediction_label(
            np.array([0.44, 0.37, 0.19]),
            {},
            {"classifier_weight": 0.75, "goal_weight": 0.25},
            classifier_label=1,
            classifier_policy="calibrated draw threshold",
        )

        self.assertEqual(label, 0)
        self.assertEqual(label, _label_from_probs(blended_probs))
        self.assertEqual(source, "result_classifier")
        self.assertIn("classifier diagnostic policy", policy)

    def test_key_factors_cover_all_nine_data_categories(self):
        factors = _build_key_factors(
            "Morocco",
            "Brazil",
            {
                "h2h_total_meetings": 1,
                "h2h_team_a_wins": 1,
                "h2h_team_b_wins": 0,
                "h2h_draws": 0,
                "team_a_form_wins": 4,
                "team_a_form_draws": 3,
                "team_a_form_losses": 3,
                "team_b_form_wins": 6,
                "team_b_form_draws": 2,
                "team_b_form_losses": 2,
                "team_a_rank": 12,
                "team_b_rank": 5,
            },
            None,
            None,
        )

        self.assertEqual(len(factors), 9)
        self.assertIn("History and form", factors[0])
        self.assertIn("Tournament pedigree", factors[1])
        self.assertIn("FIFA rankings", factors[2])
        self.assertIn("Player impact", factors[8])


if __name__ == "__main__":
    unittest.main()
