import numpy as np
import pandas as pd

from src.models.base import BaseMatchPredictor


def test_base_match_predictor_reverse_matches(mock_inputs, mock_targets):
    model = BaseMatchPredictor()
    df_X, df_y = model._reverse_matches(mock_inputs, mock_targets)
    expected_X = pd.DataFrame(
        [
            {
                "team_id": "team1",
                "opponent_id": "team2",
                "team_at_home": 1.0,
                "opponent_at_home": 0.0,
            },
            {
                "team_id": "team2",
                "opponent_id": "team1",
                "team_at_home": 0.0,
                "opponent_at_home": 1.0,
            },
            {
                "team_id": "team1",
                "opponent_id": "team3",
                "team_at_home": 0.0,
                "opponent_at_home": 0.0,
            },
            {
                "team_id": "team3",
                "opponent_id": "team1",
                "team_at_home": 0.0,
                "opponent_at_home": 0.0,
            },
            {
                "team_id": "team3",
                "opponent_id": "team2",
                "team_at_home": 0.0,
                "opponent_at_home": 1.0,
            },
            {
                "team_id": "team2",
                "opponent_id": "team3",
                "team_at_home": 1.0,
                "opponent_at_home": 0.0,
            },
        ]
    )
    expected_y = pd.DataFrame(
        [
            {
                "team_score": 2.0,
                "opponent_score": 3.0,
            },
            {
                "team_score": 3.0,
                "opponent_score": 2.0,
            },
            {
                "team_score": 3.0,
                "opponent_score": 3.0,
            },
            {
                "team_score": 3.0,
                "opponent_score": 3.0,
            },
            {
                "team_score": 1.0,
                "opponent_score": 0.0,
            },
            {
                "team_score": 0.0,
                "opponent_score": 1.0,
            },
        ]
    )
    pd.testing.assert_frame_equal(df_X, expected_X, check_like=True)
    pd.testing.assert_frame_equal(df_y, expected_y, check_like=True)
