import pandas as pd
import pytest


@pytest.fixture
def mock_inputs():
    df = pd.DataFrame(
        [
            {
                "team_id": "team1",
                "opponent_id": "team2",
                "team_at_home": 1.0,
                "opponent_at_home": 0.0,
            },
            {
                "team_id": "team1",
                "opponent_id": "team3",
                "team_at_home": 0.0,
                "opponent_at_home": 0.0,
            },
            {
                "team_id": "team3",
                "opponent_id": "team2",
                "team_at_home": 0.0,
                "opponent_at_home": 1.0,
            },
        ]
    )
    return df


@pytest.fixture
def mock_targets():
    df = pd.DataFrame(
        [
            {
                "team_score": 2.0,
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
        ]
    )
    return df
