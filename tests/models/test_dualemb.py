import pandas as pd

from src.models.dualemb import DualEmbPredictor


def test_dualebm_prepare_dataset():
    model = DualEmbPredictor()

    df = pd.DataFrame(
        [
            {
                "date": "2024-01-01",
                "home_team": "team1",
                "home_score": 2,
                "away_score": 3,
                "away_team": "team2",
                "neutral": False,
            },
            {
                "date": "2024-01-01",
                "home_team": "team1",
                "home_score": 2,
                "away_score": 1,
                "away_team": "team3",
                "neutral": True,
            },
        ]
    )
    result = model._prepare_dataset(df)
    raise Exception(result)
