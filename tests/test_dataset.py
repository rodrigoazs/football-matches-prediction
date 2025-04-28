from src.dataset import generate_features
import pandas as pd


def test_generate_features():
    input_df = pd.DataFrame(
        {
            "date": ["2023-05-31", "2023-06-02", "2023-06-04", "2023-06-06", "2023-06-08", "2023-06-10"],
            "team_id": [0, 1, 0, 2, 2, 1],
            "opponent_id": [1, 0, 2, 0, 1, 2],
            "team_at_home": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "opponent_at_home": [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            "team_score": [2.0, 3.0, 3.0, 3.0, 1.0, 0.0],
            "opponent_score": [3.0, 2.0, 3.0, 3.0, 0.0, 1.0],
        }
    )
    result_df = generate_features(input_df)