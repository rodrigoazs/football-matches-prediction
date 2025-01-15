import numpy as np
import pandas as pd

from src.models.base import BaseMatchPredictor


def determine_target(row):
    if row["home_score"] > row["away_score"]:
        return 0
    elif row["home_score"] == row["away_score"]:
        return 1
    else:
        return 2


def swap_dataset(df):
    swaped_df = pd.DataFrame()
    swaped_df["date"] = df["date"]
    swaped_df["home_team"] = df["away_team"]
    swaped_df["home_score"] = df["away_score"]
    swaped_df["away_score"] = df["home_score"]
    swaped_df["away_team"] = df["home_team"]
    return swaped_df


class FrequencyRandomMatchPredictor(BaseMatchPredictor):
    def __init__(self):
        self._neutral_prob = None
        self._home_prob = None

    def fit(self, X: pd.DataFrame) -> None:
        df = X.copy()
        neutral_df = df[df["neutral"] == True]
        swaped_df = swap_dataset(neutral_df)
        neutral_df = pd.concat([neutral_df, swaped_df])
        home_neutral_df = df[df["neutral"] == False]
        home_neutral_df["target"] = home_neutral_df.apply(determine_target, axis=1)
        neutral_df["target"] = neutral_df.apply(determine_target, axis=1)
        self._home_prob = (
            home_neutral_df["target"]
            .value_counts(normalize=True)
            .sort_index()
            .to_numpy()
        )
        self._neutral_prob = (
            neutral_df["target"].value_counts(normalize=True).sort_index().to_numpy()
        )

    def predict(self, X):
        prob = self.predict_proba(X)
        return np.array([np.random.choice([0, 1, 2], p=probs) for probs in prob])

    def predict_proba(self, X):
        return np.array(
            [
                self._neutral_prob if neutral else self._home_prob
                for neutral in X["neutral"]
            ]
        )

    def update_ratings(self, X):
        pass
