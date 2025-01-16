import numpy as np
import pandas as pd

from src.models.base import BaseMatchPredictor
from src.utils import determine_target, swap_dataset


class FrequencyRandomMatchPredictor(BaseMatchPredictor):
    def __init__(self):
        self._neutral_prob = np.ones(3) / 3
        self._home_prob = np.ones(3) / 3

    def fit(self, X: pd.DataFrame) -> None:
        df = X.copy()
        neutral_df = df[df["neutral"] == True]
        swaped_df = swap_dataset(neutral_df)
        neutral_df = pd.concat([neutral_df, swaped_df])
        home_neutral_df = df[df["neutral"] == False]
        home_neutral_df["target"] = home_neutral_df.apply(determine_target, axis=1)
        neutral_df["target"] = neutral_df.apply(determine_target, axis=1)
        if len(home_neutral_df):
            self._home_prob = (
                home_neutral_df["target"]
                .value_counts(normalize=True)
                .sort_index()
                .to_numpy()
            )
        if len(neutral_df):
            self._neutral_prob = (
                neutral_df["target"]
                .value_counts(normalize=True)
                .sort_index()
                .to_numpy()
            )

    def predict(self, X):
        prob = self.predict_proba(X)
        # return np.array([np.random.choice([0, 1, 2], p=probs) for probs in prob])
        return np.argmax(prob, axis=1)

    def predict_proba(self, X):
        return np.array(
            [
                self._neutral_prob if neutral else self._home_prob
                for neutral in X["neutral"]
            ]
        )

    def update_ratings(self, X):
        pass
