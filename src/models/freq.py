import numpy as np
import pandas as pd

from src.models.base import BaseMatchPredictor


class FrequencyMatchPredictor(BaseMatchPredictor):
    def __init__(self):
        self.prob = {}

    def fit(self, X, y):
        df_X, df_y = self._reverse_matches(X, y)
        df = df_X[["team_at_home", "opponent_at_home"]]
        df["target"] = df_y.apply(
            lambda x: "win"
            if x["team_score"] > x["opponent_score"]
            else "draw"
            if x["team_score"] == x["opponent_score"]
            else "loss",
            axis=1,
        )
        grouped = (
            df.groupby(["team_at_home", "opponent_at_home"])["target"]
            .value_counts(normalize=True)
            .unstack(fill_value=0)
        )
        prob_dict = {
            key: [value["win"], value["draw"], value["loss"]]
            for key, value in grouped.to_dict(orient="index").items()
        }
        self.prob = prob_dict

    def update(self, X, y):
        """Update a classifier from the a given set (X, y)."""
        pass

    def predict(self, X):
        """Predict class probabilities of the input samples X."""
        return np.array(
            [
                self.prob.get(
                    (row["team_at_home"], row["opponent_at_home"]),
                    [1 / 3, 1 / 3, 1 / 3],
                )
                for _, row in X.iterrows()
            ]
        )

    def predict_and_update(self, X, y):
        """Predict class probabilities and then update the classifier."""
        return self.predict(X)
