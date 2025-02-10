import numpy as np
import pandas as pd
from statsmodels.miscmodels.ordinal_model import OrderedModel

from src.models.base import BaseMatchPredictor

CATEGORICAL_DTYPE = pd.CategoricalDtype(
    categories=["win", "draw", "loss"], ordered=True
)


class EloRating:
    def __init__(self, initial_rating=1500, k0=10, delta=1, home_advantage=60):
        self.rating = {}
        self.initial_rating = initial_rating
        self.k0 = k0
        self.delta = delta
        self.home_advantage = home_advantage

    def get_rating(self, team):
        if team not in self.rating:
            self.rating[team] = self.initial_rating
        return self.rating[team]

    def create_rating(self, home, away):
        if home not in self.rating:
            self.rating[home] = self.initial_rating
        if away not in self.rating:
            self.rating[away] = self.initial_rating

    def expected_score(self, home, away, neutral):
        self.create_rating(home, away)
        home_advantage = 0 if neutral else self.home_advantage
        return 1 / (
            1 + 10 ** ((self.rating[away] - (self.rating[home] + home_advantage)) / 400)
        )

    def update_ratings(self, home, away, home_score, away_score, neutral):
        alfa_home = (
            1 if home_score > away_score else 0.5 if home_score == away_score else 0
        )
        alfa_away = (
            0 if home_score > away_score else 0.5 if home_score == away_score else 1
        )
        home_expected = self.expected_score(home, away, neutral)
        away_expected = 1 - home_expected
        k_factor = self.k0 * (1 + abs(away_score - home_score)) ** self.delta
        self.rating[home] += k_factor * (alfa_home - home_expected)
        self.rating[away] += k_factor * (alfa_away - away_expected)


class ELOgPredictor(BaseMatchPredictor):
    def __init__(self, initial_rating=1500, k0=10, delta=1, home_advantage=60):
        self.elo = EloRating(initial_rating, k0, delta, home_advantage)

    def _prepare_ratings(self, X, y):
        df = pd.concat([X, y], axis=1)
        for index, row in df.iterrows():
            home_advantage = (
                self.elo.home_advantage if row["team_at_home"] == False else 0
            )
            df.loc[index, "team_rating"] = (
                self.elo.get_rating(row["team_id"]) + home_advantage
            )
            df.loc[index, "opponent_rating"] = self.elo.get_rating(row["opponent_id"])
            self.elo.update_ratings(
                row["team_id"],
                row["opponent_id"],
                row["team_score"],
                row["opponent_score"],
                True if row["team_at_home"] == 0 else False,
            )
        return df

    def fit(self, X, y):
        df = self._prepare_ratings(X, y)
        df, _ = self._reverse_matches(df, pd.DataFrame())
        df["target"] = df.apply(
            lambda x: "win"
            if x["team_score"] > x["opponent_score"]
            else "draw"
            if x["team_score"] == x["opponent_score"]
            else "loss",
            axis=1,
        ).astype(CATEGORICAL_DTYPE)
        df["rating_difference"] = df["team_rating"] - df["opponent_rating"]
        mod_log = OrderedModel(df["target"], df[["rating_difference"]], distr="logit")
        self.logit = mod_log.fit(method="bfgs", disp=False)

    def update(self, X, y):
        """Update a classifier from the a given set (X, y)."""
        df = pd.concat([X, y], axis=1)
        for _, row in df.iterrows():
            self.elo.update_ratings(
                row["team_id"],
                row["opponent_id"],
                row["team_score"],
                row["opponent_score"],
                True if row["team_at_home"] == 0 else False,
            )

    def predict(self, X):
        """Predict class probabilities of the input samples X."""
        df = X.copy()
        for index, row in df.iterrows():
            home_advantage = (
                self.elo.home_advantage if row["team_at_home"] == False else 0
            )
            df.loc[index, "team_rating"] = (
                self.elo.get_rating(row["team_id"]) + home_advantage
            )
            df.loc[index, "opponent_rating"] = self.elo.get_rating(row["opponent_id"])
        df["rating_difference"] = df["team_rating"] - df["opponent_rating"]
        return self.logit.model.predict(
            self.logit.params, exog=df[["rating_difference"]]
        )

    def predict_and_update(self, X):
        """Predict class probabilities and then update the classifier."""
        df = self._prepare_ratings(X, y)
        df["rating_difference"] = df["team_rating"] - df["opponent_rating"]
        return self.logit.model.predict(
            self.logit.params, exog=df[["rating_difference"]]
        )
