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

    def create_rating(self, team, opponent):
        if team not in self.rating:
            self.rating[team] = self.initial_rating
        if opponent not in self.rating:
            self.rating[opponent] = self.initial_rating

    def expected_score(self, team, opponent, team_at_home, opponent_at_home):
        self.create_rating(team, opponent)
        team_home_advantage = self.home_advantage if team_at_home else 0
        opponent_home_advantage = self.home_advantage if opponent_at_home else 0
        return 1 / (
            1
            + 10
            ** (
                (
                    (self.rating[opponent] + opponent_home_advantage)
                    - (self.rating[team] + team_home_advantage)
                )
                / 400
            )
        )

    def update_ratings(
        self, team, opponent, team_score, opponent_score, team_at_home, opponent_at_home
    ):
        alfa_team = (
            1
            if team_score > opponent_score
            else 0.5
            if team_score == opponent_score
            else 0
        )
        alfa_opponent = (
            0
            if team_score > opponent_score
            else 0.5
            if team_score == opponent_score
            else 1
        )
        team_expected = self.expected_score(
            team, opponent, team_at_home, opponent_at_home
        )
        opponent_expected = 1 - team_expected
        k_factor = self.k0 * (1 + abs(opponent_score - team_score)) ** self.delta
        self.rating[team] += k_factor * (alfa_team - team_expected)
        self.rating[opponent] += k_factor * (alfa_opponent - opponent_expected)


class ELOgPredictor(BaseMatchPredictor):
    def __init__(self, initial_rating=1500, k0=10, delta=1, home_advantage=60):
        self.elo = EloRating(initial_rating, k0, delta, home_advantage)

    def _prepare_ratings(self, X, y):
        df = pd.concat([X, y], axis=1)
        for index, row in df.iterrows():
            team_home_advantage = (
                self.elo.home_advantage if row["team_at_home"] == 1.0 else 0
            )
            df.loc[index, "team_rating"] = (
                self.elo.get_rating(row["team_id"]) + team_home_advantage
            )
            opponent_home_advantage = (
                self.elo.home_advantage if row["opponent_at_home"] == 1.0 else 0
            )
            df.loc[index, "opponent_rating"] = (
                self.elo.get_rating(row["opponent_id"]) + opponent_home_advantage
            )
            self.elo.update_ratings(
                row["team_id"],
                row["opponent_id"],
                row["team_score"],
                row["opponent_score"],
                True if row["team_at_home"] == 1.0 else False,
                True if row["opponent_at_home"] == 1.0 else False,
            )
        return df

    def fit(self, X, y):
        df = self._prepare_ratings(X, y)
        self.elo.rating = {}  # Clear ratings
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
                True if row["team_at_home"] == 1.0 else False,
                True if row["opponent_at_home"] == 1.0 else False,
            )

    def predict(self, X):
        """Predict class probabilities of the input samples X."""
        df = X.copy()
        for index, row in df.iterrows():
            team_home_advantage = (
                self.elo.home_advantage if row["team_at_home"] == 1.0 else 0
            )
            df.loc[index, "team_rating"] = (
                self.elo.get_rating(row["team_id"]) + team_home_advantage
            )
            opponent_home_advantage = (
                self.elo.home_advantage if row["opponent_at_home"] == 1.0 else 0
            )
            df.loc[index, "opponent_rating"] = (
                self.elo.get_rating(row["opponent_id"]) + opponent_home_advantage
            )
        df["rating_difference"] = df["team_rating"] - df["opponent_rating"]
        return self.logit.model.predict(
            self.logit.params, exog=df[["rating_difference"]]
        )

    def predict_and_update(self, X, y):
        """Predict class probabilities and then update the classifier."""
        df = self._prepare_ratings(X, y)
        df["rating_difference"] = df["team_rating"] - df["opponent_rating"]
        return self.logit.model.predict(
            self.logit.params, exog=df[["rating_difference"]]
        )
