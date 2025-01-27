import numpy as np
import pandas as pd
from statsmodels.miscmodels.ordinal_model import OrderedModel

from src.models.base import BaseMatchPredictor
from src.utils import swap_dataset

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
            1
            + 10
            ** ((self.rating[away] - (self.rating[home] + home_advantage)) / 400)
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
    def __init__(self):
        self._res_log = None

    def _swap_dataset(self, X):
        df = X.copy()
        swaped_df = swap_dataset(df)
        return pd.concat([df, swaped_df])

    def _prepare_ratings(self, X):
        df = X.copy()
        elo = EloRating()
        df["home_score"] = df["home_score"].astype(int)
        df["away_score"] = df["away_score"].astype(int)

        for index, row in df.iterrows():
            home_advantage = elo.home_advantage if row["neutral"] == False else 0
            df.loc[index, "home_rating"] = (
                elo.get_rating(row["home_team"]) + home_advantage
            )
            df.loc[index, "away_rating"] = elo.get_rating(row["away_team"])
            elo.update_ratings(
                row["home_team"], row["away_team"], row["home_score"], row["away_score"], row["neutral"]
            )
        return df

    def fit(self, X: pd.DataFrame) -> None:
        df = self._prepare_ratings(X)
        df["categorical_result"] = df.apply(
            lambda x: "win"
            if x["home_score"] > x["away_score"]
            else "draw"
            if x["home_score"] == x["away_score"]
            else "loss",
            axis=1,
        )
        df["categorical_result"] = df["categorical_result"].astype(CATEGORICAL_DTYPE)
        df["rating_difference"] = df["home_rating"] - df["away_rating"]

        mod_log = OrderedModel(
            df["categorical_result"], df[["rating_difference"]], distr="logit"
        )

        self._res_log = mod_log.fit(method="bfgs", disp=False)

    def predict(self, X):
        prob = self.predict_proba(X)
        return np.argmax(prob, axis=1)

    def predict_proba(self, X):
        df = self._prepare_ratings(X)
        df["rating_difference"] = df["home_rating"] - df["away_rating"]
        x = df["rating_difference"].to_numpy()
        return self._res_log.model.predict(self._res_log.params, exog=x.reshape(-1, 1))

    def update_ratings(self, X):
        pass
