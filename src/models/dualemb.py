import numpy as np
import pandas as pd
# import torch
from statsmodels.miscmodels.ordinal_model import OrderedModel

from src.models.base import BaseMatchPredictor

CATEGORICAL_DTYPE = pd.CategoricalDtype(
    categories=["win", "draw", "loss"], ordered=True
)


# def swap_dataset(df):
#     swaped_df = pd.DataFrame()
#     swaped_df["date"] = df["date"]
#     swaped_df["a_team"] = df["b_team"]
#     swaped_df["b_team"] = df["a_team"]
#     swaped_df["a_score"] = df["b_score"]
#     swaped_df["b_score"] = df["a_score"]
#     swaped_df["a_home"] = df["b_home"]
#     swaped_df["b_home"] = df["a_home"]
#     return swaped_df


# class DualEmbeddingNN(torch.nn.Module):
#     def __init__(self, num_embeddings, embedding_dim, hidden_dim):
#         super(DualEmbeddingNN, self).__init__()
#         self.embedding = torch.nn.Embedding(
#             num_embeddings, embedding_dim
#         )  # Embedding layer for IDs
#         self.fc1 = torch.nn.Linear(2 * embedding_dim, hidden_dim)
#         self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = torch.nn.Linear(hidden_dim, 2)

#     def forward(self, id1, id2):
#         # Get embeddings for the IDs
#         embedding1 = self.embedding(id1)  # Shape: (batch_size, embedding_dim)
#         embedding2 = self.embedding(id2)  # Shape: (batch_size, embedding_dim)

#         # Concatenate the two embeddings and the integer input
#         # combined = torch.cat((embedding1, embedding2, integer_input.float()), dim=-1)  # Concatenate along the last dimension
#         combined = torch.cat(
#             (embedding1, embedding2), dim=-1
#         )  # Concatenate along the last dimension

#         x = torch.relu(self.fc1(combined))
#         x = torch.relu(self.fc2(x))
#         x = torch.exp(self.fc3(x))

#         return x


class DualEmbPredictor(BaseMatchPredictor):
    def __init__(self):
        self._res_log = None
        self._team_mapping = {}

    def _prepare_dataset(self, X):
        X_copy = X.copy().sort_values("date", ascending=True)
        self._team_mapping = {
            team: index
            for index, team in enumerate(
                set(X["home_team"].unique()).union(set(X["away_team"].unique()))
            )
        }
        df = []
        for _, row in X_copy.iterrows():
            df.append([
                self._team_mapping[row["home_team"]],
                self._team_mapping[row["away_team"]],
                row["home_score"],
                row["away_score"],
                0 if row["neutral"] == True else 1,
                0,
            ])
            df.append([
                self._team_mapping[row["home_team"]],
                self._team_mapping[row["away_team"]],
                row["home_score"],
                row["away_score"],
                0 if row["neutral"] == True else 1,
                0,
            ])
        return df

    def fit(self, X: pd.DataFrame) -> None:
        df = self._prepare_dataset(df)
        raise Exception(df)

    def predict(self, X):
        prob = self.predict_proba(X)
        return np.argmax(prob, axis=1)

    def predict_proba(self, X):
        df = X.copy().sort_values("date", ascending=True)
        df = self._prepare_ratings(df)
        df["rating_difference"] = df["home_rating"] - df["away_rating"]
        x = df["rating_difference"].to_numpy()
        return self._res_log.model.predict(self._res_log.params, exog=x.reshape(-1, 1))

    def update_ratings(self, X):
        pass
