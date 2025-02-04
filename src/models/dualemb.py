import numpy as np
import pandas as pd
import torch
import tqdm
from statsmodels.miscmodels.ordinal_model import OrderedModel

from src.models.base import BaseMatchPredictor

CATEGORICAL_DTYPE = pd.CategoricalDtype(
    categories=["win", "draw", "loss"], ordered=True
)


# Training function
def train(model, optimizer, criterion, data, targets, batch_size):
    num_samples = data.size(0)
    model.train()

    # Iterate over the dataset in mini-batches
    pbar = tqdm.tqdm(range(0, num_samples, batch_size))
    for start in pbar:
        end = start + batch_size
        if end > num_samples:
            end = num_samples

        # Get the current mini-batch
        batch_data = data[start:end]
        batch_targets = targets[start:end]

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass: compute predictions
        outputs = model(data)

        # Compute the loss
        loss = criterion(outputs, batch_targets)

        # Backward pass: compute gradients
        loss.backward()

        # Update weights
        optimizer.step()

        # Update progress bar
        pbar.set_postfix(loss=f"{loss.item():.4f}")


# Update function
def update(model, optimizer, criterion, data, targets, embeddings):
    # Set the embedding matrix from the given matrix
    model.embedding.weight = torch.nn.Parameter(embeddings.clone())

    # Freeze all weights except for the embedding layer
    for name, param in model.named_parameters():
        if "embedding" not in name:  # Freeze all layers except the embedding layer
            param.requires_grad = False
        else:
            param.requires_grad = True  # Ensure embedding layer is trainable

    # Set the model to training mode
    model.train()

    # Zero the gradients
    optimizer.zero_grad()

    # Forward pass: compute predictions
    outputs = model(data)

    # Compute the loss
    loss = criterion(outputs, targets)

    # Backward pass: compute gradients
    loss.backward()

    # Update weights (only the embedding layer will be updated)
    optimizer.step()

    return outputs, model.embedding.weight.grad


class DualEmbeddingNN(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, num_features, hidden_dim):
        super(DualEmbeddingNN, self).__init__()
        self.embedding = torch.nn.Embedding(
            num_embeddings, embedding_dim
        )  # Embedding layer for IDs
        self.fc1 = torch.nn.Linear(2 * embedding_dim + num_features, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 2)

    def forward(self, input_matrix):
        # # Extract the relevant columns from the input matrix
        id1 = input_matrix[:, 0]  # Shape: (batch_size,)
        id2 = input_matrix[:, 1]  # Shape: (batch_size,)

        # Get embeddings for the IDs
        embedding1 = self.embedding(id1)  # Shape: (batch_size, embedding_dim)
        embedding2 = self.embedding(id2)  # Shape: (batch_size, embedding_dim)

        # Concatenate the two embeddings and the integer input
        combined = torch.cat(
            (embedding1, embedding2, input_matrix[:, :2]),
            dim=-1,
        )  # Concatenate along the last dimension

        x = torch.relu(self.fc1(combined))
        x = torch.relu(self.fc2(x))
        x = torch.exp(self.fc3(x))

        return x


class DualEmbPredictor(BaseMatchPredictor):
    def __init__(
        self,
        embedding_dim=10,
        hidden_dim=3,
        train_batch_size=2,
        update_batch_size=2,
        train_learning_rate=0.001,
        update_learning_rate=0.001,
    ):
        self._res_log = None
        self._team_mapping = {}
        self._embedding_dim = embedding_dim
        self._hidden_dim = hidden_dim
        self._train_batch_size = train_batch_size
        self._update_batch_size = update_batch_size
        self._train_learning_rate = train_learning_rate
        self._update_learning_rate = update_learning_rate
        self._model = None

    def _prepare_dataset(self, X):
        X_copy = X.copy().sort_values("date", ascending=True)
        self._team_mapping = {
            team: index
            for index, team in enumerate(
                sorted(
                    list(
                        set(X["home_team"].unique()).union(set(X["away_team"].unique()))
                    )
                )
            )
        }
        df = []
        for _, row in X_copy.iterrows():
            df.append(
                [
                    self._team_mapping[row["home_team"]],
                    self._team_mapping[row["away_team"]],
                    0 if row["neutral"] == True else 1,
                    0,
                    row["home_score"],
                    row["away_score"],
                ]
            )
            df.append(
                [
                    self._team_mapping[row["away_team"]],
                    self._team_mapping[row["home_team"]],
                    0,
                    0 if row["neutral"] == True else 1,
                    row["away_score"],
                    row["home_score"],
                ]
            )
        return df

    def fit(self, X: pd.DataFrame) -> None:
        df = self._prepare_dataset(df)
        df = torch.tensor(df)
        data = df[:, :-2]
        score_targets = df[:, -2:]
        num_embeddings = len(self._team_mapping)
        num_features = 2
        self._model = DualEmbeddingNN(
            num_embeddings=num_embeddings,
            embedding_dim=self._embedding_dim,
            num_features=num_features,
            hidden_dim=self._hidden_dim,
        )
        criterion = torch.nn.MSELoss()
        optimizer = optim.Adam(self._model.parameters(), lr=self._train_learning_rate)
        train(
            model=self._model,
            optimizer=optimizer,
            criterion=criterion,
            data=X,
            targets=score_targets,
            batch_size=self._train_batch_size,
        )
        # Extract the average embedding to get the default embedding
        self._default_embedding = self._model.embedding.weight.grad.mean(dim=0)
        self._team_embedding = {
            team: self._default_embedding for team in self._team_mapping.keys()
        }

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
