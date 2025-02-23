import numpy as np
import pandas as pd
import torch
import tqdm
from statsmodels.miscmodels.ordinal_model import OrderedModel

from src.models.base import BaseMatchPredictor

CATEGORICAL_DTYPE = pd.CategoricalDtype(
    categories=["win", "draw", "loss"], ordered=True
)


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
            (embedding1, embedding2, input_matrix[:, 2:]),
            dim=-1,
        )  # Concatenate along the last dimension

        x = torch.relu(self.fc1(combined))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))

        return x


class DualEmbPredictor(BaseMatchPredictor):
    def __init__(
        self,
        embedding_dim=10,
        hidden_dim=3,
        num_epochs=100,
        train_batch_size=32,
        train_learning_rate=0.001,
        update_learning_rate=0.05,
    ):
        self.embeddings = {}
        self.logit = None
        self.model = None
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_epochs = num_epochs
        self.train_batch_size = train_batch_size
        self.train_learning_rate = train_learning_rate
        self.update_learning_rate = update_learning_rate

    def _average_outputs(self, outputs):
        team_indices = outputs[::2].float()
        opponent_indices = outputs[1::2][:, [1, 0]].float()
        average_outputs = (team_indices + opponent_indices) / 2
        result_matrix = torch.zeros_like(outputs).float()
        result_matrix[::2] = average_outputs
        result_matrix[1::2] = average_outputs[:, [1, 0]]
        return result_matrix

    def _prepare_predicted_dataset(self, outputs, targets):
        df = pd.DataFrame(
            {
                "predicted_team_score": outputs[:, 0].tolist(),
                "predicted_opponent_score": outputs[:, 1].tolist(),
                "team_score": targets[:, 0].tolist(),
                "opponent_score": targets[:, 1].tolist(),
            }
        )
        df["predicted_score_difference"] = (
            df["predicted_team_score"] - df["predicted_opponent_score"]
        )
        df["categorical_result"] = df.apply(
            lambda x: "win"
            if x["team_score"] > x["opponent_score"]
            else "draw"
            if x["team_score"] == x["opponent_score"]
            else "loss",
            axis=1,
        )
        df["categorical_result"] = df["categorical_result"].astype(CATEGORICAL_DTYPE)
        return df[["predicted_score_difference", "categorical_result"]]

    def _prepare_predicted_score_dataset(self, outputs):
        df = pd.DataFrame(
            {
                "predicted_team_score": outputs[:, 0].tolist(),
                "predicted_opponent_score": outputs[:, 1].tolist(),
            }
        )
        df["predicted_score_difference"] = (
            df["predicted_team_score"] - df["predicted_opponent_score"]
        )
        return df[["predicted_score_difference"]]

    def _prepare_dataset(self, X: pd.DataFrame, y: pd.DataFrame) -> list:
        team_mapping = {
            team: index
            for index, team in enumerate(
                sorted(
                    list(
                        set(X["team_id"].unique()).union(set(X["opponent_id"].unique()))
                    )
                )
            )
        }
        df_X, df_y = self._reverse_matches(X, y)
        df_X["team_id"] = df_X["team_id"].map(team_mapping)
        df_X["opponent_id"] = df_X["opponent_id"].map(team_mapping)
        return df_X, df_y, team_mapping

    def _prepare_embeddings(self, team_mapping: dict) -> list[list[float]]:
        embeddings = [None] * len(team_mapping)
        for team, index in team_mapping.items():
            embeddings[index] = self.embeddings.get(team, self.default_embedding)
        return embeddings

    def _update_embeddings(
        self, team_mapping: dict, embeddings: list[list[float]]
    ) -> None:
        embeddings_map = {k: embeddings[v] for k, v in team_mapping.items()}
        self.embeddings.update(embeddings_map)

    def _predict_and_update(
        self, X, y, model, default_embedding, learning_rate, embeddings=None
    ):
        teams_embeddings = {} if embeddings is None else embeddings
        outputs = None
        targets = None
        reversed_X, reversed_y = self._reverse_matches()
        for _, row in df.iterrows():
            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            team_embedding = teams_embeddings.get(row["team_id"], default_embedding)
            opponent_embedding = teams_embeddings.get(
                row["opponent_id"], default_embedding
            )
            embeddings = torch.tensor([team_embedding, opponent_embedding])
            output, updated_embedding = _update(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                data=X,
                targets=y,
                embeddings=embeddings,
            )
            teams_embeddings[row["home_team"]] = updated_embedding[0].tolist()
            teams_embeddings[row["away_team"]] = updated_embedding[1].tolist()
            outputs = (
                torch.cat((outputs, output), dim=0) if outputs is not None else output
            )
            targets = torch.cat((targets, y), dim=0) if targets is not None else y
        return outputs, targets, teams_embeddings

    def _update(self, model, optimizer, criterion, data, targets, embeddings):
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

    def _train(
        self,
        model,
        optimizer,
        criterion,
        data,
        targets,
        batch_size,
        val_data=None,
        val_targets=None,
    ):
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
            outputs = model(batch_data)

            # Compute the loss
            loss = criterion(outputs, batch_targets)

            # Backward pass: compute gradients
            loss.backward()

            # Update weights
            optimizer.step()

            # Update progress bar
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Validation
        if val_data is not None and val_targets is not None:
            model.eval()
            with torch.no_grad():
                val_outputs = model(val_data)
                val_loss = criterion(val_outputs, val_targets)
                print(f"Validation Loss: {val_loss.item():.4f}")

    def fit(
        self, X: pd.DataFrame, y: pd.DataFrame, validation_set: float = 0.0
    ) -> None:
        X, y, team_mapping = self._prepare_dataset(X, y)
        # Create random indices for the validation set
        if validation_set > 0.0:
            num_samples = len(X) // 2
            even_indices = np.arange(0, num_samples * 2, 2)
            np.random.shuffle(even_indices)
            indices = np.empty((even_indices.size * 2,), dtype=even_indices.dtype)
            indices[0::2] = even_indices
            indices[1::2] = even_indices + 1
            split_idx = int(num_samples * (1 - validation_set)) * 2
            train_indices, val_indices = indices[:split_idx], indices[split_idx:]
            train_X, val_X = X.iloc[train_indices], X.iloc[val_indices]
            train_y, val_y = y.iloc[train_indices], y.iloc[val_indices]
        else:
            train_X, val_X = X, None
            train_y, val_y = y, None
        # Convert data
        data = torch.tensor(train_X.values).long()
        targets = torch.tensor(train_y.values).float()
        if validation_set > 0.0:
            val_data = torch.tensor(val_X.values).long()
            val_targets = torch.tensor(val_y.values).float()
        else:
            val_data = None
            val_targets = None
        # Get parameters
        num_embeddings = len(team_mapping)
        num_features = data.shape[1] - 2
        # Train model
        self.model = DualEmbeddingNN(
            num_embeddings=num_embeddings,
            embedding_dim=self.embedding_dim,
            num_features=num_features,
            hidden_dim=self.hidden_dim,
        )
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.train_learning_rate
        )
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            self._train(
                model=self.model,
                optimizer=optimizer,
                criterion=criterion,
                data=data,
                targets=targets,
                batch_size=self.train_batch_size * 2,
                val_data=val_data,
                val_targets=val_targets,
            )
        # Extract the average embedding to get the default embedding
        self.default_embedding = self.model.embedding.weight.grad.mean(dim=0).tolist()
        # Predict
        with torch.no_grad():
            outputs = self._average_outputs(self.model(data))
        # Train logit model
        df = self._prepare_predicted_dataset(outputs, targets)
        mod_log = OrderedModel(
            df["categorical_result"], df[["predicted_score_difference"]], distr="logit"
        )
        self.logit = mod_log.fit(method="bfgs", disp=False)

    def update(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        X, y, team_mapping = self._prepare_dataset(X, y)
        # Prepare embeddings
        embeddings = self._prepare_embeddings(team_mapping)
        # Iterate pairs
        for i in range(0, len(X), 2):
            pair_X = X.iloc[i : i + 2]
            pair_y = y.iloc[i : i + 2]
            # Save ids
            team_id = int(pair_X.iloc[0]["team_id"])
            opponent_id = int(pair_X.iloc[0]["opponent_id"])
            # Temporary ids
            pair_X["team_id"] = [0, 1]
            pair_X["opponent_id"] = [1, 0]
            # Setup
            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.update_learning_rate
            )
            team_embedding = embeddings[team_id]
            opponent_embedding = embeddings[opponent_id]
            tensor_embeddings = torch.tensor([team_embedding, opponent_embedding])
            # Convert data
            data = torch.tensor(pair_X.values).long()
            targets = torch.tensor(pair_y.values).float()
            # Update
            _, updated_embedding = self._update(
                model=self.model,
                optimizer=optimizer,
                criterion=criterion,
                data=data,
                targets=targets,
                embeddings=tensor_embeddings,
            )
            embeddings[team_id] = updated_embedding[0].tolist()
            embeddings[opponent_id] = updated_embedding[1].tolist()
        self._update_embeddings(team_mapping, embeddings)

    def predict(self, X: pd.DataFrame):
        X, _, team_mapping = self._prepare_dataset(X, pd.DataFrame([]))
        # Convert data
        data = torch.tensor(X.values).long()
        # Prepare embeddings
        embeddings = torch.tensor(self._prepare_embeddings(team_mapping))
        self.model.embedding.weight = torch.nn.Parameter(embeddings)
        # Predict
        with torch.no_grad():
            outputs = self._average_outputs(self.model(data))
        df = self._prepare_predicted_score_dataset(outputs)
        return self.logit.model.predict(
            self.logit.params, exog=df[["predicted_score_difference"]]
        )

    def predict_and_update(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        pass
