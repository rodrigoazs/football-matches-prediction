import unittest

import pandas as pd
import torch

from src.models.dualemb import (
    DualEmbeddingNN,
    DualEmbPredictor,
    _predict_and_update,
    _prepare_dataset,
    _prepare_predicted_dataset,
    _train,
    _update,
)


def test_dualebm_prepare_dataset():
    df = pd.DataFrame(
        [
            {
                "date": "2024-02-01",
                "home_team": "team1",
                "home_score": 2,
                "away_score": 3,
                "away_team": "team2",
                "neutral": False,
            },
            {
                "date": "2024-01-01",
                "home_team": "team1",
                "home_score": 2,
                "away_score": 1,
                "away_team": "team3",
                "neutral": True,
            },
        ]
    )
    result, mappings = _prepare_dataset(df)
    assert result == [
        [0, 2, 0, 0, 2, 1],
        [2, 0, 0, 0, 1, 2],
        [0, 1, 1, 0, 2, 3],
        [1, 0, 0, 1, 3, 2],
    ]
    assert mappings == {"team1": 0, "team2": 1, "team3": 2}


def test_dualebm_prepare_predicted_dataset():
    outputs = torch.tensor([[1, 2], [4, 3], [5, 6], [7, 7]])
    targets = torch.tensor([[1, 2], [2, 1], [1, 1], [0, 0]])
    df = _prepare_predicted_dataset(outputs, targets)
    assert df.to_dict() == {
        "predicted_score_difference": {0: -1, 1: 1, 2: -1, 3: 0},
        "categorical_result": {0: "loss", 1: "win", 2: "draw", 3: "draw"},
    }


class TestDualEmbeddingNN(unittest.TestCase):
    def setUp(self):
        # Set up the model and test data
        self.num_embeddings = 100  # Number of unique IDs
        self.num_features = 2  # Number of features
        self.embedding_dim = 10  # Dimension of the embeddings
        self.hidden_dim = 20  # Hidden layer dimension
        self.batch_size = 5  # Batch size for testing

        # Initialize the model
        self.model = DualEmbeddingNN(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim,
            num_features=self.num_features,
            hidden_dim=self.hidden_dim,
        )

        # Create test input tensors
        self.id1 = torch.randint(0, self.num_embeddings, (self.batch_size,))
        self.id2 = torch.randint(0, self.num_embeddings, (self.batch_size,))
        self.team_at_home = torch.randint(0, 2, (self.batch_size,))
        self.opponent_at_home = torch.randint(0, 2, (self.batch_size,))
        self.score_targets = torch.randint(0, 5, (self.batch_size, 2)).float()
        self.data = torch.cat(
            (
                self.id1.unsqueeze(-1),
                self.id2.unsqueeze(-1),
                self.team_at_home.unsqueeze(-1),
                self.opponent_at_home.unsqueeze(-1),
            ),
            dim=-1,
        )

    def test_output_shape(self):
        # Test that the output shape is correct
        output = self.model(self.data)
        self.assertEqual(output.shape, (self.batch_size, 2))

    def test_embedding_application(self):
        # Test that embeddings are applied correctly
        embedding1 = self.model.embedding(self.id1)
        embedding2 = self.model.embedding(self.id2)

        self.assertEqual(embedding1.shape, (self.batch_size, self.embedding_dim))
        self.assertEqual(embedding2.shape, (self.batch_size, self.embedding_dim))

    def test_forward_pass(self):
        # Test the forward pass logic
        output = self.model(self.data)

        # Ensure the output is non-negative (since we apply torch.exp in the last layer)
        self.assertTrue(torch.all(output >= 0))

    def test_train(self):
        batch_size = 2
        learning_rate = 0.001
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        _train(
            model=self.model,
            optimizer=optimizer,
            criterion=criterion,
            data=self.data,
            targets=self.score_targets,
            batch_size=self.batch_size,
        )

    def test_update(self):
        learning_rate = 0.001
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        id1 = torch.tensor([0, 1])
        id2 = torch.tensor([1, 0])
        team_at_home = torch.tensor([0, 1])
        opponent_at_home = torch.tensor([1, 0])
        score_targets = torch.tensor([[2, 1], [1, 2]]).float()
        data = torch.cat(
            (
                id1.unsqueeze(-1),
                id2.unsqueeze(-1),
                team_at_home.unsqueeze(-1),
                opponent_at_home.unsqueeze(-1),
            ),
            dim=-1,
        )
        embeddings = torch.rand((2, self.embedding_dim))
        outputs, updated_embedding = _update(
            model=self.model,
            optimizer=optimizer,
            criterion=criterion,
            data=data,
            targets=score_targets,
            embeddings=embeddings,
        )
        self.assertEqual(outputs.shape, (2, 2))
        self.assertEqual(updated_embedding.shape, (2, self.embedding_dim))

    def test_predict_and_update(self):
        learning_rate = 0.001
        df = pd.DataFrame(
            [
                {
                    "date": "2024-02-01",
                    "home_team": "team1",
                    "home_score": 2,
                    "away_score": 3,
                    "away_team": "team2",
                    "neutral": False,
                },
                {
                    "date": "2024-02-02",
                    "home_team": "team1",
                    "home_score": 2,
                    "away_score": 3,
                    "away_team": "team3",
                    "neutral": False,
                },
            ]
        )
        default_embedding = torch.rand((1, self.embedding_dim)).tolist()[0]
        outputs, targets, teams_embeddings = _predict_and_update(
            df, self.model, default_embedding, learning_rate, embeddings=None
        )
        self.assertEqual(outputs.shape, (4, 2))
        self.assertEqual(targets.shape, (4, 2))
        self.assertEqual(len(teams_embeddings), 3)


def test_dualemb_fit():
    df = pd.DataFrame(
        [
            {
                "date": "2024-02-01",
                "home_team": "team1",
                "home_score": 2,
                "away_score": 3,
                "away_team": "team2",
                "neutral": False,
            },
            {
                "date": "2024-01-01",
                "home_team": "team1",
                "home_score": 2,
                "away_score": 1,
                "away_team": "team3",
                "neutral": True,
            },
        ]
    )
    model = DualEmbPredictor()
    model.fit(df)
    assert model._res_log is not None
    assert model._model is not None


def test_dualemb_predict_proba():
    df = pd.DataFrame(
        [
            {
                "date": "2024-02-01",
                "home_team": "team1",
                "home_score": 2,
                "away_score": 3,
                "away_team": "team2",
                "neutral": False,
            },
            {
                "date": "2024-01-01",
                "home_team": "team1",
                "home_score": 2,
                "away_score": 1,
                "away_team": "team3",
                "neutral": True,
            },
        ]
    )
    model = DualEmbPredictor()
    model.fit(df)
    pred = model.predict_proba(df)
    assert pred.shape == (2, 3)
