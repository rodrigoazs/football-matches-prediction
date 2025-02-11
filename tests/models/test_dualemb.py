import unittest

import pandas as pd
import pytest
import torch

from src.models.dualemb import (
    DualEmbeddingNN,
    DualEmbPredictor,
)


@pytest.fixture
def setup_dualemb_model():
    # Set up the model and test data
    num_embeddings = 100  # Number of unique IDs
    num_features = 2  # Number of features
    embedding_dim = 10  # Dimension of the embeddings
    hidden_dim = 20  # Hidden layer dimension
    batch_size = 5  # Batch size for testing

    # Initialize the model
    model = DualEmbeddingNN(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        num_features=num_features,
        hidden_dim=hidden_dim,
    )

    # Create test input tensors
    id1 = torch.randint(0, num_embeddings, (batch_size,))
    id2 = torch.randint(0, num_embeddings, (batch_size,))
    team_at_home = torch.randint(0, 2, (batch_size,))
    opponent_at_home = torch.randint(0, 2, (batch_size,))
    score_targets = torch.randint(0, 5, (batch_size, 2)).float()
    data = torch.cat(
        (
            id1.unsqueeze(-1),
            id2.unsqueeze(-1),
            team_at_home.unsqueeze(-1),
            opponent_at_home.unsqueeze(-1),
        ),
        dim=-1,
    )

    return model, data, score_targets


def test_dualebm_prepare_dataset(mock_inputs, mock_targets):
    model = DualEmbPredictor()
    result_X, result_y, mappings = model._prepare_dataset(mock_inputs, mock_targets)
    expected_df = pd.DataFrame(
        {
            "team_id": [0, 1, 0, 2, 2, 1],
            "opponent_id": [1, 0, 2, 0, 1, 2],
            "team_at_home": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "opponent_at_home": [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        }
    )
    expected_y = pd.DataFrame(
        {
            "team_score": [2.0, 3.0, 3.0, 3.0, 1.0, 0.0],
            "opponent_score": [3.0, 2.0, 3.0, 3.0, 0.0, 1.0],
        }
    )
    pd.testing.assert_frame_equal(result_X, expected_df)
    pd.testing.assert_frame_equal(result_y, expected_y)
    assert mappings == {"team1": 0, "team2": 1, "team3": 2}


def test_dualebm_prepare_predicted_dataset():
    model = DualEmbPredictor()
    outputs = torch.tensor([[1, 2], [4, 3], [5, 6], [7, 7]])
    targets = torch.tensor([[1, 2], [2, 1], [1, 1], [0, 0]])
    df = model._prepare_predicted_dataset(outputs, targets)
    assert df.to_dict() == {
        "predicted_score_difference": {0: -1, 1: 1, 2: -1, 3: 0},
        "categorical_result": {0: "loss", 1: "win", 2: "draw", 3: "draw"},
    }


def test_dualebm_output_shape(setup_dualemb_model):
    model, data, _ = setup_dualemb_model
    # Test that the output shape is correct
    output = model(data)
    assert output.shape == (5, 2)


def test_dualebm_embedding_application(setup_dualemb_model):
    model, _, _ = setup_dualemb_model
    id1 = torch.randint(0, 1, (5,))
    # Test that embeddings are applied correctly
    embedding1 = model.embedding(id1)
    assert embedding1.shape == (5, 10)


def test_dualebm_forward_pass(setup_dualemb_model):
    (
        model,
        data,
        _,
    ) = setup_dualemb_model
    # Test the forward pass logic
    output = model(data)
    # Ensure the output is non-negative (since we apply torch.exp in the last layer)
    assert torch.all(output >= 0)


def test_dualebm_train(setup_dualemb_model):
    model = DualEmbPredictor()
    neural_model, data, score_targets = setup_dualemb_model
    batch_size = 2
    learning_rate = 0.001
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(neural_model.parameters(), lr=learning_rate)
    model._train(
        model=neural_model,
        optimizer=optimizer,
        criterion=criterion,
        data=data,
        targets=score_targets,
        batch_size=batch_size,
    )


def test_dualebm_update(setup_dualemb_model):
    model = DualEmbPredictor()
    neural_model, _, score_targets = setup_dualemb_model
    learning_rate = 0.001
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(neural_model.parameters(), lr=learning_rate)
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
    embeddings = torch.rand((2, 10))
    outputs, updated_embedding = model._update(
        model=neural_model,
        optimizer=optimizer,
        criterion=criterion,
        data=data,
        targets=score_targets,
        embeddings=embeddings,
    )
    assert outputs.shape == (2, 2)
    assert updated_embedding.shape == (2, 10)


# def test_dualebm_predict_and_update(setup_model):
#     model, _, _, _, _, _, _ = setup_model
#     learning_rate = 0.001
#     df = pd.DataFrame(
#         [
#             {
#                 "date": "2024-02-01",
#                 "home_team": "team1",
#                 "home_score": 2,
#                 "away_score": 3,
#                 "away_team": "team2",
#                 "neutral": False,
#             },
#             {
#                 "date": "2024-02-02",
#                 "home_team": "team1",
#                 "home_score": 2,
#                 "away_score": 3,
#                 "away_team": "team3",
#                 "neutral": False,
#             },
#         ]
#     )
#     default_embedding = torch.rand((1, 10)).tolist()[0]
#     outputs, targets, teams_embeddings = _predict_and_update(
#         df, model, default_embedding, learning_rate, embeddings=None
#     )
#     assert outputs.shape == (4, 2)
#     assert targets.shape == (4, 2)
#     assert len(teams_embeddings) == 3


def test_dualemb_fit(mock_inputs, mock_targets):
    model = DualEmbPredictor()
    model.fit(mock_inputs, mock_targets)
    assert model.logit is not None
    assert model.model is not None


# def test_dualemb_predict_proba():
#     df = pd.DataFrame(
#         [
#             {
#                 "date": "2024-02-01",
#                 "home_team": "team1",
#                 "home_score": 2,
#                 "away_score": 3,
#                 "away_team": "team2",
#                 "neutral": False,
#             },
#             {
#                 "date": "2024-01-01",
#                 "home_team": "team1",
#                 "home_score": 2,
#                 "away_score": 1,
#                 "away_team": "team3",
#                 "neutral": True,
#             },
#         ]
#     )
#     model = DualEmbPredictor()
#     model.fit(df)
#     pred = model.predict_proba(df)
#     assert pred.shape == (2, 3)
