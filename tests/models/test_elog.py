import numpy as np
import pandas as pd

from src.models.elog import ELOgPredictor, EloRating


def test_elo_initial_rating():
    elo = EloRating()
    assert elo.get_rating("teamA") == 1500
    assert elo.get_rating("teamB") == 1500


def test_elo_create_rating():
    elo = EloRating()
    elo.create_rating("teamA", "teamB")
    assert elo.get_rating("teamA") == 1500
    assert elo.get_rating("teamB") == 1500


def test_elo_expected_score():
    elo = EloRating()
    elo.create_rating("teamA", "teamB")
    expected_score_A = elo.expected_score("teamA", "teamB", neutral=True)
    expected_score_B = elo.expected_score("teamB", "teamA", neutral=True)

    assert expected_score_A == 0.5
    assert expected_score_B == 0.5


def test_elo_update_ratings_home_wins():
    elo = EloRating()
    elo.create_rating("teamA", "teamB")
    elo.update_ratings("teamA", "teamB", 3, 1, neutral=False)

    rating_A = elo.get_rating("teamA")
    rating_B = elo.get_rating("teamB")

    assert rating_A > 1500  # home team (teamA) should have increased rating
    assert rating_B < 1500  # away team (teamB) should have decreased rating


def test_elo_update_ratings_away_wins():
    elo = EloRating()
    elo.create_rating("teamA", "teamB")
    elo.update_ratings("teamA", "teamB", 1, 3, neutral=False)

    rating_A = elo.get_rating("teamA")
    rating_B = elo.get_rating("teamB")

    assert rating_A < 1500  # home team (teamA) should have decreased rating
    assert rating_B > 1500  # away team (teamB) should have increased rating


def test_elo_update_ratings_draw():
    elo = EloRating()
    elo.create_rating("teamA", "teamB")
    elo.update_ratings("teamA", "teamB", 2, 2, neutral=False)

    rating_A = elo.get_rating("teamA")
    rating_B = elo.get_rating("teamB")

    assert rating_A != 1500  # Both teams' ratings should change after a draw
    assert rating_B != 1500


def test_elo_update_ratings_draw_neutral():
    elo = EloRating()
    elo.create_rating("teamA", "teamB")
    elo.update_ratings("teamA", "teamB", 2, 2, neutral=True)

    rating_A = elo.get_rating("teamA")
    rating_B = elo.get_rating("teamB")

    assert rating_A == 1500  # Both teams' ratings should change after a draw
    assert rating_B == 1500


def test_elo_home_advantage():
    elo = EloRating()
    elo.create_rating("teamA", "teamB")
    expected_score_home = elo.expected_score("teamA", "teamB", neutral=False)

    # For a home game, the home team should have an advantage, so the expected score of home should be greater
    assert expected_score_home > 0.5


def test_elog_predictor_prepare_ratings(mock_inputs, mock_targets):
    elo = ELOgPredictor()
    df = elo._prepare_ratings(mock_inputs, mock_targets)
    assert df["team_rating"][0] == 1560
    assert df["opponent_rating"][0] == 1500
    assert df["team_rating"][1] < 1500
    assert df["opponent_rating"][1] == 1500


def test_elog_predictor_fit(mock_inputs, mock_targets):
    elo = ELOgPredictor()
    elo.fit(mock_inputs, mock_targets)
    assert elo.logit


def test_elog_predictor_update(mock_inputs, mock_targets):
    elo = ELOgPredictor()
    elo.update(mock_inputs, mock_targets)
    assert elo.elo.rating["team1"] < 1500
    assert elo.elo.rating["team2"] > 1500
    assert elo.elo.rating["team3"] > 1500


def test_elog_predictor_predict(mock_inputs, mock_targets):
    elo = ELOgPredictor()
    elo.fit(mock_inputs, mock_targets)
    pred = elo.predict(mock_inputs)
    assert pred.shape == (3, 3)
    # assert symmetric
    X = pd.DataFrame(
        [
            {
                "team_id": "team998",
                "opponent_id": "team999",
                "team_at_home": 0.0,
                "opponent_at_home": 0.0,
            }
        ]
    )
    pred = elo.predict(X)
    assert np.array_equal(np.argmax(pred, axis=1), np.array([1]))
    assert np.allclose(pred[:, 0], pred[:, 2], atol=1e-3)


def test_elog_predictor_predict_symmetric(mock_inputs, mock_targets):
    X = pd.DataFrame(
        [
            {
                "team_id": "team3",
                "opponent_id": "team4",
                "team_at_home": 1.0,
                "opponent_at_home": 0.0,
            },
            {
                "team_id": "team1",
                "opponent_id": "team2",
                "team_at_home": 0.0,
                "opponent_at_home": 0.0,
            },
            {
                "team_id": "team1",
                "opponent_id": "team2",
                "team_at_home": 0.0,
                "opponent_at_home": 0.0,
            },
            {
                "team_id": "team1",
                "opponent_id": "team2",
                "team_at_home": 0.0,
                "opponent_at_home": 0.0,
            },
            {
                "team_id": "team5",
                "opponent_id": "team6",
                "team_at_home": 0.0,
                "opponent_at_home": 1.0,
            },
        ]
    )
    y = pd.DataFrame(
        [
            {
                "team_score": 1.0,
                "opponent_score": 0.0,
            },
            {
                "team_score": 0.0,
                "opponent_score": 0.0,
            },
            {
                "team_score": 1.0,
                "opponent_score": 1.0,
            },
            {
                "team_score": 2.0,
                "opponent_score": 2.0,
            },
            {
                "team_score": 0.0,
                "opponent_score": 1.0,
            },
        ]
    )
    elo = ELOgPredictor()
    elo.fit(X, y)
    pred = elo.predict(X)
    # at home win, all draws in neutral, at away loss
    assert np.array_equal(np.argmax(pred, axis=1), np.array([0, 1, 1, 1, 2]))
    # assert symmetric
    assert np.allclose(pred[1:-1, 0], pred[1:-1, 2], atol=1e-3)


def test_elog_predictor_predict_ant_update(mock_inputs, mock_targets):
    elo = ELOgPredictor()
    elo.fit(mock_inputs, mock_targets)
    pred = elo.predict_and_update(mock_inputs, mock_targets)
    assert elo.elo.rating["team1"] < 1500
    assert elo.elo.rating["team2"] > 1500
    assert elo.elo.rating["team3"] > 1500
    assert pred.shape == (3, 3)
