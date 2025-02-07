import numpy as np

from src.models.freq import FrequencyMatchPredictor


def test_freq_match_predictor_fit(mock_inputs, mock_targets):
    model = FrequencyMatchPredictor()
    model.fit(mock_inputs, mock_targets)
    assert model.prob == {
        (0.0, 0.0): [0.0, 1.0, 0.0],
        (0.0, 1.0): [1.0, 0.0, 0.0],
        (1.0, 0.0): [0.0, 0.0, 1.0],
    }


def test_freq_match_predictor_predict(mock_inputs, mock_targets):
    model = FrequencyMatchPredictor()
    pred = model.predict(mock_inputs)
    np.testing.assert_array_equal(pred, np.ones((6, 3)) / 3)
    model.fit(mock_inputs, mock_targets)
    pred = model.predict(mock_inputs)
    np.testing.assert_array_equal(
        pred,
        np.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
    )
