import numpy as np

from src.models.uni import UniformMatchPredictor


def test_uni_match_predictor_predict(mock_inputs, mock_targets):
    model = UniformMatchPredictor()
    pred = model.predict(mock_inputs)
    np.testing.assert_array_equal(
        pred,
        np.ones((len(mock_inputs), 3)) / 3,
    )
