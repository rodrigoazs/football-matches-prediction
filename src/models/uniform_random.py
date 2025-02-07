import numpy as np

from src.models.base import BaseMatchPredictor


class UniformRandomMatchPredictor(BaseMatchPredictor):
    def fit(self, X, y):
        """Build a classifier from the training set (X, y)."""
        pass

    def update(self, X, y):
        """Update a classifier from the a given set (X, y)."""
        pass

    def predict(self, X):
        """Predict class probabilities of the input samples X."""
        return np.ones((len(X), 3)) / 3

    def predict_and_update(self, X):
        """Predict class probabilities and then update the classifier."""
        return np.ones((len(X), 3)) / 3
