import numpy as np
import pandas as pd

from src.models.base import BaseMatchPredictor


class UniformMatchPredictor(BaseMatchPredictor):
    def __init__(self):
        self.prob = {}

    def fit(self, X, y):
        pass

    def update(self, X, y):
        """Update a classifier from the a given set (X, y)."""
        pass

    def predict(self, X):
        """Predict class probabilities of the input samples X."""
        return np.ones((len(X), 3)) / 3

    def predict_and_update(self, X, y):
        """Predict class probabilities and then update the classifier."""
        return self.predict(X)
