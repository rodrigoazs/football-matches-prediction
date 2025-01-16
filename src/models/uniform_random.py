import numpy as np

from src.models.base import BaseMatchPredictor


class UniformRandomMatchPredictor(BaseMatchPredictor):
    def fit(self, X):
        pass

    def predict(self, X):
        # return np.random.choice([0, 1, 2], size=len(X))
        prob = self.predict_proba(X)
        return np.argmax(prob, axis=1)

    def predict_proba(self, X):
        return np.ones((len(X), 3)) / 3

    def update_ratings(self, X):
        pass
