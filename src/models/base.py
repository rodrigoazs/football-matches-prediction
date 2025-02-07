from abc import ABC, abstractmethod


class BaseMatchPredictor(ABC):
    @abstractmethod
    def fit(self, X, y):
        """Build a classifier from the training set (X, y)."""
        pass

    @abstractmethod
    def update(self, X, y):
        """Update a classifier from the a given set (X, y)."""
        pass

    @abstractmethod
    def predict(self, X):
        """Predict class probabilities of the input samples X."""
        pass

    @abstractmethod
    def predict_and_update(self, X):
        """Predict class probabilities and then update the classifier."""
        pass
