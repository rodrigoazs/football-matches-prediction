from abc import ABC, abstractmethod


class BaseMatchPredictor(ABC):
    @abstractmethod
    def fit(self, X):
        """
        Fit the model to the data (X).
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Predict the output based on input data X.
        """
        pass

    @abstractmethod
    def predict_proba(self, X):
        """
        Predict probabilities for the output based on input data X.
        """
        pass

    @abstractmethod
    def update_ratings(self, X):
        """
        Update model ratings based on input data X.
        """
        pass
