from sklearn.linear_model import LogisticRegression
import numpy as np


class Classifier:

    def __init__(self, params):
        self.model = LogisticRegression(C=params.C, penalty=params.penalty, fit_intercept=params.fit_intercept,
                                        random_state=params.random_state)

    def fit(self, X: np.array, y: np.array) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.array) -> np.array:
        return self.model.predict(X)