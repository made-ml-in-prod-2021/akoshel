import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib


class Classifier:

    def __init__(self, params):
        self.model = LogisticRegression(C=params.C, penalty=params.penalty, fit_intercept=params.fit_intercept,
                                        random_state=params.random_state)

    def fit(self, X: np.array, y: np.array) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.array) -> np.array:
        return self.model.predict(X)

    def dump_model(self, path: str):
        joblib.dump(self.model, path)


def get_classification_report(y_true: np.array, y_pred: np.array):
    return classification_report(y_true, y_pred)