from dataclasses import dataclass


@dataclass
class ClassifierParams:
    C: float
    penalty: str
    fit_intercept: bool
    random_state: int