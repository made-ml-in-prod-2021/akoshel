import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple
from enities import SplitParams


def read_csv(path: str, **kwargs) -> pd.DataFrame:
    data = pd.read_csv(path, **kwargs)
    return data


def get_train_test_data(data: pd.DataFrame, params: SplitParams) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data to train test samples
    :param data:
    :param params:
    :return:
    """
    train_data, test_data = train_test_split(
        data, test_size=params.validation_size, random_state=params.random_state
    )
    return train_data, test_data
