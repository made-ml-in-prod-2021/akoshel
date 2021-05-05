import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple
from enities import SplitParams
import logging


APPLICATION_NAME = 'homework01'
logger = logging.getLogger(APPLICATION_NAME)

def read_csv(path: str, **kwargs) -> pd.DataFrame:
    data = pd.read_csv(path, **kwargs)
    logger.info(f"Dataframe read from csv shape: {data.shape}")
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
