import pandas as pd
from sklearn.model_selection import train_test_split
import os
from ml_project.train_pipeline import train_pipeline
from ml_project.enities import read_training_pipeline_params


def train_model(year: str,
                 month: str,
                 day: str,
                 config_path: str,
                 ) -> None:
    params = read_training_pipeline_params(config_path)
    directory = f"{params.output_model_path}/{year}/{month}/{day}"
    if not os.path.exists(directory):
        os.makedirs(directory)
    params.input_data_path += f"/{year}/{month}/{day}/train.csv"
    params.output_model_path += f"/{year}/{month}/{day}/logistic_regression.pkl"
    params.output_data_preprocessor_path += f"/{year}/{month}/{day}/data_preprocessor_lr.pkl"
    train_pipeline(params)
