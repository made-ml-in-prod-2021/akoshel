import os
from ml_project.train_pipeline import get_model_and_dataprocessor
from ml_project.enities import read_training_pipeline_params

def validate(year: str,
             month: str,
             day: str,
             config_path: str,
             ) -> None:
    params = read_training_pipeline_params(config_path)
    directory = f"{params.output_model_path}/{year}/{month}/{day}"
    if not os.path.exists(directory):
        os.makedirs(directory)
    params.input_data_path += f"/{year}/{month}/{day}/train.csv"
    params.output_model_path += f"/{year}/{month}/{day}/logistic_regression.p
    params.output_model_path = os.path.join(DEFAULT_PATH_HW1, params.output_model_path)
    params.output_data_preprocessor_path = os.path.join(DEFAULT_PATH_HW1, params.output_data_preprocessor_path)
    classifier, data_processor = get_model_and_dataprocessor(params)