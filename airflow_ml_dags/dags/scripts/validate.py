import os
import json
import pandas as pd
from ml_project.train_pipeline import get_model_and_dataprocessor, get_classification_report
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
    params.input_data_path += f"/{year}/{month}/{day}/val.csv"
    params.output_model_path += f"/{year}/{month}/{day}/logistic_regression.pkl"
    params.output_data_preprocessor_path += f"/{year}/{month}/{day}/data_preprocessor_lr.pkl"
    classifier, data_processor = get_model_and_dataprocessor(params)
    val_df = pd.read_csv(params.input_data_path)
    transformed_data = data_processor.transform(val_df)
    y_pred = classifier.predict(transformed_data)
    report = get_classification_report(val_df['target'].values, y_pred)
    params.predicts_path += f"/{year}/{month}/{day}"
    if not os.path.exists(params.predicts_path):
        os.makedirs(params.predicts_path)
    with open(params.predicts_path + "/report.txt", 'w', encoding='utf-8') as f:
        f.write(report)

