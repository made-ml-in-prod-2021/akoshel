import os
import json
import pandas as pd
from typing import Optional
from ml_project.train_pipeline import get_model_and_dataprocessor, get_classification_report
from ml_project.enities import read_training_pipeline_params


def validate(year: str,
             month: str,
             day: str,
             config_path: str,
             mode: str,
             validate_model_path: Optional[str] = None,
             ) -> None:
    params = read_training_pipeline_params(config_path)
    directory = f"{params.output_model_path}/{year}/{month}/{day}"
    if not os.path.exists(directory):
        os.makedirs(directory)

    if validate_model_path is None:
        params.input_data_path += f"/{year}/{month}/{day}/val.csv"
        params.output_model_path += f"/{year}/{month}/{day}/logistic_regression.pkl"
        params.output_data_preprocessor_path += f"/{year}/{month}/{day}/data_preprocessor_lr.pkl"
    else:
        params.input_data_path = f"data/raw/{year}/{month}/{day}/data.csv"
        params.output_model_path = f"{validate_model_path}/logistic_regression.pkl"
        params.output_data_preprocessor_path = f"{validate_model_path}/data_preprocessor_lr.pkl"

    classifier, data_processor = get_model_and_dataprocessor(params)
    val_df = pd.read_csv(params.input_data_path)
    transformed_data = data_processor.transform(val_df)
    y_pred = classifier.predict(transformed_data)
    if mode == "get_metrics":
        report = get_classification_report(val_df['target'].values, y_pred)
        params.predicts_path += f"/{year}/{month}/{day}"
        if not os.path.exists(params.predicts_path):
            os.makedirs(params.predicts_path)
        with open(params.predicts_path + "/report.txt", 'w', encoding='utf-8') as f:
            f.write(report)
    elif mode == "get_predicts":
        params.predicts_path = f"data/predicts/{year}/{month}/{day}"
        if not os.path.exists(params.predicts_path):
            os.makedirs(params.predicts_path)
        val_df["predicted"] = y_pred
        val_df[["predicted"]].to_csv(params.predicts_path + "/predicts.csv")
    else:
        raise NotImplementedError

