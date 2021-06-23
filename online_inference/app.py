import pandas as pd
import time
import logging
from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn
from typing import List
import os
import sys
import subprocess
import signal

sys.path.append("..")
from ml_project.train_pipeline import get_model_and_dataprocessor
from ml_project.enities import read_training_pipeline_params

APPLICATION_NAME = 'homework02'
logger = logging.getLogger(APPLICATION_NAME)
DEFAULT_PATH_HW1 = '../homework1'


class ModelResponse(BaseModel):
    id: int
    target: float


class InputData(BaseModel):
    id: int
    chol: float
    thalach: int
    oldpeak: float
    trestbps: int
    fbs: int
    age: int
    sex: int
    cp: int
    restecg: int
    exang: int
    slope: int
    ca: int
    thal: int


app = FastAPI()


@app.on_event("startup")
def load_model():
    global classifier, data_processor
    time.sleep(30)
    print(os.path.join(DEFAULT_PATH_HW1, "configs/config_lr.yml"))
    params = read_training_pipeline_params(os.path.join(DEFAULT_PATH_HW1, "configs/config_lr.yml"))
    params.output_model_path = os.path.join(DEFAULT_PATH_HW1, params.output_model_path)
    params.output_data_preprocessor_path = os.path.join(DEFAULT_PATH_HW1, params.output_data_preprocessor_path)
    classifier, data_processor = get_model_and_dataprocessor(params)


@app.get("/")
def main():
    return "it is entry point of our predictor"


@app.get("/healz")
def health() -> bool:
    return not (classifier is None and data_processor is None)

@app.get("/kill")
def timeout_check() -> bool:
    proc = subprocess.Popen(["pgrep", "uvi"], stdout=subprocess.PIPE)
    for pid in proc.stdout:
        os.kill(int(pid), signal.SIGTERM)
        # Check if the process that we killed is alive.
        try:
            os.kill(int(pid), 0)
            raise Exception("""wasn't able to kill the process 
                              HINT:use signal.SIGKILL or signal.SIGABORT""")
        except OSError as ex:
            continue
    return False


@app.get("/predict/", response_model=List[ModelResponse])
def predict(request: List[InputData]):
    predict_df = prepare_predict_df(request)
    return make_predict(predict_df)


def prepare_predict_df(input_data: List[InputData]) -> pd.DataFrame:
    df = pd.DataFrame(columns=InputData.__fields__.keys())
    for row in input_data:
        df = df.append(row.__dict__, ignore_index=True)
    return df


def make_predict(df: pd.DataFrame) -> List[ModelResponse]:
    transformed_data = data_processor.transform(df)
    df['target'] = classifier.predict(transformed_data)
    model_response_list = []
    for row in df.loc[:, ['id', 'target']].itertuples():
        model_response_list.append(ModelResponse(
            id=row.id,
            target=row.target,
        ))
    return model_response_list


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=os.getenv("PORT", 8000))
