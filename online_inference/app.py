import logging
from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn
from typing import List
import os
import sys
sys.path.append("..")
from ml_project.train_pipeline import get_model_and_dataprocessor
from ml_project.enities import read_training_pipeline_params

APPLICATION_NAME = 'homework02'
logger = logging.getLogger(APPLICATION_NAME)
DEFAULT_PATH_HW1 = '../homework1'


class ModelResponse(BaseModel):
    id: str
    target: float

class InputData(BaseModel):
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
    print(os.path.join(DEFAULT_PATH_HW1, "configs/config_lr.yml"))
    params = read_training_pipeline_params(os.path.join(DEFAULT_PATH_HW1, "configs/config_lr.yml"))
    params.output_model_path = os.path.join(DEFAULT_PATH_HW1, params.output_model_path)
    params.output_data_preprocessor_path  = os.path.join(DEFAULT_PATH_HW1, params.output_data_preprocessor_path)
    classifier, data_processor = get_model_and_dataprocessor(params)

@app.get("/")
def main():
    return "it is entry point of our predictor"


@app.get("/healz")
def health() -> bool:
    return not (classifier is None and data_processor is None)

@app.get("/predict/", response_model=List[ModelResponse])
def predict(request: InputData):
    return [ModelResponse(id="1", target=1)]


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=os.getenv("PORT", 8000))