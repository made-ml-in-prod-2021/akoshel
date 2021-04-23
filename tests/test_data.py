import pytest
from ml_project.data import get_train_test_data, read_csv
from ml_project.enities import read_training_pipeline_params
from ml_project.model_pipeline import DataProcessingPipeline


@pytest.fixture
def init_data():
    data = read_csv("data/heart.csv")
    return data


@pytest.fixture
def train_params():
    params = read_training_pipeline_params("configs/config.yml")
    return params


def test_get_train_test_data(init_data, train_params):
    train, test = get_train_test_data(init_data, train_params.split_params)
    assert not train.empty
    assert not test.empty


def test_data_processing_pipeline(init_data, train_params):
    pipeline = DataProcessingPipeline(train_params.feature_params.categorical_features,
                           train_params.feature_params.numerical_features)
    pipeline.fit(init_data)
    transformed_data = pipeline.transform(init_data)
    assert init_data.shape[0] == transformed_data.shape[0]
    assert init_data.shape[1] <= transformed_data.shape[1]