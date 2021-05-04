import sys, os
sys.path.insert(0, "../ml_project")
import pytest
from data import get_train_test_data, read_csv
from enities import read_training_pipeline_params
from model_pipeline import DataProcessingPipeline, Classifier, CustomMinMaxScaler
from hypothesis import given, strategies
from hypothesis.extra.pandas import data_frames, column



@pytest.fixture
def init_data():
    data = read_csv("../data/heart.csv")
    return data


@pytest.fixture
def train_params():
    params = read_training_pipeline_params("../configs/config_lr.yml")
    return params


@pytest.fixture(scope="session")
def data_processing_model():
    init_data = read_csv("../data/heart.csv")
    train_params = read_training_pipeline_params("../configs/config_lr.yml")
    classifier = Classifier(train_params.classifier_params, train_params.model_type)
    pipeline = DataProcessingPipeline(train_params.feature_params.categorical_features,
                                      train_params.feature_params.numerical_features)
    pipeline.fit(init_data)
    transformed_data = pipeline.transform(init_data)
    classifier.fit(transformed_data, init_data['target'].values)
    return pipeline, classifier


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


def test_classifier(init_data, train_params):
    classifier = Classifier(train_params.classifier_params, train_params.model_type)
    pipeline = DataProcessingPipeline(train_params.feature_params.categorical_features,
                                      train_params.feature_params.numerical_features)
    pipeline.fit(init_data)
    transformed_data = pipeline.transform(init_data)
    classifier.fit(transformed_data, init_data['target'].values)
    predicted = classifier.predict(transformed_data)
    assert predicted.shape[0] == init_data.shape[0]


@given(data_frames([
    column('chol', dtype=float, elements=strategies.floats(min_value=100, max_value=900)),
    column('thalach', dtype=int, elements=strategies.integers(min_value=30, max_value=250)),
    column('oldpeak', dtype=float, elements=strategies.floats(min_value=0, max_value=7)),
    column('trestbps', dtype=int, elements=strategies.integers(min_value=80, max_value=250)),
    column('fbs', dtype=int, elements=strategies.integers(min_value=0, max_value=1)),
    column('age', dtype=int, elements=strategies.integers(min_value=18, max_value=120)),
    column('sex', dtype=int, elements=strategies.integers(min_value=0, max_value=1)),
    column('cp', dtype=int, elements=strategies.integers(min_value=0, max_value=3)),
    column('restecg', dtype=int, elements=strategies.integers(min_value=0, max_value=2)),
    column('exang', dtype=int, elements=strategies.integers(min_value=0, max_value=1)),
    column('slope', dtype=int, elements=strategies.integers(min_value=0, max_value=2)),
    column('ca', dtype=int, elements=strategies.integers(min_value=0, max_value=4)),
    column('thal', dtype=int, elements=strategies.integers(min_value=1, max_value=3)),
]))
def test_whole_pipeline(data_processing_model, data):
    data_processor, classifier = data_processing_model
    transformed_data = data_processor.transform(data)
    classifier.predict(transformed_data)
    assert True



def test_custom_min_max_scaler(init_data, train_params):
    scaler = CustomMinMaxScaler()
    fit_data = init_data.iloc[: init_data.shape[0] // 2]
    transform_data = init_data.iloc[init_data.shape[0] // 2 :]
    scaler.fit(fit_data[train_params.feature_params.numerical_features])
    transformed_data = scaler.transform(transform_data[train_params.feature_params.numerical_features])
    assert all(val < 1 for val in transformed_data.mean(axis=0).values)