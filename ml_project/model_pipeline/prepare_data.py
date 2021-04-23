from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from typing import List
import pandas as pd


class DataProcessingPipeline:

    def __init__(self, categorical_features: List[str], numerical_features: List[str]):
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.pipeline = None

    @staticmethod
    def _build_categorical_pipeline(categorical_features) -> Pipeline:
        categorical_pipeline = Pipeline(
            [
                ('selection', FunctionTransformer(lambda x: x[categorical_features], validate=False)),
                ("ohe", OneHotEncoder()),
            ]
        )
        return categorical_pipeline

    @staticmethod
    def _build_numerical_pipeline(numerical_features) -> Pipeline:
        numerical_pipeline = Pipeline(
            [
                ('selection', FunctionTransformer(lambda x: x[numerical_features], validate=False)),
                ("scaler", StandardScaler()),
            ]
        )
        return numerical_pipeline

    def fit(self, data: pd.DataFrame):
        self.pipeline = Pipeline(steps=[
            ('feature_processing', FeatureUnion(
                transformer_list=[
                    ('numeric_processing', self._build_numerical_pipeline(self.numerical_features)),
                    ('category_processing', self._build_categorical_pipeline(self.categorical_features)),
                ]
            ))
        ])
        self.pipeline.fit(data)

    def transform(self, data: pd.DataFrame):
        return self.pipeline.transform(data)


