from dataclasses import dataclass
from enities import SplitParams, FeatureParams, ClassifierParams
from marshmallow_dataclass import class_schema
import yaml


@dataclass
class TrainingPipelineParams:
    input_data_path: str
    output_model_path: str
    logger_config: str
    split_params: SplitParams
    feature_params: FeatureParams
    classifier_params: ClassifierParams


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_training_pipeline_params(path: str) -> TrainingPipelineParams:
    with open(path, "r") as input_stream:
        schema = TrainingPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))