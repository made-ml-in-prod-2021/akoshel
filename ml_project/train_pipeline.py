import logging.config
from model_pipeline import DataProcessingPipeline, Classifier
from data import read_csv, get_train_test_data
from enities import read_training_pipeline_params
import yaml
import click


APPLICATION_NAME = 'homework01'
logger = logging.getLogger(APPLICATION_NAME)


def setup_logging(path):
    with open(path) as config_fin:
        logging.config.dictConfig(yaml.safe_load(config_fin))


def train_pipeline(params):
    data = read_csv(params.input_data_path)
    logger.info(f"data loaded, DF shape{data.shape[0]}, {data.shape[1]}")
    train_df, test_df = get_train_test_data(data, params.split_params)
    logger.info("train_df and test_df prepared")


@click.command(name="train_pipeline")
@click.argument("config_path")
def train_pipeline_command(config_path: str):
    params = read_training_pipeline_params(config_path)
    setup_logging(params.logger_config)
    train_pipeline(params)


if __name__ == "__main__":
    train_pipeline_command()
