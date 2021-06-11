import sys
import pytest
from airflow.models import DagBag

sys.path.append("dags")


@pytest.fixture()
def dag_bag():
    return DagBag(dag_folder="dags/", include_examples=False)


def test_dag_bag_import(dag_bag):
    assert dag_bag.dags is not None
    assert dag_bag.import_errors == {}


def test_dag_generate_data_load(dag_bag):
    assert "01_generate_data.py" in dag_bag.dags
    assert len(dag_bag.dags["01_generate_data.py"].tasks) == 1


def test_dag_pipeline(dag_bag):
    assert "02_pipeline.py" in dag_bag.dags
    assert len(dag_bag.dags["02_pipeline.py"].tasks) == 4


def test_dag_predict(dag_bag):
    assert "03_validation.py" in dag_bag.dags
    assert len(dag_bag.dags["03_validation.py"].tasks) == 1


def test_dag_generate_data_structure(dag_bag):
    structure = {
        "_": "generate_data",
        "generate_data": "process_data",
        "process_data": "train",
        "train": "validate",
        "validate": "",
    }
    dag = dag_bag.dags["02_pipeline.py"]
    for name, task in dag.task_dict.items():
        if name is not "validate":
            assert set([structure[name]]) == task.downstream_task_ids
