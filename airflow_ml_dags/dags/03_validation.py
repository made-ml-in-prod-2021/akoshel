import airflow.utils.dates
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from scripts import validate

with DAG(
        dag_id="03_validation.py",
        start_date=airflow.utils.dates.days_ago(1),
        schedule_interval="@daily",
        max_active_runs=1,
) as dag:
    validate_model_path = Variable.get("amazing_model_path")
    validate = PythonOperator(
        task_id="validate",
        python_callable=validate,
        op_kwargs={
            "year": "{{ execution_date.year }}",
            "month": "{{ execution_date.month }}",
            "day": "{{ execution_date.day }}",
            "config_path": "data/configs/config_lr.yml",
            "mode": "get_predicts",
            "validate_model_path": validate_model_path,
        }
    )
    validate
