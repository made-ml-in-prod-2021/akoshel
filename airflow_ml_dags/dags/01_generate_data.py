import os
import json

import airflow.utils.dates
from airflow import DAG
from airflow.models import TaskInstance
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from generate_data import generate_data


with DAG(
    dag_id="01_generate_data.py",
    start_date=airflow.utils.dates.days_ago(1),
    schedule_interval="@daily",
    max_active_runs=1,
) as dag:
    get_data = PythonOperator(
        task_id="get_data",
        python_callable=generate_data,
        op_kwargs={
            "year": "{{ execution_date.year }}",
            "month": "{{ execution_date.month }}",
            "day": "{{ execution_date.day }}",
            "hour": "{{ execution_date.hour }}",
            "output_dir": "data/raw",
        }
    )
