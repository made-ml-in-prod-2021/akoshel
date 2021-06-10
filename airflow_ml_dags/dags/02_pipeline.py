import subprocess
import sys




import airflow.utils.dates
from airflow import DAG
from airflow.operators.python import PythonOperator
from scripts import generate_data, process_data, train_model

with DAG(
        dag_id="02_pipeline.py",
        start_date=airflow.utils.dates.days_ago(1),
        schedule_interval="@daily",
        max_active_runs=1,
) as dag:
    gen_data = PythonOperator(
        task_id="generate_data",
        python_callable=generate_data,
        op_kwargs={
            "year": "{{ execution_date.year }}",
            "month": "{{ execution_date.month }}",
            "day": "{{ execution_date.day }}",
            "output_dir": "data/raw",
        }
    )
    proc_data = PythonOperator(
        task_id="process_data",
        python_callable=process_data,
        op_kwargs={
            "year": "{{ execution_date.year }}",
            "month": "{{ execution_date.month }}",
            "day": "{{ execution_date.day }}",
            "raw_data_path": "data/raw",
            "output_dir": "data/processed",
        }
    )
    train = PythonOperator(
        task_id="train",
        python_callable=train_model,
        op_kwargs={
            "year": "{{ execution_date.year }}",
            "month": "{{ execution_date.month }}",
            "day": "{{ execution_date.day }}",
            "config_path": "data/configs/config_lr.yml",
        }
    )
    gen_data >> proc_data >> train
