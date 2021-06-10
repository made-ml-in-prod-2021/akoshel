ARG AIRFLOW_BASE_IMAGE="apache/airflow:2.0.0-python3.6"
FROM ${AIRFLOW_BASE_IMAGE}

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install --user --no-cache-dir \
    apache-airflow-providers-docker==1.2.0

COPY ml_project-0.1.0.tar.gz ml_project-0.1.0.tar.gz
RUN pip install ml_project-0.1.0.tar.gz


USER root
ENV PYTHONPATH=/home/airflow/.local/lib/python3.6/site-packages
