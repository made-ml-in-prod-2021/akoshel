FROM python:3.7

COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

COPY configs/ /configs
COPY data /data
RUN mkdir /models

COPY dist/ml_project-0.1.0.tar.gz /ml_project-0.1.0.tar.gz
RUN pip install /ml_project-0.1.0.tar.gz
WORKDIR .

CMD ml_project_train configs/config_lr.yml train
CMD ml_project_train configs/config_rf.yml train