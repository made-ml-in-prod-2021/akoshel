FROM python:3.7

COPY /requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

COPY /homework1/dist/ml_project-0.1.0.tar.gz /ml_project-0.1.0.tar.gz
RUN pip install /ml_project-0.1.0.tar.gz

COPY /homework1/configs/ homework1/configs
COPY /homework1/data homework1/data
COPY /homework1/models homework1/models
COPY /online_inference /online_inference
#RUN mkdir /homework1/models


#RUN cd /homework1 && ml_project_train configs/config_lr.yml train
#RUN cd /homework1 && ml_project_train configs/config_rf.yml train

WORKDIR /online_inference
ENV APP_PORT=8000

ENV LOG_LEVEL=info

ENV START_DELAY=10

ENV WORK_TIME=120


#CMD ["bash", "-c", "sleep ${START_DELAY}; (uvicorn app:app --host 0.0.0.0 --port ${APP_PORT}) & sleep ${WORK_TIME}; kill $!" ]
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]