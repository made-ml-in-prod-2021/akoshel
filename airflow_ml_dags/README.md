#HW3 Airflow 


Для локального запуска:


```
cd airflow_ml_dags
pip install -r requirement
```
Запуск тестов
```
pytest -v
```

Не успел выложить модель с hw1 в pypi. Поэтому скачайте архив и положите в airflow_ml_dahs/images/airflow-docker <br>
https://disk.yandex.ru/d/WdKe9Q9cnCHoxQ

Скачайте папку data и положите ее airflow_ml_dags
https://disk.yandex.ru/d/AXKJMJJ1oSm93A

Поднять airflow server
```
# для корректной работы с переменными, созданными из UI
export FERNET_KEY=$(python -c "from cryptography.fernet import Fernet; FERNET_KEY = Fernet.generate_key().decode(); print(FERNET_KEY)")
docker compose up --build
```


```
В airflow variables установить переменную
key: amazing_model_path, 	value=data/models/amazing_model
```

Короч я не понял, что можно 