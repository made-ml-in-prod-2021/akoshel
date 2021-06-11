# Homework3. Airflow

Самооценка
1. done +5
2. done +10
3. done +5 
4. 0
5. done +5
6. 0
7. 0
8. 0
9. done +1 <br> 
Total: 26

Тестирование
```
cd airflow_ml_dags
pytest tests
```
Поднять airflow server
```
export FERNET_KEY=$(python -c "from cryptography.fernet import Fernet; FERNET_KEY = Fernet.generate_key().decode(); print(FERNET_KEY)")
docker compose up --build
```

Чтобы запустить у себя необходимо установить пакет homework1
