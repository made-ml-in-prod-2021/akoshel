## Homework 2

### Идеи

Решил не оставлять результаты дз № 1, поэтому собрал это в пакет и использовал в дз № 2
В образ не передаю готовую предобученную модель, а подготавливаю ее там же при сборке докер образа. 
Каких-то конкретных целей это не приследует, просто хотел поэкспериментировать

### Запуск и тестирование приложения
Для использования тестирования приложения сначала установите необходимые зависимости
```
pip install -r requirements.txt
```
Скачайте докер образ 
```
docker pull akoshelev/online-inference:v1
```
Запустите докер образ
```
docker run -d --name online_inference -p 8000:8000 akoshelev/online-inference:v1
```

Для проверки работы приложения запустите скрипт с запросами make_request.py

```
cd online_inference
python make_request.py
```

Для просмотра логов контейнера выполните команду

```
docker logs online_inference
```

### Подсказки для себя
Для сборки контейнера выполните команду

```
docker build -t akoshelev/online-inference:v1 -f online_inference/Dockerfile .
```

Для локального запуска приложение
```
cd online_inference
uvicorn app:app --host 0.0.0.0
```


Для запуска тестов
```
cd online_inference
pytest --cov
```

### Самоанализ
1. done +3
2. done +3
3. done +2
4. --
5. done +4
6. Посмотрел по ссылке, показалось что оптимально сделал. На ваше усмотрение :-) ?+3
7. done +2
8. done +1
9. done +1
<br>
Итого: 16(19)
