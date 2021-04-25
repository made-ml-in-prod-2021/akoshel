# akoshel

## Машинное обучение в продакшене
### Студент DS-21 Кошелев Александр 
https://data.mail.ru/profile/ale.koshelev/

## ДЗ № 1

Данные для модели были взяты отсюда https://www.kaggle.com/ronitf/heart-disease-uci <br>
Для обработки данных построен пайплайн:<br>
1. Для числовых признаков применена нормализация<br>
2. Для категориальных признаков OneHotEncoding

Для классификации построены следующие модели:
1. Logistic Regression
2. Random Forest Classifier

Для обучения логистической регрессии
```
python ml_project/train_pipeline.py configs/config_lr.yml train
```
Для обучения случайного леса
```
python ml_project/train_pipeline.py configs/config_rf.yml train
```

После обучения моделей обученные пайплайн обработки данных и классификатор выгружаются в /models<br>
После этого можно запустить модель в режиме валидации
 ```
python ml_project/train_pipeline.py configs/config_lr.yml validate
```
```
python ml_project/train_pipeline.py configs/config_rf.yml validate
```