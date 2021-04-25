# akoshel

## Машинное обучение в продакшене
### Студент DS-21 Кошелев Александр 
https://data.mail.ru/profile/ale.koshelev/

## ДЗ № 1

Сначала установите необходимые пакеты
```
pip install -r requirements.txt
```

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
После этого предикты запишутся в model/predicts.csv

Для тестирования модели со статистикой покрытия тестами
```
cd ml_project
pytest ../tests --cov
```

Самоанализ<br>
-2)) done<br>
-1)) done<br>
0)) done<br>
1)) Датасет простой, в целом для его изучения было достаточно отчета из pandas_profiling<br>
2)) done<br>
3)) done<br>
4)) done<br>
5)) done. Использовано hypothesis<br>
6)) done. 2 конфига для лр и рф<br>
7)) done<br>
8)) не сделано, тк. не знаю как сдампить кастомый трансформер для последующей валидации<br>
9)) done<br>
10)) done <br>
11)) пока что -<br>
12)) пока что - <br>
13)) done<br>