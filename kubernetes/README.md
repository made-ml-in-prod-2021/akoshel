# Homework 4

1. Выбрал GKE https://cloud.google.com/kubernetes-engine

2. Ресурсы нужно ограничивать, чтобы приложение не отодрало все
ресурсы кластера. В случае превышения лимитов по
памяти контейнер будет убит, по cpu - ограничен.

3. readinessProbe - проверяет готовность приложения. В нашем случае ждет пока загрузится модель
livenessProbe -  проверяет живо ли приложение, если нет то перезапускает контейнер.
Чтобы уронить приложение необходимо дернуть за ручку /kill

4. При увеличении числа реплик с 3 до 5. 3 пода остались со старой версией, 2 добавились с новой
При уменьшении числа реплик с 5 до 2. Удалились 3 пода, а контейнеры с новой версией не выкатились
Остались поды которые были созданы в при первом поднятии

5. Из этих 2-х случаев вариант blue-green предпочтителен, так как всегда есть нужное количество работающих реплик


Пробросить порт: kubectl port-forward pod/fastapi-ml 8000:8000