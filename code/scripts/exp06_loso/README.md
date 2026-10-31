# exp06 — LOSO на 27 участниках

[Отчёт и порядок запуска](../../../experiments/exp06_loso/README.md) · [Протокол](../../../experiments/exp06_loso/protocol.json)

| Скрипт | Задача |
|---|---|
| [download_loso.py](download_loso.py) | Загрузка записей и ICA для серии. |
| [prepare_loso.py](prepare_loso.py) | Подготовка окон, разбиений и нормировки. |
| [train_loso.py](train_loso.py) | Обучение общих моделей, подбор и оценка калибраторов. |
| [run_loso_series.py](run_loso_series.py) | Планировщик всей серии на одной GPU. |
| [summarize_loso.py](summarize_loso.py) | Агрегация по участникам, статистика и графики. |
| [diagnose_mean_ws.py](diagnose_mean_ws.py) | Диагностика средней матрицы и влияния нормировки. |
| [mlflow_import_loso_summary.py](mlflow_import_loso_summary.py) | Предпросмотр и явная публикация агрегатов exp06 в MLflow. |

Команды выполняются из корня репозитория по полному пути `code/scripts/exp06_loso/имя_скрипта.py`. Общие правила и проверки — в [каталоге скриптов](../README.md).
