# exp04 — небольшой baseline

[Отчёт и порядок запуска](../../../experiments/exp04_small_baseline/README.md) · [Протокол](../../../experiments/exp04_small_baseline/protocol.json)

| Скрипт | Задача |
|---|---|
| [download_baseline.py](download_baseline.py) | Загрузка фиксированного поднабора одного участника. |
| [prepare_baseline.py](prepare_baseline.py) | Подготовка окон и разбиений; проверки и хеширование переиспользуются в exp05/exp06. |
| [train_baseline.py](train_baseline.py) | Обучение модели с выбором по validation. |
| [test_baseline.py](test_baseline.py) | Проверка целостности разбиений и retrieval; это отдельная проверка эксперимента, а не тест из корневого tests/. |
| [plot_meg_example.py](plot_meg_example.py) | Иллюстрация подготовленной пары МЭГ/аудио. |

Команды выполняются из корня репозитория по полному пути `code/scripts/exp04_small_baseline/имя_скрипта.py`. Общие правила и проверки — в [каталоге скриптов](../README.md).
