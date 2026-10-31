# exp08/exp09 — полный пайплайн LISA

[Отчёт и порядок запуска](../../../experiments/exp08_cross_session/README.md) · [Протокол](../../../experiments/exp08_09_full_lisa/protocol.json)

| Скрипт | Задача |
|---|---|
| [full_lisa.py](full_lisa.py) | Общий код воспроизведения, фильтрации участников/сессий, обучения и замороженной оценки. |
| [run_full_series.py](run_full_series.py) | Запуск серий воспроизведения, межсессионного и межсубъектного переноса. |
| [summarize_full.py](summarize_full.py) | Агрегация результатов exp08/exp09, парные сравнения и статистика. |
| [vast/download_nemar.py](vast/download_nemar.py) | Загрузка полного датасета с зеркала NEMAR. |
| [vast/nemar_events_to_osf.py](vast/nemar_events_to_osf.py) | Преобразование событий NEMAR к формату входа LISA с сохранением оригиналов. |
| [vast/prepare.sh](vast/prepare.sh) | Подготовка данных авторским пайплайном и проверка канонических чисел. |
| [vast/setup.sh](vast/setup.sh) | Подготовка серверных окружений LISA и MFAligner. |
| [vast/push.sh](vast/push.sh) | Передача кода и конфигурации на сервер. |
| [vast/pull.sh](vast/pull.sh) | Получение результатов и завершённых checkpoint. |

Команды выполняются из корня репозитория по полному пути `code/scripts/exp08_09_full_lisa/имя_скрипта.py`. Общие правила и проверки — в [каталоге скриптов](../README.md).

Этот код совместно используется [exp08](../../../experiments/exp08_cross_session/README.md), [exp09](../../../experiments/exp09_new_subjects/README.md) и [контролем на 22 участниках](../exp08_matched22/README.md).
