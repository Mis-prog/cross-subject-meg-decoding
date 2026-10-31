# exp08 — контроль на 22 участниках

[Отчёт и порядок запуска](../../../experiments/exp08_matched22/README.md) · [Протокол](../../../experiments/exp08_matched22/protocol.json)

| Скрипт | Задача |
|---|---|
| [run_matched22.py](run_matched22.py) | Проверка совместимости старых B-моделей, новый запуск A и сводка контроля. |
| [vast/matched22.sh](vast/matched22.sh) | Серверные этапы setup/data/check/pilot/run/summary. |
| [vast/push_matched22.sh](vast/push_matched22.sh) | Передача кода и необходимых B-референсов. |
| [vast/pull_matched22.sh](vast/pull_matched22.sh) | Получение результатов, аудита и логов контроля. |

Команды выполняются из корня репозитория по полному пути `code/scripts/exp08_matched22/имя_скрипта.py`. Общие правила и проверки — в [каталоге скриптов](../README.md).

Данные, обучение и статистика переиспользуют [общий полный пайплайн](../exp08_09_full_lisa/README.md); серверные обёртки этого контроля остаются здесь.
