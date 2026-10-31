# exp07 — калибровка без дообучения

[Отчёт и порядок запуска](../../../experiments/exp07_frozen_calibration/README.md) · [Протокол](../../../experiments/exp07_frozen_calibration/protocol.json)

| Скрипт | Задача |
|---|---|
| [frozen_loso_math.py](frozen_loso_math.py) | Численные методы нормировки, выбора блоков и аффинного выравнивания. |
| [frozen_loso.py](frozen_loso.py) | Проверка входов, калибровка, подбор и оценка с контролем неизменности весов. |
| [run_frozen_loso.py](run_frozen_loso.py) | Планировщик этапов и dry-run. |
| [summarize_frozen_loso.py](summarize_frozen_loso.py) | Сводка проверенных результатов и интервалы по участникам. |

Команды выполняются из корня репозитория по полному пути `code/scripts/exp07_frozen_calibration/имя_скрипта.py`. Общие правила и проверки — в [каталоге скриптов](../README.md).
