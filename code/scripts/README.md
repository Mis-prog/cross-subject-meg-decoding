# Скрипты экспериментов

Каждый скрипт находится в папке своего эксперимента. Номера соответствуют отчётам в `experiments/`; имена файлов сохранены. Общий полный пайплайн exp08/exp09 хранится один раз в `exp08_09_full_lisa/`, инфраструктурная проверка MLflow — в `common/`.

## Каталог

| Папка | Назначение | Скриптов |
|---|---|---:|
| [exp00_lisa_readiness/](exp00_lisa_readiness/README.md) | exp00 — готовность LISA | 2 |
| [exp03_real_data_pilot/](exp03_real_data_pilot/README.md) | exp03 — пилот на реальных данных | 4 |
| [exp04_small_baseline/](exp04_small_baseline/README.md) | exp04 — небольшой baseline | 5 |
| [exp05_mini_loso/](exp05_mini_loso/README.md) | exp05 — mini-LOSO | 3 |
| [exp06_loso/](exp06_loso/README.md) | exp06 — LOSO на 27 участниках | 7 |
| [exp07_frozen_calibration/](exp07_frozen_calibration/README.md) | exp07 — калибровка без дообучения | 4 |
| [exp08_09_full_lisa/](exp08_09_full_lisa/README.md) | exp08/exp09 — полный пайплайн LISA | 9 |
| [exp08_matched22/](exp08_matched22/README.md) | exp08 — контроль на 22 участниках | 4 |
| [exp09_new_subjects/](exp09_new_subjects/README.md) | exp09 — перенос на новых участников | 1 |
| [common/](common/README.md) | Общие инструменты | 1 |

Для exp01/exp02 пока есть постановки и планы; отдельных скриптов этих этапов в рабочем дереве нет. Воспроизведение авторской модели входит в полный пайплайн exp08/exp09.

## Запуск

Команды в документации выполняются из корня репозитория, например:

```sh
.venv/bin/python code/scripts/exp00_lisa_readiness/smoke_lisa.py --output /tmp/fqw_lisa_smoke.json
.venv/bin/python code/scripts/exp07_frozen_calibration/run_frozen_loso.py --stage all --dry-run
.venv/bin/python code/scripts/exp08_09_full_lisa/full_lisa.py tasks
```

`--help` поддерживается скриптами с CLI; подготовительные скрипты без аргументов запускают работу сразу. Серверные команды и порядок полного эксперимента приведены в соответствующих отчётах и README папок.

## Зависимости между опытами

- exp04 использует загрузчик exp03.
- exp05 использует загрузку exp03 и подготовку exp04.
- exp06 использует проверенные функции exp03–exp05.
- exp08_matched22 и дополнительная оценка exp09 используют `exp08_09_full_lisa/`.

Импорты явно указывают пакет эксперимента (`scripts.exp06_loso`, например). Прямой запуск файла поддерживается без ручного задания `PYTHONPATH`. При добавлении нового скрипта выбирайте папку эксперимента и добавляйте его в её README.

Протоколы и небольшие результаты находятся в `experiments/expNN_*/protocol.json` и `results/` соответствующего опыта; [каталог экспериментов](../../experiments/README.md) связывает их с кодом. Массивы и checkpoint хранятся в `data/` и `outputs/`. Сохранённые отчёты и provenance-манифесты отражают исходные запуски. Контроль происхождения в exp07/matched22 учитывает код и его расположение: при несовпадении с прежним прогоном используйте отдельный каталог результатов (`--output-dir` / `--out`), сохраняя старый.

## Проверка

```sh
.venv/bin/python -m pytest tests
.venv/bin/python -m unittest discover -s tests -p 'test_frozen_loso_math.py'
```

Тесты находятся в корневой `tests/`. Полный smoke-тест использует синтетические данные и локальный checkout LISA; обучение на реальных данных не требуется.
