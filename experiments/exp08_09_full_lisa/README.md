# Общий полный пайплайн exp08/exp09

Здесь хранятся материалы, совместно используемые воспроизведением LISA, межсессионным и межсубъектным переносом.

- [protocol.json](protocol.json) — полный протокол: воспроизведение, exp08 и exp09.
- [results/full_raw/](results/full_raw/) — сохранённые таблицы серии без ICA: `summary.csv`, `per_subject.csv`, `comparisons.csv`, `trainer_agreement.json`.
- [Код и серверные инструменты](../../code/scripts/exp08_09_full_lisa/README.md).
- [Отчёт exp08: перенос между сессиями](../exp08_cross_session/README.md).
- [Отчёт exp09: новые участники](../exp09_new_subjects/README.md).

Контроль на 22 участниках имеет [собственный протокол и результаты](../exp08_matched22/README.md). Таблицы полной серии хранятся здесь один раз; оба отчёта ссылаются на них.

Каталог по умолчанию у `summarize_full.py` — `results/full/` внутри этой папки. Для сохранённой raw-серии из корня репозитория передавайте `--out outputs/full_raw --results experiments/exp08_09_full_lisa/results/full_raw`.

Полные прогоны и веса остаются в `outputs/`. Серверный `vast/pull.sh` сохраняет сводки сюда и поддерживает получение результатов из прежней структуры сервера.

## Гипотеза

У этой папки нет отдельной научной гипотезы: она объединяет инфраструктуру exp08 и exp09. Проверяемое техническое ожидание — общий пайплайн сохраняет заданные фильтры участников и сессий, единый банк кандидатов и согласованность собственной оценки с тренером в пределах критериев протокола. Научные гипотезы сформулированы в README соответствующих опытов.

## Задачи

1. Подготовить и проверить канонические входы LISA, зафиксировав фактическую предобработку.
2. По одному протоколу построить задачи воспроизведения, межсессионного переноса и фолдов новых участников.
3. Выполнить обучение и замороженную оценку с контролем допустимых subject-слотов, блокировок и завершённости задач.
4. Сохранить общие таблицы и статистику по людям для exp08/exp09; обеспечить отдельный учёт matched22.

## Скрипты

Все пути ниже указаны от корня репозитория.

| Путь | Назначение |
|---|---|
| [code/scripts/exp08_09_full_lisa/full_lisa.py](../../code/scripts/exp08_09_full_lisa/full_lisa.py) | Основная реализация check/tasks/train/eval. |
| [code/scripts/exp08_09_full_lisa/run_full_series.py](../../code/scripts/exp08_09_full_lisa/run_full_series.py) | Оркестрация задач и проверка воспроизведения перед продолжением серии. |
| [code/scripts/exp08_09_full_lisa/summarize_full.py](../../code/scripts/exp08_09_full_lisa/summarize_full.py) | Таблицы и статистика обоих экспериментов. |
| [code/scripts/exp08_09_full_lisa/vast/download_nemar.py](../../code/scripts/exp08_09_full_lisa/vast/download_nemar.py) | Загрузка полного датасета. |
| [code/scripts/exp08_09_full_lisa/vast/nemar_events_to_osf.py](../../code/scripts/exp08_09_full_lisa/vast/nemar_events_to_osf.py) | Совместимый формат событий. |
| [code/scripts/exp08_09_full_lisa/vast/prepare.sh](../../code/scripts/exp08_09_full_lisa/vast/prepare.sh) | Подготовка датасета. |
| [code/scripts/exp08_09_full_lisa/vast/setup.sh](../../code/scripts/exp08_09_full_lisa/vast/setup.sh) | Подготовка серверного окружения. |
| [code/scripts/exp08_09_full_lisa/vast/push.sh](../../code/scripts/exp08_09_full_lisa/vast/push.sh) | Передача кода и конфигурации. |
| [code/scripts/exp08_09_full_lisa/vast/pull.sh](../../code/scripts/exp08_09_full_lisa/vast/pull.sh) | Получение результатов и завершённых checkpoint. |

