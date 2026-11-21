# exp10 — общая матрица испытуемого

[Протокол, архитектура и серверные команды](../../../experiments/exp10_shared_matrix/README.md).

`subject_layer.py` содержит только вычисления слоя. `model.py` подключает его к
LISA без изменения upstream. `recordings.py` загружает записи отдельно для каждого
worker. `run_experiment.py` организует разбиения, обучение, валидацию, замороженный
тест и сводку. `run_server.sh` запускает команду в текущем серверном Python-окружении
с сохранением журнала; переменная `PYTHON` выбирает интерпретатор.

```bash
python code/scripts/exp10_shared_matrix/run_experiment.py tasks --fold 0
PYTHON=python bash code/scripts/exp10_shared_matrix/run_server.sh train --all --fold 0 \
  --canon /data/preprocessed --out outputs/exp10_pilot
```

Данные и CUDA-окружение должны быть подготовлены заранее. `train` не вычисляет
тестовые метрики; `eval --all` выбирает p по сохранённой валидации и затем открывает
тест. Все команды поддерживают абсолютные пути и запуск вне корня репозитория.
