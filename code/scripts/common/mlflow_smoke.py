"""Проверка связи с MLflow-сервером. Запуск: uv run --env-file .env python code/scripts/common/mlflow_smoke.py"""
import os, platform, subprocess
import mlflow

uri = os.environ.get("MLFLOW_TRACKING_URI")
assert uri, "MLFLOW_TRACKING_URI не задан — запускай с --env-file .env"
assert os.environ.get("MLFLOW_TRACKING_PASSWORD"), "Впиши пароль в .env"

try:
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
except Exception:
    commit = "unknown"

mlflow.set_experiment("smoke")
with mlflow.start_run(run_name="hello-from-" + platform.node()):
    mlflow.log_params({"host": platform.node(), "git_commit": commit,
                       "mlflow_client": mlflow.__version__})
    mlflow.log_metric("hello", 1.0)
print(f"✅ прогон записан в {uri} (эксперимент 'smoke')")
