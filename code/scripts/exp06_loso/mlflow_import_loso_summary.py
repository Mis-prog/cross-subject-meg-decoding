"""Import existing exp06 aggregates into MLflow; no model training.

Preview: .venv/bin/python code/scripts/exp06_loso/mlflow_import_loso_summary.py
Publish: .venv/bin/python code/scripts/exp06_loso/mlflow_import_loso_summary.py --publish --metrics-only
Credentials are loaded from the project .env. No files or artifacts are uploaded.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[3]
SUMMARY = ROOT / 'experiments/exp06_loso/results/summary.csv'
PROTOCOL = ROOT / 'experiments/exp06_loso/protocol.json'
EXPERIMENT = 'FQW-exp06-LOSO'
METHODS = ('mean_Ws', 'ft_Ws', 'ft_lowrank', 'ft_all', 'ea_ft_Ws')
METRICS = {
    'top1_pct': ('top1', 100), 'top10_pct': ('top10', 100),
    'top1_ci95_halfwidth_pp': ('top1_ci95', 100),
    'top10_ci95_halfwidth_pp': ('top10_ci95', 100),
    'mean_rank': ('mean_rank', 1), 'mean_rank_ci95_halfwidth': ('mean_rank_ci95', 1),
    'clip_loss': ('loss', 1), 'clip_loss_ci95_halfwidth': ('loss_ci95', 1),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publish', action='store_true')
    parser.add_argument('--metrics-only', action='store_true', help='Explicitly publish numerical aggregate points only; no files')
    args = parser.parse_args()
    if args.publish and not args.metrics_only:
        parser.error('Publishing requires --metrics-only; file uploads are not supported')
    with SUMMARY.open() as handle:
        rows = list(csv.DictReader(handle))
    protocol = json.loads(PROTOCOL.read_text())
    selected = {method: sorted([r for r in rows if r['method'] == method], key=lambda r: int(r['budget_seconds'])) for method in METHODS}
    for method, values in selected.items():
        budgets = [int(r['budget_seconds']) for r in values]
        if not values or len(budgets) != len(set(budgets)):
            raise ValueError(f'Missing or duplicate budget rows for {method}')
        for row in values:
            if int(row['subjects']) != 24 or not all(math.isfinite(float(row[k])) for k, _ in METRICS.values()):
                raise ValueError(f'Invalid summary row for {method}')
    print(json.dumps({'experiment': EXPERIMENT, 'series': {k: [int(r['budget_seconds']) for r in v] for k, v in selected.items()}, 'metrics_per_point': len(METRICS)}, indent=2))
    if not args.publish:
        preview = {
            'experiment': EXPERIMENT, 'step_unit': 'calibration_seconds',
            'artifact_uploads': [],
            'points': [{'method': method, 'step': int(row['budget_seconds']),
                        'metrics': {name: float(row[column]) * factor for name, (column, factor) in METRICS.items()}}
                       for method, values in selected.items() for row in values],
        }
        path = ROOT / 'experiments/exp06_loso/results/mlflow_pending_points.json'
        path.write_text(json.dumps(preview, indent=2) + '\n')
        print('Prepared local preview:', path.relative_to(ROOT))
        return

    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env')
    uri = os.environ.get('MLFLOW_TRACKING_URI', '')
    if not uri or not os.environ.get('MLFLOW_TRACKING_PASSWORD'):
        raise RuntimeError('MLflow URI or password missing from environment/.env')
    os.environ.setdefault('MLFLOW_DISABLE_AGENT_HINT', '1')
    os.environ.setdefault('MLFLOW_HTTP_REQUEST_TIMEOUT', '15')
    os.environ.setdefault('MLFLOW_HTTP_REQUEST_MAX_RETRIES', '1')
    from mlflow.entities import Metric, Param
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name(EXPERIMENT)
    experiment_id = experiment.experiment_id if experiment else client.create_experiment(EXPERIMENT)
    fingerprint = hashlib.sha256(SUMMARY.read_bytes() + PROTOCOL.read_bytes()).hexdigest()
    results = []
    for method, values in selected.items():
        import_id = f'metrics-only-v1-{fingerprint}-{method}'
        existing = client.search_runs([experiment_id], filter_string=f"tags.import_id = '{import_id}'", max_results=2)
        if len(existing) > 1:
            raise RuntimeError(f'Multiple existing imports for {method}')
        if existing:
            run_id = existing[0].info.run_id
            if existing[0].info.status != 'FINISHED':
                raise RuntimeError(f'Incomplete import for {method}: {run_id}; inspect before retrying')
        else:
            run = client.create_run(experiment_id, run_name=f'exp06-summary-{method}', tags={
                'import_id': import_id, 'method': method, 'source': 'existing_local_results',
                'run_kind': 'aggregate_import', 'metric_step': 'calibration_budget_seconds',
                'source_summary_sha256': hashlib.sha256(SUMMARY.read_bytes()).hexdigest(),
                'mlflow.note.content': (
                    'Импорт сохранённых результатов exp06 от 2026-09-17; новое обучение не запускалось. '
                    'Step = бюджет калибровки в секундах. Top-1/Top-10 в процентах; CI95 — полуширина интервала. '
                    '24 человека, 3 сида, сначала усреднение повторов/сидов внутри человека. '
                    'Банк из 99 кандидатов, две знакомые истории одной сессии; ICA по полной записи. '
                    'Результаты не сопоставимы напрямую с полной оценкой статьи LISA.'
                ),
            })
            run_id = run.info.run_id
            try:
                now = int(time.time() * 1000)
                metrics = [Metric(name, float(row[column]) * factor, now, int(row['budget_seconds']))
                           for row in values for name, (column, factor) in METRICS.items()]
                params = {'method': method, 'subjects': 24, 'seeds': ','.join(map(str, protocol['seeds'])),
                          'candidate_count': 99, 'metric_step_unit': 'seconds',
                          'metric_accuracy_unit': 'percent', 'source_experiment': 'exp06'}
                client.log_batch(run_id, metrics=metrics, params=[Param(k, str(v)) for k, v in params.items()], synchronous=True)
                client.set_terminated(run_id, status='FINISHED')
            except Exception:
                client.set_terminated(run_id, status='FAILED')
                raise
        for name, (column, factor) in METRICS.items():
            history = client.get_metric_history(run_id, name)
            expected = {int(row['budget_seconds']): float(row[column]) * factor for row in values}
            if len(history) != len(expected) or any(p.step not in expected or not math.isclose(p.value, expected[p.step], rel_tol=1e-12, abs_tol=1e-12) for p in history):
                raise RuntimeError(f'Metric verification failed: {method}/{name}')
        results.append({'method': method, 'run_id': run_id, 'budget_points': len(values), 'verified': True})
        print(json.dumps(results[-1]), flush=True)
    parsed = urlsplit(uri)
    public_url = f'{parsed.scheme}://{parsed.hostname}' + (f':{parsed.port}' if parsed.port else '') + parsed.path.rstrip('/')
    receipt = {'experiment_id': experiment_id, 'experiment_url': f'{public_url}/#/experiments/{experiment_id}',
               'source_fingerprint': fingerprint, 'runs': results}
    (ROOT / 'experiments/exp06_loso/results/mlflow_summary_import.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
