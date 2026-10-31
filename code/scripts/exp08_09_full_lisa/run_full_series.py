"""Run exp08/exp09 on one GPU: trainings, then frozen evaluations, several processes at a time.

Finished tasks are skipped, so the series can simply be restarted after an interruption. Each job
holds a lock, so jobs already running elsewhere (another instance of this script) are waited for, not
duplicated, and count towards --procs. Nothing
except the first reproduction run starts until that run reproduces the authors' Top-10 (the gate).
A training needs about 18 GB of RAM (the whole MEG array is loaded per process); choose --procs
accordingly.

  python run_full_series.py --procs 3
  python run_full_series.py --only repro_s42        # time a single run first
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


# Support direct execution from any working directory.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.exp08_09_full_lisa.full_lisa import BUSY, OUT, find_run_dir, load_protocol, lock_holder, train_tasks

GATE_TASK = 'repro_s42'


def gate(protocol, out):
    """Top-10 of the first reproduction run, and whether it is within 2 p.p. of the authors' range."""
    run_dir = find_run_dir(GATE_TASK, out)
    top10 = 100 * float((pd.read_csv(run_dir / 'final_test_per_window.csv').rank_of_true < 10).mean())
    return top10, 70.0 - 2 <= top10 <= 70.9 + 2


def jobs(protocol, out):
    out_jobs = []
    for name in train_tasks(protocol):
        out_jobs.append({'name': f'train_{name}', 'args': ['train', '--task', name],
                         'done': lambda n=name: find_run_dir(n, out) is not None,
                         'needs': [] if name == GATE_TASK else ['gate']})
        out_jobs.append({'name': f'eval_{name}', 'args': ['eval', '--task', name],
                         'done': lambda n=name: (Path(out) / 'eval' / n / 'done.json').exists(),
                         'needs': [f'train_{name}']})
    return out_jobs


def log_mlflow(task, out):
    """Final metrics of a finished training to MLflow, if it is configured; never fails the series."""
    if not os.environ.get('MLFLOW_TRACKING_URI'):
        return
    try:
        import mlflow
        run_dir = find_run_dir(task, out)
        config = json.loads((run_dir / 'config.json').read_text())
        metrics = pd.read_csv(run_dir / 'metrics.csv')
        final = metrics.dropna(subset=['top10s_final_test']).iloc[-1]
        mlflow.set_experiment('FQW-full-LISA')
        with mlflow.start_run(run_name=task):
            mlflow.log_params({'task': task, 'seed': config['seed'], 'run_dir': run_dir.name,
                               'dirprocess': Path(config['dirprocess']).name})
            mlflow.log_metrics({'top1_final_test': float(final['top1s_final_test']),
                                'top10_final_test': float(final['top10s_final_test']),
                                'epochs': int(metrics['epoch'].max()) if 'epoch' in metrics else len(metrics)})
    except Exception as exc:  # noqa: BLE001 — logging is optional
        print(f'MLflow logging of {task} failed: {exc}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--procs', type=int, default=3)
    parser.add_argument('--only', help='run only jobs whose task name starts with this prefix')
    parser.add_argument('--tasks', help='comma-separated task names to run (all others are left alone)')
    parser.add_argument('--ignore-gate', action='store_true')
    parser.add_argument('--retries', type=int, default=1, help='re-run a failed job this many times')
    parser.add_argument('--protocol')
    parser.add_argument('--out', default=str(OUT))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--canon')
    parser.add_argument('--eval-batch-size', type=int, default=128)
    args, extra = parser.parse_known_args()  # forwarded to lisa-train, e.g. --nepoch 1 for a smoke run
    if args.eval_batch_size < 1:
        parser.error('--eval-batch-size must be positive')
    protocol = load_protocol(args.protocol) if args.protocol else load_protocol()
    common = ['--out', args.out, '--device', args.device]
    common += ['--protocol', args.protocol] if args.protocol else []
    common += ['--canon', args.canon] if args.canon else []
    logs = Path(args.out) / 'logs'
    logs.mkdir(parents=True, exist_ok=True)

    all_jobs = jobs(protocol, args.out)
    done = {j['name'] for j in all_jobs if j['done']()}
    pending = [j for j in all_jobs if j['name'] not in done
               and (not args.only or j['name'].split('_', 1)[1].startswith(args.only))
               and (not args.tasks or j['name'].split('_', 1)[1] in args.tasks.split(','))]
    running, failed, attempts = {}, [], {}
    gate_state = 'open' if args.ignore_gate else None
    print(f'{len(pending)} jobs to run', flush=True)
    while pending or running:
        for name, (proc, job, started) in list(running.items()):
            if proc.poll() is None:
                continue
            del running[name]
            if proc.returncode == BUSY:  # started elsewhere in the meantime: wait for it
                pending.append(job)
                continue
            ok = proc.returncode == 0 and job['done']()
            attempts[name] = attempts.get(name, 0) + 1
            if not ok and attempts[name] <= args.retries:
                print(f'{time.strftime("%H:%M:%S")} {name}: FAILED ({proc.returncode}), retry {attempts[name]}', flush=True)
                pending.insert(0, job)
                continue
            (done.add if ok else failed.append)(name)
            print(f'{time.strftime("%H:%M:%S")} {name}: {"ok" if ok else f"FAILED ({proc.returncode})"}, '
                  f'{(time.time() - started) / 60:.1f} min; {len(pending)} pending, {len(running)} running', flush=True)
            if ok and name.startswith('train_'):
                log_mlflow(name[len('train_'):], args.out)
        if gate_state is None and f'train_{GATE_TASK}' in done:
            top10, passed = gate(protocol, args.out)
            gate_state = 'open' if passed else 'closed'
            print(f'gate: {GATE_TASK} Top-10 = {top10:.2f}% -> {gate_state}', flush=True)
        if gate_state is None and f'train_{GATE_TASK}' not in done | set(running) | {j['name'] for j in pending}:
            gate_state = 'closed'
        if gate_state == 'closed' and any('gate' in j['needs'] for j in pending):
            print(f'Reproduction gate not passed ({GATE_TASK} failed, not run or off the authors\' range); '
                  'the other trainings are not started. Investigate, or rerun with --ignore-gate.', flush=True)
        for job in [j for j in pending if j['name'] not in running and j['done']()]:  # done elsewhere
            pending.remove(job)
            done.add(job['name'])
        ready = done | ({'gate'} if gate_state == 'open' else set())
        producible = ready | set(running) | {j['name'] for j in pending} | ({'gate'} if gate_state is None else set())
        external = {j['name'] for j in pending if lock_holder(j['name'], args.out)}
        for job in list(pending):
            if job['name'] in external:
                if job['done']():  # finished elsewhere
                    pending.remove(job)
                    done.add(job['name'])
                continue
            if any(n in failed or n not in producible for n in job['needs']):
                pending.remove(job)
                failed.append(job['name'])
                print(f'{job["name"]}: SKIPPED, a prerequisite failed or is not run', flush=True)
                continue
            if len(running) + len(external) >= args.procs or not all(n in ready for n in job['needs']):
                continue
            pending.remove(job)
            log = (logs / f'{job["name"]}.log').open('a')
            cmd = [sys.executable, 'full_lisa.py', *job['args'], *common]
            cmd += extra if job['args'][0] == 'train' else ['--eval-batch-size', str(args.eval_batch_size)]
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=Path(__file__).parent)
            running[job['name']] = (proc, job, time.time())
            print(f'{time.strftime("%H:%M:%S")} {job["name"]}: started', flush=True)
        time.sleep(5)
    print('Failed:', failed or 'none', flush=True)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
