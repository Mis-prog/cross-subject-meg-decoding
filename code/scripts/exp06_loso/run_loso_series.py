"""Run the whole LOSO series on one GPU: model fits, tuning on dev subjects, calibration.

Jobs run as subprocesses, several at a time; a job starts once the files it needs exist and is
skipped if its output already exists, so the series can be restarted after an interruption.

  python run_loso_series.py --procs 4
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


# Support direct execution from any working directory.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.exp06_loso.prepare_loso import PROTOCOL
from scripts.exp06_loso.train_loso import RES, TUNE_LRS, model_path

LOGS = Path(__file__).resolve().parents[3] / 'logs/loso'


def jobs():
    protocol = json.loads(PROTOCOL.read_text())
    dev, seeds = protocol['dev_subjects'], protocol['seeds']
    evaluated = [s for s in protocol['subjects'] if s not in dev]
    out = []

    def fit(kind, holdout, seed):
        output = model_path(kind, holdout, seed)[1]
        args = ['fit', '--model', kind, '--seed', str(seed)] + ([] if holdout is None else ['--holdout', str(holdout)])
        out.append({'name': output.stem, 'args': args, 'output': output, 'needs': []})

    for h in dev:
        fit('shared', h, 0)
        fit('shared_ea', h, 0)
    for h in dev:
        for method in TUNE_LRS:
            kind = 'shared_ea' if method.startswith('ea_') else 'shared'
            out.append({'name': f'tune_{method}_h{h:02d}', 'args': ['tune', '--holdout', str(h), '--method', method],
                        'output': RES / 'tuning' / f'{method}_h{h:02d}.csv',
                        'needs': [model_path(kind, h, 0)[1]]})
    out.append({'name': 'choose', 'args': ['choose'], 'output': RES / 'tuning.json',
                'needs': [j['output'] for j in out if j['name'].startswith('tune_')]})
    for seed in seeds:
        fit('ceiling', None, seed)
    for seed in seeds:
        for h in evaluated:
            fit('shared', h, seed)
            fit('shared_ea', h, seed)
            out.append({'name': f'calibrate_h{h:02d}_s{seed}', 'args': ['calibrate', '--holdout', str(h), '--seed', str(seed)],
                        'output': RES / 'calibration' / f'h{h:02d}_s{seed}.csv',
                        'needs': [RES / 'tuning.json', model_path('shared', h, seed)[1], model_path('shared_ea', h, seed)[1]]})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--procs', type=int, default=4)
    parser.add_argument('--only', help='run only jobs whose name starts with this prefix')
    args = parser.parse_args()
    LOGS.mkdir(parents=True, exist_ok=True)
    pending = [j for j in jobs() if not j['output'].exists() and (not args.only or j['name'].startswith(args.only))]
    running, failed, lost = {}, [], set()

    def give_up(job, reason):
        pending.remove(job)
        failed.append(job['name'])
        lost.add(job['output'])
        print(f'{job["name"]}: {reason}', flush=True)

    print(f'{len(pending)} jobs to run', flush=True)
    while pending or running:
        for name, (proc, job, started) in list(running.items()):
            if proc.poll() is None:
                continue
            del running[name]
            ok = proc.returncode == 0 and job['output'].exists()
            if not ok:
                failed.append(name)
                lost.add(job['output'])
            print(f'{time.strftime("%H:%M:%S")} {name}: {"ok" if ok else f"FAILED ({proc.returncode})"}, '
                  f'{time.time() - started:.0f} s; {len(pending)} pending, {len(running)} running', flush=True)
        producible = {j['output'] for j in pending} | {j['output'] for _, j, _ in running.values()}
        for job in list(pending):
            missing = [p for p in job['needs'] if not p.exists()]
            if any(p in lost or p not in producible for p in missing):
                give_up(job, 'SKIPPED, an input failed or is not produced by any job')
                continue
            if missing or len(running) >= args.procs:
                continue
            pending.remove(job)
            log = (LOGS / f'{job["name"]}.log').open('w')
            proc = subprocess.Popen([sys.executable, 'train_loso.py', *job['args']], stdout=log, stderr=subprocess.STDOUT,
                                    cwd=Path(__file__).parent)
            running[job['name']] = (proc, job, time.time())
        time.sleep(5)
    print('Failed:', failed or 'none', flush=True)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
