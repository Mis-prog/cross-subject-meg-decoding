"""Matched 22-subject session control. Reuse B only after checking it on current data.

python run_matched22.py check
python run_matched22.py run --procs 1 --seed 42  # measured pilot, no final summary
python run_matched22.py run --procs 1            # remaining seeds + final summary
python run_matched22.py summarize

Run in the same LISA environment/preprocessing as the previous full series.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


# Support direct execution from any working directory.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.exp08_09_full_lisa import full_lisa
from scripts.exp08_09_full_lisa.full_lisa import (CANON, DF_TEST, DF_TRAIN, ROOT, Recordings, check, filter_rows,
                       job_lock, load_protocol, load_run, rank_windows, train_tasks)

PROTOCOL = ROOT / 'experiments/exp08_matched22/protocol.json'
REFERENCE = ROOT / 'outputs/full_raw'
OUT = ROOT / 'outputs/full_matched22'
RESULTS = ROOT / 'experiments/exp08_matched22/results'
SEEDS = (42, 43, 44)
SUBJECTS = set(range(1, 28)) - {3, 12, 16, 20, 21}
CONDITIONS = ('own_same', 'own_cross', 'mean_cross', 'other_cross')
KEYS = ['subject_id', 'session_id', 'story_id', 'sound_id', 'wav_index']
NON_MODEL = {'device', 'dirprocess', 'meg_files_dir', 'experiments_root', 'run_group', 'run_name',
             'logging_project', 'logger', 'upload_metrics_csv', 'log_images_to_server',
             'image_formats', 'max_branches_to_plot', 'plot_filter_graphs'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def signature(config):
    return {k: v for k, v in config.items() if k not in NON_MODEL}


def reference_run(reference, seed):
    done = json.loads((reference / 'eval' / f'xsB_s{seed}' / 'done.json').read_text())
    run = (reference / done['run_dir']).resolve()
    if not run.is_relative_to((reference / 'runs').resolve()):
        raise ValueError('Reference run_dir escapes runs/')
    if not (run / 'config.json').is_file():
        raise FileNotFoundError(run / 'config.json')
    return run


def validate_rows(tr, te, protocol):
    """Same cohort and exact audio identities/counts, not merely equal total row counts."""
    tasks = train_tasks(protocol)
    counts = {}
    for label, frame in [('train_including_validation', tr), ('test', te)]:
        pairs = []
        for direction in 'AB':
            rows = filter_rows(frame, tasks[f'xs{direction}_s42'])
            if set(rows.subject_id) != SUBJECTS:
                raise ValueError(f'{label}/{direction}: expected the fixed 22 subjects')
            cols = ['subject_id', 'story_id', 'sound_id', 'wav_index']
            if rows.duplicated(cols).any():
                raise ValueError(f'{label}/{direction}: duplicate windows')
            pairs.append(rows[cols].sort_values(cols).reset_index(drop=True))
            counts[f'{label}_{direction}'] = len(rows)
        if not pairs[0].equals(pairs[1]):
            raise ValueError(f'{label}: session audio windows differ; do not silently subsample old B')
    return counts


def validate_eval(df, test, direction, condition):
    session = (0 if direction == 'A' else 1) if condition == 'own_same' else (1 if direction == 'A' else 0)
    expected = test[(test.session_id == session) & test.subject_id.isin(SUBJECTS)][KEYS]
    if condition == 'other_cross':
        expected = pd.concat([expected[expected.subject_id != s].assign(slot=s) for s in sorted(SUBJECTS)])
    else:
        expected = expected.assign(slot=expected.subject_id)
    keys = KEYS + ['slot']
    a = df[keys].sort_values(keys).reset_index(drop=True)
    b = expected[keys].sort_values(keys).reset_index(drop=True)
    if not a.equals(b):
        raise ValueError(f'{direction}/{condition}: wrong subjects, windows, sessions or donors')
    if df['rank'].isna().any() or not df['rank'].between(0, 1004).all():
        raise ValueError(f'{direction}/{condition}: invalid ranks')


def rank_agreement(current, saved):
    keys = KEYS + ['slot']
    merged = current.merge(saved, on=keys, suffixes=('_new', '_old'), validate='one_to_one')
    if len(merged) != len(current) or len(merged) != len(saved) or not len(merged):
        raise ValueError('Reference rank check: different evaluated rows')
    gap = (merged['rank_new'] - merged['rank_old']).abs()
    result = {'rows': len(merged), 'identical': float((gap == 0).mean()),
              'within_2': float((gap <= 2).mean()),
              'top10_abs_diff': abs(float((merged.rank_new < 10).mean() - (merged.rank_old < 10).mean()))}
    # .93, not .95: on RTX A4000 own_cross gave 94.2-95.2% with unchanged Top-10 (exp08_matched22.md).
    if result['identical'] < .93 or result['within_2'] < .999 or result['top10_abs_diff'] > .002:
        raise ValueError(f'Current inputs/model do not reproduce saved B ranks: {result}. '
                         'Restore the original preprocessing or plan fresh training of both directions.')
    return result


def verify_reference(canon, reference, device):
    """Functional compatibility check of old checkpoints with new/recovered arrays, both sessions."""
    import torch
    test = pd.read_csv(canon / 'dataframe' / DF_TEST)
    meg = Recordings(canon / 'meg/meg27_sr100.npz')
    reports = {}
    for seed in SEEDS:
        print(f'Checking old B seed {seed} on both sessions...', flush=True)
        model, audio, hyper = load_run(reference_run(reference, seed), device)
        bank = torch.tensor(np.load(canon / 'audio/extract_features_test4.npy'),
                            dtype=torch.float32, device=device).permute(0, 2, 1)
        with torch.no_grad():
            bank = audio(bank)
        for condition, session in [('own_same', 1), ('own_cross', 0)]:
            rows = test[(test.session_id == session) & test.subject_id.isin(SUBJECTS)]
            current = rank_windows(model, bank, rows.assign(slot=rows.subject_id), meg, hyper, device, batch=32)
            saved = pd.read_csv(reference / 'eval' / f'xsB_s{seed}' / f'{condition}.csv.gz')
            reports[f'xsB_s{seed}/{condition}'] = rank_agreement(current, saved)
        del model, audio, bank
    meg.npz.close()
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()  # release the verifier's cached VRAM before training subprocesses
    return reports


def input_manifest(canon, reference, protocol_path):
    files = [protocol_path, Path(__file__), Path(full_lisa.__file__),
             Path(full_lisa.__file__).with_name('run_full_series.py'), Path(full_lisa.__file__).with_name('summarize_full.py')]
    for folder, pattern in [('dataframe', '*.csv'), ('audio', '*4.npy'), ('coords', '*.npy')]:
        files.extend(sorted((canon / folder).glob(pattern)))
    files.append(canon / 'meg/meg27_sr100.npz')
    files.extend(sorted((ROOT / 'code/LISA/src').rglob('*.py')))
    for seed in SEEDS:
        run = reference_run(reference, seed)
        files.extend(sorted(run.glob('*.pt')))
        files.extend([run / 'config.json', run / 'final_test_per_window.csv'])
        files.extend(sorted((reference / 'eval' / f'xsB_s{seed}').glob('*')))
    return {str(p.resolve()): digest(p) for p in files if p.is_file()}


def prepare(args, protocol):
    canon, reference, out = args.canon, args.reference, args.out
    if out == reference or out.is_relative_to(reference) or reference.is_relative_to(out):
        raise ValueError('Use a separate output directory, outside the reference tree')
    report = check(canon, protocol)
    if report['problems']:
        raise ValueError(report['problems'])
    tr, te = (pd.read_csv(canon / 'dataframe' / f) for f in (DF_TRAIN, DF_TEST))
    counts = validate_rows(tr, te, protocol)
    configs = []
    for seed in SEEDS:
        run = reference_run(reference, seed)
        config = json.loads((run / 'config.json').read_text())
        if config['seed'] != seed or not (run / f'{config["checkpoint"]}.pt').is_file():
            raise ValueError(f'Invalid B config/checkpoint for seed {seed}')
        configs.append({k: v for k, v in signature(config).items() if k != 'seed'})
        old_test = pd.read_csv(run / 'final_test_per_window.csv')
        expected = te[(te.session_id == 1) & te.subject_id.isin(SUBJECTS)][KEYS]
        if not old_test[KEYS].sort_values(KEYS).reset_index(drop=True).equals(expected.sort_values(KEYS).reset_index(drop=True)):
            raise ValueError(f'B seed {seed}: trainer test rows mismatch')
        for c in CONDITIONS:
            validate_eval(pd.read_csv(reference / 'eval' / f'xsB_s{seed}' / f'{c}.csv.gz'), te, 'B', c)
    if any(c != configs[0] for c in configs[1:]):
        raise ValueError('B configurations differ across seeds')
    from lisa.cli.parse_experiment_args import parse_experiment_arguments
    candidate = parse_experiment_arguments([*protocol['training']['lisa_train_args'],
                                            '--device', args.device, '--seed', '42', '--run-group', 'xsA_s42'])
    candidate = json.loads(json.dumps(candidate, default=str))
    mismatches = [k for k, v in signature(candidate).items()
                  if k != 'seed' and (k not in configs[0] or configs[0][k] != v)]
    if mismatches:
        raise ValueError(f'Current training arguments differ from old B: {mismatches}')
    print('Hashing inputs for safe resume (large MEG array may take a while)...', flush=True)
    manifest = {'inputs': input_manifest(canon, reference, args.protocol), 'counts': counts,
                'versions': {p: importlib.metadata.version(p) for p in ('torch', 'numpy', 'scipy', 'pandas')}}
    receipt = out / 'matched22_inputs.json'
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved['manifest'] != manifest:
            raise ValueError('Inputs or code changed since preparation; use a new --out directory')
    else:
        if any((out / p).exists() for p in ('runs', 'eval')):
            raise ValueError('Existing results without matched22 receipt; use a new --out directory')
        verification = verify_reference(canon, reference, args.device)
        out.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps({'manifest': manifest, 'reference_rank_checks': verification}, indent=2))
    # Copy only checked B evaluations. No fake training completion markers or retraining of B.
    for seed in SEEDS:
        source = reference / 'eval' / f'xsB_s{seed}'
        target = out / 'eval' / source.name
        if target.exists():
            if any(not (target / p.name).exists() or digest(target / p.name) != digest(p) for p in source.iterdir() if p.is_file()):
                raise ValueError(f'Imported B results changed: {target}')
        else:
            shutil.copytree(source, target)
    print(f'Matched cohort verified: {counts}', flush=True)


def complete_summary(args, protocol):
    from scripts.exp08_09_full_lisa.summarize_full import summarize
    test = pd.read_csv(args.canon / 'dataframe' / DF_TEST)
    from scripts.exp08_09_full_lisa.full_lisa import find_run_dir
    for seed in SEEDS:
        run = find_run_dir(f'xsA_s{seed}', args.out)
        if run is None:
            raise ValueError(f'Missing completed A training for seed {seed}')
        old = json.loads((reference_run(args.reference, seed) / 'config.json').read_text())
        new = json.loads((run / 'config.json').read_text())
        if signature(old) != signature(new):
            differing = sorted(k for k in set(old) | set(new) if k not in NON_MODEL and old.get(k) != new.get(k))
            raise ValueError(f'A/B training configuration mismatch: {differing}')
        for direction in 'AB':
            folder = args.out / 'eval' / f'xs{direction}_s{seed}'
            if not (folder / 'done.json').exists():
                raise ValueError(f'Incomplete evaluation: {folder}')
            for c in CONDITIONS:
                validate_eval(pd.read_csv(folder / f'{c}.csv.gz'), test, direction, c)
    summarize(protocol, args.out, args.canon, args.results)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=['check', 'run', 'summarize'])
    parser.add_argument('--canon', type=Path, default=CANON)
    parser.add_argument('--reference', type=Path, default=REFERENCE)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--results', type=Path, default=RESULTS)
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--procs', type=int, choices=[1, 2, 3], default=1)
    parser.add_argument('--seed', type=int, choices=SEEDS, help='pilot: run only this A seed, without final summary')
    args = parser.parse_args()
    for key in ('canon', 'reference', 'out', 'results', 'protocol'):
        setattr(args, key, getattr(args, key).resolve())
    protocol = load_protocol(args.protocol)
    with job_lock('matched22_driver', args.out):
        prepare(args, protocol)
        if args.stage == 'check':
            return
        if args.stage == 'run':
            seeds = (args.seed,) if args.seed is not None else SEEDS
            cmd = [sys.executable, str(Path(full_lisa.__file__).with_name('run_full_series.py')),
                   '--protocol', str(args.protocol), '--canon', str(args.canon), '--out', str(args.out),
                   '--device', args.device, '--procs', str(args.procs), '--ignore-gate', '--retries', '0',
                   '--eval-batch-size', '32',
                   '--tasks', ','.join(f'xsA_s{s}' for s in seeds)]
            start = time.monotonic()
            proc = subprocess.Popen(cmd, start_new_session=True)
            try:
                code = proc.wait()
            except KeyboardInterrupt:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait()
                raise
            elapsed = time.monotonic() - start
            with (args.out / 'timings.jsonl').open('a') as f:
                f.write(json.dumps({'seeds': seeds, 'procs': args.procs, 'seconds': elapsed, 'returncode': code}) + '\n')
            if code:
                raise SystemExit(code)
            print(f'Training/evaluation batch: {elapsed / 60:.1f} min', flush=True)
            if args.seed is not None:
                print('Pilot complete; run again without --seed for remaining seeds and summary.')
                return
        complete_summary(args, protocol)


if __name__ == '__main__':
    main()
