"""Evaluate mean subject matrices in the existing reproduction model, without training.

Defaults match exp09 own_seen: repro_s42, the 18 subjects in folds 0-5,
all their test sessions, 1005 audio candidates, subject-weighted aggregation.
Two means are reported separately: all 27 slots, and the other 26 slots.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
# Support direct execution from any working directory.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.exp08_09_full_lisa.full_lisa import CANON, ROOT, Recordings, load_protocol, rank_windows


def digest(path):
    with open(path, 'rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--canon', type=Path, default=CANON)
    parser.add_argument('--runs', type=Path, default=ROOT / 'outputs/full_raw/runs/repro_s42')
    parser.add_argument('--out', type=Path, default=ROOT / 'outputs/mean_seen_s42')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--check-inputs', action='store_true')
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        parser.error('batch-size and threads must be positive')
    runs = sorted(args.runs.glob('*/config.json'))
    if len(runs) != 1:
        parser.error(f'Expected one config.json under {args.runs}; found {len(runs)}')
    run = runs[0].parent
    hyper = json.loads(runs[0].read_text())
    checkpoint = run / f'{hyper["checkpoint"]}.pt'
    required = [checkpoint, args.canon / 'dataframe/df_train27.csv',
                args.canon / 'dataframe/df_test27.csv',
                args.canon / 'meg/meg27_sr100.npz',
                args.canon / f'audio/extract_features_test{hyper["embedding_layers"]}.npy',
                args.canon / 'coords/sensor_xyz.npy',
                args.canon / 'coords/coords208_xy_scaled.npy']
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        print(json.dumps({'ready': False, 'missing': missing}, indent=2))
        return 2
    if args.check_inputs:
        print(json.dumps({'ready': True, 'run': str(run)}, indent=2))
        return 0

    import torch
    from scipy import stats
    torch.set_num_threads(args.threads)
    subjects = sorted(sum(load_protocol()['experiments']['exp09_new_subjects']['folds'][:6], []))
    trained = sorted(pd.read_csv(required[1]).subject_id.unique().tolist())
    if trained != list(range(1, 28)) or hyper['seed'] != 42:
        raise ValueError('This comparison requires the seed-42 reproduction trained on all 27 subjects')
    test = pd.read_csv(required[2])
    test = test[test.subject_id.isin(subjects)].copy()
    if test.wav_index.nunique() != 1005 or set(test.subject_id) != set(subjects):
        raise ValueError('Expected the full test bank and all 18 exp09 subjects')
    # The upstream loader uses coordinates from config.dirprocess. Resolve that
    # path in memory; do not alter the saved run or its config.
    from lisa.model.load_model import (
        load_from_config, build_audio_model_from_config, load_audio_model_weights,
    )
    hyper['dirprocess'] = str(args.canon.resolve())
    model, hyper = load_from_config(hyper, str(checkpoint), torch.device(args.device))
    model.eval()
    audio = load_audio_model_weights(
        build_audio_model_from_config(hyper, torch.device(args.device)),
        str(run), torch.device(args.device)).eval()
    bank = torch.tensor(np.load(required[4]), dtype=torch.float32,
                        device=args.device).permute(0, 2, 1)
    with torch.no_grad():
        bank = audio(bank)
    if len(bank) != 1005:
        raise ValueError('Expected 1005 test audio candidates')
    layer = model.spatial_module.subject_layer
    original = layer.A.detach().clone()
    indices = torch.tensor([s - 1 for s in trained], device=original.device)
    mean_all = original[indices].mean(0)
    recordings = Recordings(required[3])
    metadata = {
        'checkpoint': str(checkpoint.resolve()), 'checkpoint_sha256': digest(checkpoint),
        'seed': 42, 'subjects': subjects, 'trained_subjects': trained, 'bank_size': len(bank),
        'test_dataframe_sha256': digest(required[2]), 'device': args.device,
        'conditions': {'own_seen': 'Original own slot',
                       'mean_all_seen': 'Arithmetic mean of all 27 trained slots, including target',
                       'mean_other_seen': 'Arithmetic mean of the other 26 trained slots, excluding target'},
        'aggregation': 'windows -> subject -> equal-weight mean over the same 18 subjects as exp09',
        'training': False,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    meta_path = args.out / 'protocol.json'
    if meta_path.exists() and json.loads(meta_path.read_text()) != metadata:
        raise ValueError('Output directory contains a different evaluation protocol; use another --out')
    meta_path.write_text(json.dumps(metadata, indent=2) + '\n')
    shards = args.out / 'per_window'
    shards.mkdir(exist_ok=True)
    for subject, rows in test.groupby('subject_id'):
        rows = rows.assign(slot=subject)
        other_indices = [s - 1 for s in trained if s != subject]
        means = {'own_seen': original[subject - 1], 'mean_all_seen': mean_all,
                 'mean_other_seen': original[other_indices].mean(0)}
        try:
            for condition, matrix in means.items():
                target = shards / f'{condition}_sub{subject:02d}.csv.gz'
                if target.exists():
                    continue
                with torch.no_grad():
                    layer.A.copy_(original)
                    layer.A[subject - 1].copy_(matrix)
                result = rank_windows(model, bank, rows, recordings, hyper,
                                      args.device, batch=args.batch_size)
                tmp = target.with_suffix('.tmp')
                result.to_csv(tmp, index=False, compression='gzip')
                tmp.replace(target)
                print(f'{condition} subject={subject} rows={len(result)} '
                      f'Top-1={(result["rank"] == 0).mean():.4f} '
                      f'Top-10={(result["rank"] < 10).mean():.4f}', flush=True)
        finally:
            with torch.no_grad():
                layer.A.copy_(original)
        recordings.clear()  # avoid retaining all full recordings in RAM

    tables = []
    for condition in metadata['conditions']:
        for subject in subjects:
            rows = pd.read_csv(shards / f'{condition}_sub{subject:02d}.csv.gz')
            tables.append({'condition': condition, 'subject_id': subject,
                           'top1': float((rows['rank'] == 0).mean()),
                           'top10': float((rows['rank'] < 10).mean()),
                           'rank': float(rows['rank'].mean())})
    per_subject = pd.DataFrame(tables)
    per_subject.to_csv(args.out / 'per_subject.csv', index=False)
    summary = []
    for condition, rows in per_subject.groupby('condition'):
        for metric in ['top1', 'top10', 'rank']:
            values = rows[metric]
            summary.append({'condition': condition, 'metric': metric, 'mean': values.mean(),
                            'ci95': stats.t.ppf(.975, len(values) - 1) * stats.sem(values),
                            'n_subjects': len(values)})
    result = pd.DataFrame(summary)
    result.to_csv(args.out / 'summary.csv', index=False)
    # Verify the reference on exactly the same rows against the previous evaluation.
    reference = args.runs.parents[1] / 'eval/repro_s42/own.csv.gz'
    if not reference.exists():
        reference = run / 'final_test_per_window.csv'
    if not reference.exists():
        raise FileNotFoundError('Need saved own-slot evaluation to verify the reference')
    saved = pd.read_csv(reference)
    if 'rank' not in saved:
        raise ValueError(f'Reference {reference} has no rank column')
    saved = saved[saved.subject_id.isin(subjects)]
    ref = saved.assign(top1=saved['rank'] == 0, top10=saved['rank'] < 10)
    ref = ref.groupby('subject_id')[['top1', 'top10']].mean()
    own = per_subject[per_subject.condition == 'own_seen'].set_index('subject_id')
    differences = (own[['top1', 'top10']] - ref).abs()
    max_diff = float(differences.max().max())
    agreement = {'reference': str(reference), 'max_subject_metric_abs_diff': max_diff,
                 'passed': bool(len(ref) == len(subjects) and max_diff <= .002)}
    (args.out / 'reference_agreement.json').write_text(json.dumps(agreement, indent=2) + '\n')
    if not agreement['passed']:
        raise RuntimeError(f'Own-slot reference differs from previous evaluation: {agreement}')
    (args.out / 'done.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(result.to_string(index=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
