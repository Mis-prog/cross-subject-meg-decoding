"""exp08/exp09 on the authors' full LISA pipeline: data check, filtered training, frozen evaluation.

The authors' preprocessed data, model and trainer are used unchanged. A task only filters the rows
of df_train27.csv / df_test27.csv (by subject and session) into its own dirprocess, whose audio/,
meg/ and coords/ are symlinks to the canonical ones. Evaluation never trains: it loads the best
checkpoint, optionally replaces one subject-layer slot, and ranks every test window against the
same bank of 1005 test audio windows.

  python full_lisa.py check                    # canonical counts after prepare_dataset.sh
  python full_lisa.py tasks                    # list training and evaluation tasks
  python full_lisa.py train --task xsA_s42
  python full_lisa.py eval --task xsA_s42
"""
import argparse
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT / 'experiments/exp08_09_full_lisa/protocol.json'
CANON = ROOT / 'code/LISA/data/preprocessed'
OUT = ROOT / 'outputs/full'
DF_TRAIN, DF_TEST = 'df_train27.csv', 'df_test27.csv'
VAL_SOUND = (3, 4)
BUSY = 75  # exit code: the job is already running in another process
STRIDE_SAMPLES = 16000  # 1 s of 16 kHz audio between neighbouring windows


def load_protocol(path=PROTOCOL):
    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------- tasks

def train_tasks(protocol):
    """name -> row filter. sessions=None keeps both; exclude lists subjects removed from training."""
    exps = protocol['experiments']
    tasks = {}
    for seed in exps['exp08_reproduction']['seeds']:
        tasks[f'repro_s{seed}'] = {'sessions': None, 'exclude': [], 'seed': seed}
    for direction, spec in exps['exp08_cross_session']['directions'].items():
        for seed in exps['exp08_cross_session']['seeds']:
            tasks[f'xs{direction}_s{seed}'] = {'sessions': [spec['train_session']],
                                             'exclude': list(exps['exp08_cross_session'].get('exclude_subjects', [])),
                                             'seed': seed}
    for k, fold in enumerate(exps['exp09_new_subjects']['folds']):
        for seed in exps['exp09_new_subjects']['seeds']:
            tasks[f'ns_f{k}_s{seed}'] = {'sessions': None, 'exclude': list(fold), 'seed': seed}
    return tasks


def filter_rows(df, spec):
    keep = ~df.subject_id.isin(spec['exclude'])
    if spec['sessions'] is not None:
        keep &= df.session_id.isin(spec['sessions'])
    return df[keep].reset_index(drop=True)


def task_dirprocess(name, spec, canon=CANON, out=OUT):
    """Canonical dir for the unfiltered task, otherwise a dir with filtered dataframes + symlinks."""
    if spec['sessions'] is None and not spec['exclude']:
        return Path(canon)
    target = Path(out) / 'dirprocess' / name
    (target / 'dataframe').mkdir(parents=True, exist_ok=True)
    for sub in ('audio', 'meg', 'coords'):
        link = target / sub
        if not link.exists():
            link.symlink_to(Path(canon).resolve() / sub, target_is_directory=True)
    for fname in (DF_TRAIN, DF_TEST):
        df = filter_rows(pd.read_csv(Path(canon) / 'dataframe' / fname), spec)
        if df.empty:
            raise ValueError(f'{name}: filter leaves {fname} empty')
        df.to_csv(target / 'dataframe' / fname, index=False)
    return target


def find_run_dir(name, out=OUT):
    """The completed run dir of a training task, or None. Completion = filters_*.npz written last."""
    group = Path(out) / 'runs' / name
    done = sorted(p.parent for p in group.glob('*/filters_*.npz'))
    if len(done) > 1:
        raise RuntimeError(f'{name}: several completed runs {done}; keep one')
    return done[0] if done else None


# ---------------------------------------------------------------- locks

def lock_holder(job, out=OUT):
    """PID of the live process holding the job's lock, or None (a stale lock is removed)."""
    path = Path(out) / 'locks' / f'{job}.lock'
    try:
        pid = int(path.read_text())
        os.kill(pid, 0)
        stat = Path(f'/proc/{pid}/stat')
        if stat.exists() and stat.read_text().rsplit(')', 1)[1].split()[0] == 'Z':  # exited, not reaped
            raise ProcessLookupError
        return pid
    except (FileNotFoundError, ValueError):
        return None
    except ProcessLookupError:
        path.unlink(missing_ok=True)
        return None


@contextmanager
def job_lock(job, out=OUT):
    """Run a job in at most one process, even across several run_full_series.py instances."""
    path = Path(out) / 'locks' / f'{job}.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            if lock_holder(job, out):
                print(f'{job}: already running in process {lock_holder(job, out)}')
                sys.exit(BUSY)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


# ---------------------------------------------------------------- check

def check(canon=CANON, protocol=None, strict=True):
    """Verify the prepared files and the split assumptions the experiments rely on."""
    canon = Path(canon)
    protocol = protocol or load_protocol()
    tr = pd.read_csv(canon / 'dataframe' / DF_TRAIN)
    te = pd.read_csv(canon / 'dataframe' / DF_TEST)
    val = (tr.story_id == VAL_SOUND[0]) & (tr.sound_id == VAL_SOUND[1])
    counts = {
        'train_audio_windows': tr.wav_index.nunique(), 'train_rows': len(tr),
        'train_fit_audio_windows': tr[~val].wav_index.nunique(), 'train_fit_rows': int((~val).sum()),
        'validation_audio_windows': tr[val].wav_index.nunique(), 'validation_rows': int(val.sum()),
        'test_audio_windows': te.wav_index.nunique(), 'test_rows': len(te),
    }
    problems = []
    if strict:
        for key, expected in protocol['data']['canonical_counts'].items():
            if counts[key] != expected:
                problems.append(f'{key}: {counts[key]} != {expected}')
    train_sounds = set(zip(tr.story_id, tr.sound_id))
    test_sounds = set(zip(te.story_id, te.sound_id))
    if train_sounds & test_sounds:
        problems.append(f'sounds in both train and test: {sorted(train_sounds & test_sounds)}')
    single = sorted(set(te.subject_id) - set(te[te.session_id == 1].subject_id))
    if strict and single != protocol['data']['subjects_without_session_1']:
        problems.append(f'subjects without session 1: {single}')
    audio = {s: np.load(canon / 'audio' / f'extract_features_{s}4.npy', mmap_mode='r') for s in ('train', 'test')}
    if audio['train'].shape[0] != tr.wav_index.max() + 1 or audio['test'].shape[0] != te.wav_index.max() + 1:
        problems.append(f'audio embeddings {audio["train"].shape}/{audio["test"].shape} do not match wav_index')
    with np.load(canon / 'meg' / 'meg27_sr100.npz') as meg:
        keys = set(meg.files)
    needed = {f'subject{s:02d}_session{ses}_story{st}' for s, ses, st in
              pd.concat([tr, te])[['subject_id', 'session_id', 'story_id']].drop_duplicates().itertuples(index=False)}
    if needed - keys:
        problems.append(f'missing MEG recordings: {sorted(needed - keys)[:5]}')
    for name in ('sensor_xyz.npy', 'coords208_xy_scaled.npy'):
        if not (canon / 'coords' / name).exists():
            problems.append(f'missing coords/{name}')
    report = {'counts': counts, 'meg_recordings': len(keys), 'subjects_without_session_1': single,
              'audio_shapes': {k: list(v.shape) for k, v in audio.items()}, 'problems': problems}
    print(json.dumps(report, indent=2))
    return report


# ---------------------------------------------------------------- train

def train(name, protocol, canon=CANON, out=OUT, device='cuda', extra=()):
    spec = train_tasks(protocol)[name]
    if find_run_dir(name, out):
        print(f'{name}: already trained')
        return
    dirprocess = task_dirprocess(name, spec, canon, out)
    cmd = [str(Path(sys.executable).parent / 'lisa-train'), *protocol['training']['lisa_train_args'],
           '--experiments-root', str(Path(out) / 'runs'), '--run-group', name, '--seed', str(spec['seed']),
           '--dirprocess', str(dirprocess), '--device', device, *extra]
    print(' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    if not find_run_dir(name, out):
        raise RuntimeError(f'{name}: lisa-train finished without filters_*.npz')


# ---------------------------------------------------------------- eval

def load_run(run_dir, device):
    import torch
    from lisa.model.load_model import build_audio_model_from_config, load_audio_model_weights, load_from_config
    hyper = json.loads((run_dir / 'config.json').read_text())
    model, hyper = load_from_config(hyper, str(run_dir / f'{hyper["checkpoint"]}.pt'), torch.device(device))
    audio = build_audio_model_from_config(hyper, torch.device(device))
    audio = load_audio_model_weights(audio, str(run_dir), torch.device(device))
    return model.eval(), audio.eval(), hyper


def rank_windows(model, bank, df, meg, hyper, device, batch=128):
    """Rank of the true window and top-1 candidate for every row of df; the 'slot' column picks W_s.

    Similarity is LISA's get_clip_similarity (cosine over the whole feature x time map), computed as one
    matrix product against the bank normalised once, instead of renormalising the bank every batch.
    """
    import torch
    import torch.nn.functional as F
    sr = hyper['meg_sr']
    offset = int(hyper['meg_offset'] * sr)  # same expression as DoubleDataset
    bank_flat = F.normalize(bank, dim=(-2, -1)).flatten(1)
    ranks, top1 = [], []
    with torch.no_grad():
        for lo in range(0, len(df), batch):
            rows = df.iloc[lo:lo + batch]
            x = np.stack([meg[f'subject{r.subject_id:02d}_session{r.session_id}_story{r.story_id}']
                          [:, getattr(r, f'meg{sr}_start') + offset:getattr(r, f'meg{sr}_stop') + offset]
                          for r in rows.itertuples(index=False)])
            slot = torch.tensor(rows.slot.to_numpy() - 1, dtype=torch.long, device=device)
            _, emb = model((torch.tensor(x, dtype=torch.float32, device=device), slot))
            if emb.shape[1:] != bank.shape[1:]:
                raise ValueError(f'MEG output {tuple(emb.shape)} does not match the bank {tuple(bank.shape)}')
            sim = F.normalize(emb, dim=(-2, -1)).flatten(1) @ bank_flat.T
            true = torch.tensor(rows.wav_index.to_numpy(), device=device)
            ranks.append((sim > sim.gather(1, true[:, None])).sum(1).cpu().numpy())
            top1.append(sim.argmax(1).cpu().numpy())
    out = df[['subject_id', 'session_id', 'story_id', 'sound_id', 'wav_index', 'slot']].copy()
    out['rank'] = np.concatenate(ranks)
    out['top1_wav_index'] = np.concatenate(top1)
    return out


class Recordings(dict):
    """Loads only the MEG recordings that are asked for, from the canonical npz."""

    def __init__(self, path):
        super().__init__()
        self.npz = np.load(path)

    def __missing__(self, key):
        self[key] = self.npz[key]
        return self[key]


def slot_conditions(name, protocol, A, trained, test_df):
    """Yield (condition, rows with a 'slot' column, {slot: matrix override}), one subject at a time."""
    exps = protocol['experiments']

    def per_subject(condition, rows, overrides=lambda s: {}):
        for s, part in rows.groupby('subject_id'):
            yield condition, part.assign(slot=s), overrides(s)

    if name.startswith('repro'):
        yield from per_subject('own', test_df)
    elif name.startswith('xs'):
        direction = exps['exp08_cross_session']['directions'][name[2]]
        both = set(test_df[test_df.session_id == 0].subject_id) & set(test_df[test_df.session_id == 1].subject_id)
        same = test_df[(test_df.session_id == direction['train_session']) & test_df.subject_id.isin(trained)]
        cross = test_df[(test_df.session_id == direction['target_session']) & test_df.subject_id.isin(trained)]
        yield from per_subject('own_same', same)
        yield from per_subject('own_cross', cross)
        cross = cross[cross.subject_id.isin(both)]
        yield from per_subject('mean_cross', cross,
                               lambda s: {s: A[[t - 1 for t in trained if t != s]].mean(0)})
        for s, rows in cross.groupby('subject_id'):
            for donor in (t for t in trained if t != s):
                yield 'other_cross', rows.assign(slot=donor), {}
    elif name.startswith('ns'):
        fold = exps['exp09_new_subjects']['folds'][int(name.split('_')[1][1:])]
        rows = test_df[test_df.subject_id.isin(fold)]
        mean = A[[t - 1 for t in trained]].mean(0)
        norm = np.mean([np.linalg.norm(A[t - 1]) for t in trained]) / np.linalg.norm(mean)
        yield from per_subject('mean', rows, lambda s: {s: mean})
        yield from per_subject('mean_norm', rows, lambda s: {s: mean * norm})
        yield from per_subject('untrained', rows)


def rank_conditions(name, protocol, canon, out, device, shard=0, n_shards=1, threads=None, batch=128):
    """Ranked windows per condition for the target subjects of this shard.

    Work is split by target subject, so a shard only ever loads that subject's recordings.
    """
    import torch
    if threads:
        torch.set_num_threads(threads)
    spec = train_tasks(protocol)[name]
    run_dir = find_run_dir(name, out)
    model, audio, hyper = load_run(run_dir, device)
    bank = torch.tensor(np.load(Path(canon) / 'audio' / f'extract_features_test{hyper["embedding_layers"]}.npy'),
                        dtype=torch.float32, device=device).permute(0, 2, 1)
    with torch.no_grad():
        bank = audio(bank)
    test_df = pd.read_csv(Path(canon) / 'dataframe' / DF_TEST)
    trained = sorted(filter_rows(pd.read_csv(Path(canon) / 'dataframe' / DF_TRAIN), spec).subject_id.unique().tolist())
    meg = Recordings(Path(canon) / 'meg' / 'meg27_sr100.npz')
    layer = model.spatial_module.subject_layer
    original = layer.A.detach().clone()
    A = original.cpu().numpy()
    results = {}
    subjects = sorted(test_df.subject_id.unique())
    mine = set(subjects[shard::n_shards])
    for condition, rows, overrides in slot_conditions(name, protocol, A, trained, test_df):
        if rows.subject_id.iloc[0] not in mine:
            continue
        with torch.no_grad():
            layer.A.copy_(original)
            for s, matrix in overrides.items():
                layer.A[s - 1] = torch.tensor(matrix, dtype=layer.A.dtype, device=layer.A.device)
        results.setdefault(condition, []).append(rank_windows(model, bank, rows, meg, hyper, device, batch=batch))
    return results


def evaluate(name, protocol, canon=CANON, out=OUT, device='cuda', shards=1, batch=128):
    """Frozen evaluation of a trained task; shards > 1 splits the work over CPU processes."""
    spec = train_tasks(protocol)[name]
    target = Path(out) / 'eval' / name
    if (target / 'done.json').exists():
        print(f'{name}: already evaluated')
        return
    run_dir = find_run_dir(name, out)
    if run_dir is None:
        raise FileNotFoundError(f'{name}: no completed training run')
    if shards == 1:
        results = rank_conditions(name, protocol, canon, out, device, batch=batch)
    else:
        from concurrent.futures import ProcessPoolExecutor
        from multiprocessing import get_context
        shards = min(shards, pd.read_csv(Path(canon) / 'dataframe' / DF_TEST).subject_id.nunique())
        threads = max(1, (os.cpu_count() or shards) // (2 * shards))
        results = {}
        with ProcessPoolExecutor(shards, mp_context=get_context('spawn')) as pool:
            futures = [pool.submit(rank_conditions, name, protocol, canon, out, device, i, shards, threads, batch)
                       for i in range(shards)]
            for future in futures:
                for condition, parts in future.result().items():
                    results.setdefault(condition, []).extend(parts)
    test_df = pd.read_csv(Path(canon) / 'dataframe' / DF_TEST)
    target.mkdir(parents=True, exist_ok=True)
    summary = {}
    for condition, parts in results.items():
        df = pd.concat(parts, ignore_index=True).sort_values(['subject_id', 'session_id', 'slot', 'wav_index'])
        df.to_csv(target / f'{condition}.csv.gz', index=False)
        summary[condition] = {'rows': len(df), 'top1': float((df['rank'] == 0).mean()),
                              'top10': float((df['rank'] < 10).mean())}
    own = 'own' if 'own' in results else 'own_same' if 'own_same' in results else None
    if own:
        summary['trainer_agreement'] = agreement(pd.concat(results[own]), run_dir, spec, test_df)
    summary['run_dir'] = str(run_dir.relative_to(Path(out)))
    (target / 'done.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def agreement(ours, run_dir, spec, test_df):
    """Compare our own-slot ranks with the trainer's final test on the rows both evaluated.

    Our similarity is a matrix product, the trainer's an einsum, so near-ties can swap neighbouring
    ranks; the metrics must still match and every rank must stay within a couple of places.
    """
    theirs = pd.read_csv(run_dir / 'final_test_per_window.csv')
    keys = ['subject_id', 'session_id', 'wav_index']
    merged = ours.merge(theirs[keys + ['rank_of_true']], on=keys)
    expected = len(filter_rows(test_df, spec))
    gap = (merged['rank'] - merged.rank_of_true).abs()
    same, close = float((gap == 0).mean()), float((gap <= 2).mean())
    delta = abs(float((merged['rank'] < 10).mean() - (merged.rank_of_true < 10).mean()))
    if len(merged) != expected or same < 0.95 or close < 0.999 or delta > 0.002:
        raise RuntimeError(f'evaluation disagrees with the trainer: {len(merged)}/{expected} rows, '
                           f'{same:.4f} identical ranks, {close:.4f} within 2, top-10 differs by {delta:.4f}')
    return {'rows': len(merged), 'identical_ranks': same, 'within_2': close, 'top10_abs_diff': delta,
            'max_rank_gap': int(gap.max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=('check', 'tasks', 'train', 'eval'))
    parser.add_argument('--task')
    parser.add_argument('--protocol', default=str(PROTOCOL))
    parser.add_argument('--canon', default=str(CANON), help='authors\' data/preprocessed')
    parser.add_argument('--out', default=str(OUT))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--no-strict', action='store_true', help='check: skip the canonical-count comparison')
    parser.add_argument('--shards', type=int, default=1, help='eval: number of parallel CPU processes')
    parser.add_argument('--eval-batch-size', type=int, default=128)
    args, extra = parser.parse_known_args()  # train: unknown options go to lisa-train (e.g. --nepoch 1)
    if args.eval_batch_size < 1:
        parser.error('--eval-batch-size must be positive')
    if extra and args.command != 'train':
        parser.error(f'unrecognized arguments: {extra}')
    protocol = load_protocol(args.protocol)
    if args.command == 'check':
        sys.exit(1 if check(args.canon, protocol, strict=not args.no_strict)['problems'] else 0)
    if args.command == 'tasks':
        for name, spec in train_tasks(protocol).items():
            print(name, json.dumps(spec))
        return
    if not args.task:
        parser.error('--task is required')
    with job_lock(f'{args.command}_{args.task}', args.out):
        if args.command == 'train':
            train(args.task, protocol, args.canon, args.out, args.device, extra)
        else:
            evaluate(args.task, protocol, args.canon, args.out, args.device, args.shards, args.eval_batch_size)


if __name__ == '__main__':
    os.environ.setdefault('PYTHONUNBUFFERED', '1')
    main()
