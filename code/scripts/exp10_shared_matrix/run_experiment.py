"""Server runner for exp10: tasks, train, eval, summarize (no downloads)."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / 'code') not in sys.path:
    sys.path.insert(0, str(ROOT / 'code'))

from scripts.exp08_09_full_lisa import full_lisa
from scripts.exp10_shared_matrix.recordings import RecordingStore

PROTOCOL = ROOT / 'experiments/exp10_shared_matrix/protocol.json'
OUT = ROOT / 'outputs/exp10_shared_matrix'


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def tasks(protocol):
    """Circular held-out validation folds, preserving exp09's test triples."""
    folds = protocol['split']['folds']
    subjects = protocol['data']['subjects']
    flat = [s for fold in folds for s in fold]
    if (len(folds) < 3 or any(not fold for fold in folds) or len(flat) != len(set(flat))
            or sorted(flat) != sorted(subjects) or len(subjects) != len(set(subjects))
            or any(type(s) is not int or not 1 <= s <= 27 for s in subjects)):
        raise ValueError('folds must partition unique subject IDs in [1, 27], in at least 3 groups')
    seeds = protocol['seeds']
    probabilities = protocol['routing']['p_personal']
    if (not seeds or len(seeds) != len(set(seeds)) or any(type(s) is not int or s < 0 for s in seeds)
            or not probabilities or len(probabilities) != len(set(probabilities))
            or any(not math.isfinite(p) or not 0 < p < 1 for p in probabilities)):
        raise ValueError('unique nonnegative seeds and switch probabilities strictly inside (0, 1) required')
    variants = protocol['variants']
    if sorted(variants) != ['individual', 'shared_only', 'switch']:
        raise ValueError('protocol must include individual, shared_only and switch exactly once')
    result = {}
    for k, test in enumerate(folds):
        val = folds[(k + 1) % len(folds)]
        train = sorted(set(subjects) - set(test) - set(val))
        for seed in seeds:
            for variant in variants:
                ps = probabilities if variant == 'switch' else [int(variant == 'individual')]
                for p in ps:
                    label = f'switch_p{str(p).replace(".", "p")}' if variant == 'switch' else variant
                    name = f'{label}_f{k}_s{seed}'
                    result[name] = dict(variant=variant, p_personal=p, fold=k, seed=seed,
                                        train_subjects=train, val_subjects=sorted(val),
                                        test_subjects=sorted(test))
    return result


def split_rows(frame, spec, validation_sound):
    """Training and validation differ in BOTH subject and continuous audio segment."""
    is_val = (frame.story_id == validation_sound[0]) & (frame.sound_id == validation_sound[1])
    train = frame[frame.subject_id.isin(spec['train_subjects']) & ~is_val].reset_index(drop=True)
    val = frame[frame.subject_id.isin(spec['val_subjects']) & is_val].reset_index(drop=True)
    for label, rows, expected in [('train', train, spec['train_subjects']),
                                   ('val', val, spec['val_subjects'])]:
        if set(rows.subject_id) != set(expected):
            raise ValueError(f'{label}: missing subjects or empty split')
    if set(train.wav_index) & set(val.wav_index):
        raise ValueError('training and validation audio indices overlap')
    return train, val


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_head(checkout):
    """Read HEAD from the checkout even if the compute node has no Git executable."""
    try:
        return subprocess.run(['git', '-C', str(checkout), 'rev-parse', 'HEAD'],
                              check=True, capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        git_dir = Path(checkout) / '.git'
        if git_dir.is_file():
            pointer = git_dir.read_text().strip()
            if not pointer.startswith('gitdir: '):
                raise ValueError(f'{git_dir}: invalid Git directory pointer')
            git_dir = git_dir.parent / pointer.removeprefix('gitdir: ')

        head = (git_dir / 'HEAD').read_text().strip()
        if head.startswith('ref: '):
            ref = head.removeprefix('ref: ')
            if not ref.startswith('refs/') or '..' in Path(ref).parts:
                raise ValueError(f'{git_dir / "HEAD"}: invalid Git reference')
            common = git_dir / 'commondir'
            refs = git_dir / common.read_text().strip() if common.is_file() else git_dir
            loose_ref = refs / ref
            if loose_ref.is_file():
                head = loose_ref.read_text().strip()
            else:
                packed = refs / 'packed-refs'
                if packed.is_file():
                    for line in packed.read_text().splitlines():
                        revision, _, name = line.partition(' ')
                        if name == ref:
                            head = revision
                            break

        if not re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}', head):
            raise ValueError(f'{checkout}: cannot resolve Git HEAD without Git; check out the pinned commit')
        return head


def source_identity(canon):
    """Small sources are hashed; large input arrays have explicit stat fingerprints."""
    sources = list(Path(__file__).parent.glob('*.py')) + [
        ROOT / 'code/scripts/exp08_09_full_lisa/full_lisa.py',
        ROOT / 'code/LISA/src/lisa/model/nn_modules.py',
        ROOT / 'code/LISA/src/lisa/model/load_model.py',
        ROOT / 'code/LISA/src/lisa/training/criteria.py',
        ROOT / 'code/LISA/src/lisa/data/datasets.py',
        ROOT / 'code/LISA/src/lisa/cli/parse_experiment_args.py',
        Path(canon) / 'dataframe' / full_lisa.DF_TRAIN,
        Path(canon) / 'dataframe' / full_lisa.DF_TEST,
        Path(canon) / 'coords/sensor_xyz.npy',
        Path(canon) / 'coords/coords208_xy_scaled.npy',
    ]
    hashes = {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): sha256(p)
              for p in sorted(sources)}
    large_files = [Path(canon) / 'audio/extract_features_train4.npy',
                   Path(canon) / 'audio/extract_features_test4.npy',
                   Path(canon) / 'meg/meg27_sr100.npz']
    stats = {str(p): {'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns} for p in large_files}
    head = git_head(ROOT / 'code/LISA')
    return {'sha256': hashes, 'large_inputs_stat_only': stats, 'upstream_head': head}


def validate_rows(rows, hidden, label):
    indices = rows.wav_index.to_numpy()
    if (rows.empty or not np.issubdtype(indices.dtype, np.integer)
            or indices.min() < 0 or indices.max() >= len(hidden)):
        raise ValueError(f'{label}: invalid audio indices or empty dataframe')
    if hidden.ndim != 3 or hidden.shape[-1] != 768 or not np.isfinite(hidden).all():
        raise ValueError(f'{label}: expected finite audio features (windows, time, 768)')


def rank_rows(model, rows, hidden, candidate_ids, recordings, hyper, device, batch):
    """Frozen retrieval against a fixed bank; candidate IDs need not be contiguous."""
    import torch
    import torch.nn.functional as F
    if batch < 1 or rows.empty:
        raise ValueError('positive batch and nonempty rows required')
    ids = np.asarray(candidate_ids, dtype=np.int64)
    if len(ids) < 10 or not np.array_equal(ids, np.unique(ids)):
        raise ValueError('candidate IDs must be sorted, unique and contain at least 10 windows')
    if not set(rows.wav_index) <= set(ids):
        raise ValueError('retrieval labels must belong to the candidate bank')
    model.eval()
    sr = hyper['meg_sr']
    offset = int(hyper['meg_offset'] * sr)
    ranks, top1 = [], []
    with torch.no_grad():
        bank = torch.as_tensor(np.asarray(hidden[ids]), dtype=torch.float32, device=device).permute(0, 2, 1)
        bank_flat = F.normalize(bank, dim=(-2, -1)).flatten(1)
        for lo in range(0, len(rows), batch):
            part = rows.iloc[lo:lo + batch]
            signals = [recordings[f'subject{r.subject_id:02d}_session{r.session_id}_story{r.story_id}']
                       [:, getattr(r, f'meg{sr}_start') + offset:getattr(r, f'meg{sr}_stop') + offset]
                       for r in part.itertuples(index=False)]
            subjects = torch.tensor(part.subject_id.to_numpy() - 1, dtype=torch.long, device=device)
            _, embedding = model((torch.tensor(np.stack(signals), device=device, dtype=torch.float32), subjects))
            if embedding.shape[1:] != bank.shape[1:]:
                raise ValueError(f'MEG/audio shapes differ: {embedding.shape} vs {bank.shape}')
            similarity = F.normalize(embedding, dim=(-2, -1)).flatten(1) @ bank_flat.T
            if not torch.isfinite(similarity).all():
                raise FloatingPointError('nonfinite retrieval similarity')
            true = torch.tensor(np.searchsorted(ids, part.wav_index.to_numpy()), device=device)
            ranks.extend((similarity > similarity.gather(1, true[:, None])).sum(1).cpu().tolist())
            top1.extend(ids[similarity.argmax(1).cpu().numpy()].tolist())
    result = rows[['subject_id', 'session_id', 'story_id', 'sound_id', 'wav_index']].copy()
    result['rank'], result['top1_wav_index'] = ranks, top1
    return result


def subject_metrics(ranks):
    return (ranks.assign(top1=ranks['rank'].eq(0), top10=ranks['rank'].lt(10))
            .groupby('subject_id').agg(top1=('top1', 'mean'), top10=('top10', 'mean'),
                                      mean_rank=('rank', 'mean'), windows=('rank', 'size')))


def validation_score(ranks):
    metrics = subject_metrics(ranks)
    return float(metrics.top10.mean()), float(metrics.top1.mean())


def hyperparameters(protocol, spec, canon, settings, device):
    from lisa.cli.parse_experiment_args import parse_experiment_arguments
    hyper = parse_experiment_arguments([
        *protocol['training']['lisa_args'], '--device', device, '--seed', str(spec['seed']),
        '--dirprocess', str(canon), '--batch-size', str(settings['batch_size']), '--run-group', 'exp10',
    ])
    # This runner intentionally implements the fixed exp10 objective, not every LISA ablation.
    if (hyper['feature_reduction'] != 'none' or hyper['time_reduction'] != 'none'
            or hyper['embedding_layers'] != 4 or hyper['meg_sr'] != 100
            or hyper['window_tag'] or not hyper['use_subject_layer']
            or hyper['optim'] != 'AdamW' or hyper['tf_smooth'] != 0
            or hyper['tf_unfreeze_epoch'] is not None or hyper['temporal_filter_freeze']
            or hyper['lr_temporal_mult'] != 1 or hyper['wd_temporal'] is not None):
        raise ValueError('unsupported change to the fixed exp10 model/objective')
    return hyper


def train_task(name, spec, protocol, canon, out, device, settings):
    import torch
    from torch.utils.data import DataLoader
    from lisa.data.datasets import DoubleDataset
    from lisa.training.criteria import CLIPLoss
    from lisa.utils.determinism import fix_seed
    from scripts.exp10_shared_matrix.model import build_model

    run = Path(out) / 'runs' / name
    identity = source_identity(canon)
    if identity['upstream_head'] != protocol['upstream_commit']:
        raise ValueError('LISA HEAD differs from the protocol pin')
    runtime = dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
                   pandas=pd.__version__, cuda=torch.version.cuda,
                   gpu=torch.cuda.get_device_name(device) if device.startswith('cuda') and torch.cuda.is_available() else None)
    manifest = dict(protocol=protocol, task=spec, settings=settings, source=identity, device=device, runtime=runtime)
    if (run / 'done.json').exists():
        if read_json(run / 'manifest.json') != manifest:
            raise ValueError(f'{name}: provenance/settings changed; use a fresh --out directory')
        print(f'{name}: already trained', flush=True)
        return
    if run.exists():
        raise FileExistsError(f'{run}: incomplete run; use a fresh --out or inspect it before removing')
    hyper = hyperparameters(protocol, spec, canon, settings, device)
    frame = pd.read_csv(Path(canon) / 'dataframe' / full_lisa.DF_TRAIN)
    if set(frame.subject_id) != set(protocol['data']['subjects']):
        raise ValueError('canonical training subjects differ from protocol')
    test_metadata = pd.read_csv(Path(canon) / 'dataframe' / full_lisa.DF_TEST, usecols=['story_id', 'sound_id'])
    if set(zip(frame.story_id, frame.sound_id)) & set(map(tuple, test_metadata.to_numpy())):
        raise ValueError('canonical training and test sounds overlap')
    train_rows, val_rows = split_rows(frame, spec, protocol['data']['validation_sound'])
    hidden = np.load(Path(canon) / 'audio/extract_features_train4.npy')
    validate_rows(pd.concat([train_rows, val_rows]), hidden, 'train/validation')
    candidate_ids = np.sort(val_rows.wav_index.unique())
    if len(candidate_ids) != protocol['data']['validation_candidates']:
        raise ValueError('validation candidate count differs from protocol')
    recordings = RecordingStore(Path(canon) / 'meg/meg27_sr100.npz')
    try:
        dataset = DoubleDataset(recordings, hidden, train_rows, hyper['meg_sr'], hyper['meg_offset'])
        if len(dataset) < settings['batch_size']:
            raise ValueError('training split smaller than batch_size (drop_last=True)')
        fix_seed(spec['seed'], hyper['torch_deterministic'])
        model = build_model(hyper, spec, device=device)
        criterion = CLIPLoss(hyper['clip_temperature']).to(device)
        optimizer = torch.optim.AdamW([
            {'params': [p for p in model.parameters() if p.requires_grad],
             'lr': hyper['lr_fe'], 'weight_decay': hyper['weight_decay']},
            {'params': criterion.parameters(), 'lr': hyper['clip_temperature_lr'], 'weight_decay': 0},
        ])
        loader = DataLoader(dataset, batch_size=settings['batch_size'], shuffle=True, drop_last=True,
                            num_workers=settings['workers'],
                            generator=torch.Generator().manual_seed(spec['seed']))
        run.mkdir(parents=True)
        print(f'{name}: train={len(train_rows)} rows/{len(spec["train_subjects"])} subjects; '
              f'validation={len(val_rows)} rows/{len(spec["val_subjects"])} subjects; '
              f'gallery={len(candidate_ids)}; device={device}', flush=True)
        write_json(run / 'manifest.json', manifest)
        write_json(run / 'config.json', {**hyper, 'device': str(device)})
        write_json(run / 'protocol.json', protocol)
        best, stale, history = (-1.0, -1.0), 0, []
        for epoch in range(settings['nepoch']):
            model.train()
            criterion.train()
            layer = model.spatial_module.subject_layer
            layer.route_counts.zero_()
            total_loss, seen, skipped = 0.0, 0, 0
            for step, (meg, subject, audio, wav_index, _) in enumerate(loader):
                if wav_index.unique().numel() < 2:
                    skipped += 1
                    continue
                optimizer.zero_grad(set_to_none=True)
                _, prediction = model((meg.to(device), subject.to(device)))
                _, loss = criterion(prediction, audio.to(device).permute(0, 2, 1), wav_index.to(device))
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'{name}: nonfinite loss at epoch {epoch}')
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach()) * len(meg)
                seen += len(meg)
                if (step + 1) % 100 == 0:
                    print(f'{name} epoch={epoch} batch={step + 1}/{len(loader)} '
                          f'loss={total_loss / seen:.5f}', flush=True)
            if not seen:
                raise ValueError('no usable training batches with distinct audio positives')
            ranks = rank_rows(model, val_rows, hidden, candidate_ids, recordings, hyper, device,
                              settings['eval_batch_size'])
            score = validation_score(ranks)
            if not all(math.isfinite(s) for s in score):
                raise FloatingPointError('nonfinite validation score')
            improved = score > best
            counts = layer.route_counts.cpu().tolist()
            history.append(dict(epoch=epoch, train_loss=total_loss / seen, val_top10=score[0],
                                val_top1=score[1], personal_examples=counts[0], shared_examples=counts[1],
                                skipped_batches=skipped))
            pd.DataFrame(history).to_csv(run / 'metrics.csv', index=False)
            print(f'{name} epoch={epoch} loss={total_loss / seen:.5f} '
                  f'val Top-10={score[0]:.4f} Top-1={score[1]:.4f} routes={counts}', flush=True)
            if improved:
                best, stale = score, 0
                temporary = run / 'best.tmp.pt'
                torch.save(dict(model_state_dict=model.state_dict(), criterion_state_dict=criterion.state_dict(),
                                optimizer_state_dict=optimizer.state_dict(), epoch=epoch, val_score=list(score)), temporary)
                temporary.replace(run / 'best.pt')
                ranks.to_csv(run / 'validation.csv.gz', index=False)
                subject_metrics(ranks).to_csv(run / 'validation_subjects.csv')
                write_json(run / 'best.json', dict(epoch=epoch, val_top10=score[0], val_top1=score[1],
                                                 validation_candidates=len(candidate_ids)))
            else:
                stale += 1
            if stale >= protocol['training']['early_stopping_patience']:
                break
        write_json(run / 'done.json', read_json(run / 'best.json'))
    finally:
        recordings.close()


def select_tasks(protocol, out, folds=None):
    """Select p using validation only; retain both predefined control conditions."""
    all_tasks = tasks(protocol)
    selected = []
    for fold in range(len(protocol['split']['folds'])) if folds is None else sorted(folds):
        for seed in protocol['seeds']:
            group = [(name, spec) for name, spec in all_tasks.items()
                     if spec['fold'] == fold and spec['seed'] == seed]
            # All p values and both controls must have completed before opening test data.
            scores = {}
            for name, _ in group:
                scores[name] = read_json(Path(out) / 'runs' / name / 'done.json')
            switching = [(n, s) for n, s in group if s['variant'] == 'switch']
            winner, _ = max(switching, key=lambda item: (scores[item[0]]['val_top10'],
                                                        scores[item[0]]['val_top1'],
                                                        -item[1]['p_personal']))
            selected.extend(n for n, s in group if s['variant'] != 'switch' or n == winner)
    return selected


def evaluate_task(name, spec, protocol, canon, out, device, batch):
    from scripts.exp10_shared_matrix.model import build_model
    run = Path(out) / 'runs' / name
    read_json(run / 'done.json')
    manifest = read_json(run / 'manifest.json')
    if (manifest['protocol'] != protocol or manifest['task'] != spec
            or manifest['source'] != source_identity(canon)):
        raise ValueError(f'{name}: protocol/data/code differs from training; use matching sources')
    target = Path(out) / 'eval' / name
    checkpoint_hash = sha256(run / 'best.pt')
    if (target / 'done.json').exists():
        if read_json(target / 'done.json')['checkpoint_sha256'] != checkpoint_hash:
            raise ValueError('evaluated checkpoint changed; use a fresh output directory')
        print(f'{name}: already evaluated', flush=True)
        return
    hyper = read_json(run / 'config.json')
    hyper['dirprocess'] = str(canon)
    model = build_model(hyper, spec, checkpoint=run / 'best.pt', device=device)
    rows = pd.read_csv(Path(canon) / 'dataframe' / full_lisa.DF_TEST)
    train_sounds = pd.read_csv(Path(canon) / 'dataframe' / full_lisa.DF_TRAIN)[['story_id', 'sound_id']]
    if set(map(tuple, train_sounds.to_numpy())) & set(zip(rows.story_id, rows.sound_id)):
        raise ValueError('canonical training and test sounds overlap')
    rows = rows[rows.subject_id.isin(spec['test_subjects'])].reset_index(drop=True)
    if set(rows.subject_id) != set(spec['test_subjects']):
        raise ValueError('missing test subjects')
    hidden = np.load(Path(canon) / 'audio/extract_features_test4.npy')
    validate_rows(rows, hidden, 'test')
    if len(hidden) != protocol['data']['test_candidates']:
        raise ValueError('test candidate count differs from protocol')
    recordings = RecordingStore(Path(canon) / 'meg/meg27_sr100.npz')
    try:
        ranks = rank_rows(model, rows, hidden, np.arange(len(hidden)), recordings, hyper, device, batch)
    finally:
        recordings.close()
    target.mkdir(parents=True, exist_ok=True)
    ranks.to_csv(target / 'windows.csv.gz', index=False)
    metrics = subject_metrics(ranks)
    metrics.to_csv(target / 'subjects.csv')
    write_json(target / 'done.json', dict(task=spec, checkpoint_sha256=checkpoint_hash,
                                        test_candidates=len(hidden), subjects=len(metrics),
                                        top10=float(metrics.top10.mean()), top1=float(metrics.top1.mean())))
    print(f'{name}: test Top-10={metrics.top10.mean():.4f}', flush=True)


def summarize(protocol, out):
    from scipy import stats
    # A partial pilot is allowed, but never silently mix different p values or missing controls.
    tables = []
    all_tasks = tasks(protocol)
    completed_groups = set()
    for path in sorted((Path(out) / 'eval').glob('*/done.json')):
        spec = all_tasks[path.parent.name]
        manifest = read_json(Path(out) / 'runs' / path.parent.name / 'manifest.json')
        if manifest['protocol'] != protocol or manifest['task'] != spec:
            raise ValueError('summary protocol/task differs from evaluated runs')
        if read_json(path)['checkpoint_sha256'] != sha256(Path(out) / 'runs' / path.parent.name / 'best.pt'):
            raise ValueError('summary checkpoint differs from evaluated checkpoint')
        completed_groups.add((spec['fold'], spec['seed']))
    if not completed_groups:
        raise ValueError('no evaluated tasks')
    for fold, seed in sorted(completed_groups):
        group = [(n, s) for n, s in all_tasks.items() if s['fold'] == fold and s['seed'] == seed]
        manifests = [read_json(Path(out) / 'runs' / n / 'manifest.json') for n, _ in group]
        reference = manifests[0]
        for manifest in manifests[1:]:
            if manifest['settings'] != reference['settings'] or manifest['source'] != reference['source']:
                raise ValueError('compared variants must use identical training settings and source/data fingerprints')
        switches = [(n, s) for n, s in group if s['variant'] == 'switch']
        best = max(switches, key=lambda item: (
            read_json(Path(out) / 'runs' / item[0] / 'done.json')['val_top10'],
            read_json(Path(out) / 'runs' / item[0] / 'done.json')['val_top1'], -item[1]['p_personal']))[0]
        for name, spec in group:
            if spec['variant'] == 'switch' and name != best:
                continue
            table = pd.read_csv(Path(out) / 'eval' / name / 'subjects.csv')
            tables.append(table.assign(variant=spec['variant'], seed=seed, fold=fold,
                                       p_personal=spec['p_personal']))
    per_seed = pd.concat(tables, ignore_index=True)
    per_subject = per_seed.groupby(['variant', 'subject_id'])[['top1', 'top10', 'mean_rank']].mean().reset_index()
    summary = []
    for variant, table in per_subject.groupby('variant'):
        for metric in ['top1', 'top10', 'mean_rank']:
            values = table[metric]
            half = float(stats.t.ppf(0.975, len(values) - 1) * stats.sem(values)) if len(values) > 1 else None
            summary.append(dict(variant=variant, metric=metric, subjects=len(values), mean=float(values.mean()),
                                ci95_half=half))
    wide = per_subject.pivot(index='subject_id', columns='variant', values='top10')
    comparisons = []
    for left, right in [('switch', 'individual'), ('switch', 'shared_only'), ('shared_only', 'individual')]:
        difference = wide[left] - wide[right]
        if difference.isna().any():
            raise ValueError('paired comparison is missing subjects')
        p = float(stats.wilcoxon(difference).pvalue) if (difference != 0).any() else 1.0
        comparisons.append(dict(comparison=f'{left}-{right}', subjects=len(difference),
                                top10_delta=float(difference.mean()), p=p))
    order = sorted(range(len(comparisons)), key=lambda i: comparisons[i]['p'])
    adjusted = 0.0
    for rank, idx in enumerate(order):
        adjusted = max(adjusted, min(1.0, (len(order) - rank) * comparisons[idx]['p']))
        comparisons[idx]['p_holm'] = adjusted
    target = Path(out) / 'summary'
    target.mkdir(exist_ok=True)
    per_seed.to_csv(target / 'per_seed.csv', index=False)
    per_subject.to_csv(target / 'per_subject.csv', index=False)
    pd.DataFrame(summary).to_csv(target / 'summary.csv', index=False)
    pd.DataFrame(comparisons).to_csv(target / 'comparisons.csv', index=False)
    write_json(target / 'coverage.json', dict(completed_fold_seeds=sorted(completed_groups),
                                            planned_fold_seeds=len(protocol['split']['folds']) * len(protocol['seeds']),
                                            complete=len(completed_groups) == len(protocol['split']['folds']) * len(protocol['seeds'])))
    print(pd.DataFrame(summary).to_string(index=False))
    return per_subject


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('command', choices=['tasks', 'train', 'eval', 'summarize'])
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    parser.add_argument('--canon', type=Path, default=full_lisa.CANON)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--task', help='exact task name from tasks')
    parser.add_argument('--fold', type=int, help='limit --all to one fold (pilot or distributed servers)')
    parser.add_argument('--all', action='store_true', help='train all tasks or evaluate validation-selected tasks')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--nepoch', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--eval-batch-size', type=int)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--threads', type=int, help='CPU threads per process; useful for smoke runs')
    parser.add_argument('--dry-run', action='store_true', help='print tasks without loading MEG or Torch')
    args = parser.parse_args()
    protocol = read_json(args.protocol)
    all_tasks = tasks(protocol)
    settings = {key: protocol['training'][key] for key in ['nepoch', 'batch_size', 'eval_batch_size', 'workers']}
    for key in settings:
        if getattr(args, key) is not None:
            settings[key] = getattr(args, key)
    if (settings['nepoch'] < 1 or settings['batch_size'] < 2 or settings['eval_batch_size'] < 1
            or settings['workers'] < 0 or (args.threads is not None and args.threads < 1)
            or protocol['training']['early_stopping_patience'] < 1):
        parser.error('positive epochs/eval batch/threads/patience, batch_size >= 2 and workers >= 0 required')
    if args.fold is not None and not 0 <= args.fold < len(protocol['split']['folds']):
        parser.error('--fold is out of range')
    if args.task and (args.all or args.fold is not None):
        parser.error('--task cannot be combined with --all or --fold')
    if args.command == 'summarize':
        summarize(protocol, args.out)
        return
    if args.task:
        if args.task not in all_tasks:
            parser.error('unknown task; use tasks to list names')
        names = [args.task]
    elif args.command == 'tasks' or args.all:
        names = list(all_tasks)
    else:
        parser.error('specify --task or --all')
    names = [n for n in names if args.fold is None or all_tasks[n]['fold'] == args.fold]
    if args.command == 'tasks' or args.dry_run:
        for name in names:
            print(name, json.dumps(all_tasks[name]))
        return
    if args.threads:
        import torch
        torch.set_num_threads(args.threads)
    canon, out = args.canon.resolve(), args.out.resolve()
    # For --all evaluation, choose p before opening test data (within the requested folds).
    if args.command == 'eval' and args.all:
        names = select_tasks(protocol, out, {all_tasks[n]['fold'] for n in names})
    for name in names:
        with full_lisa.job_lock(f'{args.command}_{name}', out):
            if args.command == 'train':
                train_task(name, all_tasks[name], protocol, canon, out, args.device, settings)
            else:
                evaluate_task(name, all_tasks[name], protocol, canon, out, args.device, settings['eval_batch_size'])


if __name__ == '__main__':
    main()
