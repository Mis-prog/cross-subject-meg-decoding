"""27-subject LOSO series: shared models, ceiling, and holdout calibrators on one GPU.

Usage (from code/scripts/exp06_loso):
  python train_loso.py fit --model shared --holdout 5 --seed 0
  python train_loso.py fit --model shared_ea --holdout 5 --seed 0
  python train_loso.py fit --model ceiling --seed 0
  python train_loso.py tune --holdout 1 --method ft_Ws          # dev subjects only, validation windows
  python train_loso.py calibrate --holdout 5 --seed 0            # needs experiments/exp06_loso/results/tuning.json
"""
import argparse
import json
import os
import platform
import time

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/fqw-mpl')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from lisa.cli.parse_experiment_args import parse_experiment_arguments
from lisa.model.load_model import load_from_config
from lisa.training.criteria import CLIPLoss, get_clip_similarity
from lisa.utils.determinism import fix_seed
import sys
from pathlib import Path

# Support direct execution from any working directory.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.exp04_small_baseline.prepare_baseline import sha256
from scripts.exp06_loso.prepare_loso import OUT, PROTOCOL, RESULTS, ROOT, apply_scaler, fit_scaler

RUNS = ROOT / 'data/pilot/loso_runs'
RES = RESULTS
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH, EVAL_BATCH = 64, 256
LR, TEMP_LR = 3e-4, 1e-3
MAX_EPOCHS, PATIENCE = 150, 15
EA_SHRINK = 1e-2
RANK = 8
TUNE_BUDGETS = (30, 120, 600)
TUNE_STEPS = (0, 10, 25, 50, 100, 200, 400)
TUNE_LRS = {'ft_Ws': (3e-4, 1e-3, 3e-3, 1e-2), 'ft_lowrank': (3e-4, 1e-3, 3e-3, 1e-2),
            'ea_ft_Ws': (3e-4, 1e-3, 3e-3, 1e-2), 'ft_all': (3e-6, 1e-5, 3e-5, 1e-4)}
# Tuned steps are chosen per budget group: each budget uses the nearest tuning budget.
STEP_GROUP = {30: 30, 60: 30, 120: 120, 300: 120, 600: 600, 882: 600}


class Data:
    """Prepared windows; MEG stays memory-mapped on the host and batches are moved to the GPU."""

    def __init__(self):
        self.protocol = json.loads(PROTOCOL.read_text())
        prep = json.loads((RESULTS / 'loso_preparation.json').read_text())
        if sha256(PROTOCOL) != prep['protocol_sha256']:
            raise ValueError('Protocol changed since preparation')
        self.windows = pd.read_csv(OUT / 'meg_windows.csv')
        self.scaled = np.load(OUT / 'meg_windows.npy', mmap_mode='r')
        self.unscaled = np.load(OUT / 'meg_unscaled.npy', mmap_mode='r')
        self.y = torch.from_numpy(np.load(OUT / 'audio_embeddings.npy')).transpose(1, 2).contiguous().to(DEVICE)
        self.scalers = json.loads((OUT / 'scalers.json').read_text())
        self.subject_index = torch.as_tensor(self.windows.subject.to_numpy() - 1, dtype=torch.long, device=DEVICE)
        self.audio_index = torch.as_tensor(self.windows.audio_id.to_numpy(), dtype=torch.long, device=DEVICE)

    def ids(self, split, subjects):
        rows = self.windows[(self.windows.split == split) & self.windows.subject.isin(subjects)]
        return rows.window_id.to_numpy()

    def batch(self, ids):
        return torch.from_numpy(self.scaled[np.sort(ids)]).to(DEVICE), np.sort(ids)

    def pool(self, subject):
        rows = self.windows[(self.windows.subject == subject) & (self.windows.split == 'train')]
        return rows.sort_values(['task', 'meg_start_sample_1000hz']).window_id.to_numpy()


def build_model(data, seed):
    config = parse_experiment_arguments([
        '--run-group', data.protocol['name'], '--device', DEVICE.type, '--seed', str(seed),
        '--spatial-dropout-number', '0', '--dirprocess', str(ROOT / 'code/LISA/data/preprocessed'),
    ])
    model, _ = load_from_config(config, None, DEVICE)
    if model.spatial_module.subject_layer.A.shape[0] < len(data.protocol['subjects']):
        raise ValueError('Model has fewer subject slots than subjects')
    return model, config


def ea_matrix(x):
    """Inverse square root of the shrunk mean spatial covariance of windows x (n, C, T)."""
    x = x.double()
    cov = torch.einsum('nct,ndt->cd', x, x) / (x.shape[0] * x.shape[2])
    cov = (1 - EA_SHRINK) * cov + EA_SHRINK * cov.trace() / cov.shape[0] * torch.eye(cov.shape[0], device=x.device, dtype=x.dtype)
    values, vectors = torch.linalg.eigh(cov)
    return (vectors * values.clamp_min(1e-12).rsqrt()) @ vectors.T


def align(x, transform):
    return x if transform is None else torch.einsum('cd,ndt->nct', transform.to(x.dtype), x)


@torch.no_grad()
def evaluate(model, criterion, x, subject_idx, audio_ids, bank_ids):
    """Rank each query (x[i], audio_ids[i]) against the bank; x is already scaled/aligned, on device."""
    model.eval()
    bank_ids = torch.as_tensor(np.sort(bank_ids), device=DEVICE)
    candidates = DATA.y[bank_ids]
    target = torch.searchsorted(bank_ids, torch.as_tensor(audio_ids, device=DEVICE))
    losses, ranks = [], []
    for chunk in torch.arange(len(x), device=DEVICE).split(EVAL_BATCH):
        _, predicted = model((x[chunk], subject_idx[chunk]))
        _, _, sim = get_clip_similarity(predicted, candidates)
        losses.append(F.cross_entropy(sim / criterion.temperature.exp(), target[chunk], reduction='none'))
        true = sim.gather(1, target[chunk, None])
        ranks.append((sim > true).sum(dim=1) + 1)
    loss, rank = torch.cat(losses), torch.cat(ranks).float()
    return {'loss': loss.mean().item(), 'top1': (rank == 1).float().mean().item(),
            'top10': (rank <= 10).float().mean().item(), 'mean_rank': rank.mean().item(),
            'queries': len(rank), 'candidates': len(bank_ids)}


def bank(split):
    return DATA.windows.loc[DATA.windows.split == split, 'audio_id'].unique()


def load_aligned(ids, transforms=None):
    """Stored scaled windows on the device, each subject aligned with its EA matrix if given."""
    x, ids = DATA.batch(ids)
    if transforms:
        subjects = DATA.windows.subject.to_numpy()[ids]
        for s in np.unique(subjects):
            mask = torch.as_tensor(subjects == s, device=DEVICE)
            x[mask] = align(x[mask], transforms[int(s)])
    return x, ids


def evaluate_ids(model, criterion, ids, transforms=None, loaded=None):
    x, ids = loaded if loaded is not None else load_aligned(ids, transforms)
    split = DATA.windows.split.to_numpy()[ids]
    if len(set(split)) != 1:
        raise ValueError('Queries must come from one split')
    return evaluate(model, criterion, x, DATA.subject_index[ids], DATA.audio_index[ids], bank(split[0]))


def model_path(kind, holdout, seed):
    name = f'{kind}_s{seed}' if kind == 'ceiling' else f'{kind}_h{holdout:02d}_s{seed}'
    return RUNS / f'{name}.pt', RES / 'models' / f'{name}.json'


def fit(kind, holdout, seed, deterministic):
    fix_seed(seed, torch_deterministic=deterministic)
    subjects = DATA.protocol['subjects']
    fit_subjects = subjects if kind == 'ceiling' else [s for s in subjects if s != holdout]
    checkpoint, summary_path = model_path(kind, holdout, seed)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    transforms = None
    if kind == 'shared_ea':
        transforms = {s: ea_matrix(DATA.batch(DATA.ids('train', [s]))[0]) for s in fit_subjects}
    model, config = build_model(DATA, seed)
    criterion = CLIPLoss(1.0).to(DEVICE)
    optimizer = torch.optim.AdamW([{'params': model.parameters(), 'lr': LR},
                                   {'params': criterion.parameters(), 'lr': TEMP_LR}], weight_decay=0.0)
    train_x, train_ids = load_aligned(DATA.ids('train', fit_subjects), transforms)
    val = load_aligned(DATA.ids('validation', fit_subjects), transforms)
    train_s, train_a = DATA.subject_index[train_ids], DATA.audio_index[train_ids]
    started = time.perf_counter()
    best = evaluate_ids(model, criterion, None, loaded=val)
    best_epoch, history = 0, [{'epoch': 0, **{f'val_{k}': v for k, v in best.items()}}]
    state = {k: v.clone() for k, v in model.state_dict().items()}, {k: v.clone() for k, v in criterion.state_dict().items()}
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = torch.randperm(len(train_ids), device=DEVICE, generator=generator)
        batches = list(order.split(BATCH))
        if len(batches[-1]) == 1:
            batches[-2] = torch.cat((batches[-2], batches.pop()))
        total = torch.zeros((), device=DEVICE)
        for b in batches:
            optimizer.zero_grad(set_to_none=True)
            _, predicted = model((train_x[b], train_s[b]))
            _, loss = criterion(predicted, DATA.y[train_a[b]], train_a[b])
            loss.backward()
            optimizer.step()
            total += loss.detach() * len(b)
        train_loss = total.item() / len(train_ids)
        if not np.isfinite(train_loss):
            raise RuntimeError(f'Non-finite loss at epoch {epoch}')
        score = evaluate_ids(model, criterion, None, loaded=val)
        history.append({'epoch': epoch, 'train_loss': train_loss, **{f'val_{k}': v for k, v in score.items()},
                        'temperature': criterion.temperature.exp().item()})
        if score['loss'] < best['loss']:
            best, best_epoch = score, epoch
            state = {k: v.clone() for k, v in model.state_dict().items()}, {k: v.clone() for k, v in criterion.state_dict().items()}
        if epoch % 10 == 0:
            print(json.dumps({'epoch': epoch, 'train_loss': round(train_loss, 4), 'val_loss': round(score['loss'], 4),
                              'val_top10': round(score['top10'], 4), 'best_epoch': best_epoch}), flush=True)
        if epoch - best_epoch >= PATIENCE:
            break
    model.load_state_dict(state[0])
    criterion.load_state_dict(state[1])
    torch.save({'model_state_dict': state[0], 'criterion_state_dict': state[1], 'config': config,
                'kind': kind, 'holdout': holdout, 'seed': seed, 'fit_subjects': fit_subjects,
                'ea_transforms': {s: t.cpu() for s, t in transforms.items()} if transforms else None,
                'epoch': best_epoch, 'validation': best, 'protocol_sha256': sha256(PROTOCOL)}, checkpoint)
    tests = {f'sub-{s:02d}': evaluate_ids(model, criterion, DATA.ids('test', [s]), transforms) for s in fit_subjects}
    result = {'kind': kind, 'holdout': holdout, 'seed': seed, 'fit_subjects': fit_subjects,
              'best_epoch': best_epoch, 'epochs_run': epoch, 'selected_validation': best,
              'test_by_subject': tests, 'seconds': time.perf_counter() - started,
              'deterministic': deterministic, 'checkpoint_sha256': sha256(checkpoint),
              'peak_gpu_bytes': torch.cuda.max_memory_allocated() if DEVICE.type == 'cuda' else None,
              'python': platform.python_version(), 'torch': torch.__version__, 'history': history}
    summary_path.write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps({k: result[k] for k in ('kind', 'holdout', 'seed', 'best_epoch', 'epochs_run', 'seconds')}))


class Calibrator:
    """One holdout on top of a trained shared model; the holdout matrix is re-parameterised."""

    def __init__(self, kind, holdout, seed):
        path, _ = model_path(kind, holdout, seed)
        self.saved = torch.load(path, map_location=DEVICE, weights_only=False)
        if holdout in self.saved['fit_subjects']:
            raise ValueError('Shared checkpoint saw the holdout subject')
        self.model, _ = build_model(DATA, seed)
        self.criterion = CLIPLoss(1.0).to(DEVICE)
        self.h = holdout - 1
        self.sources = [s - 1 for s in self.saved['fit_subjects']]
        self.reset()
        with torch.no_grad():
            self.mean_ws = self.model.spatial_module.subject_layer.A[self.sources].mean(dim=0).clone()

    def reset(self, init=None, method=None, lr=None):
        """Restore the checkpoint; set the holdout row to `init`; return the optimizer for `method`."""
        self.model.load_state_dict(self.saved['model_state_dict'])
        self.criterion.load_state_dict(self.saved['criterion_state_dict'])
        layer = self.model.spatial_module.subject_layer
        if 'forward' in layer.__dict__:
            del layer.forward
        if init is None:
            return None
        base = layer.A.detach().clone()
        base[self.h] = init
        shape = base.shape[1:]
        if method == 'ft_lowrank':
            u = torch.zeros(shape[0], RANK, device=DEVICE, requires_grad=True)
            v = (torch.randn(RANK, shape[1], device=DEVICE) / shape[1] ** 0.5).requires_grad_(True)
            delta, params = (lambda: u @ v), [u, v]
        else:
            d = torch.zeros(shape, device=DEVICE, requires_grad=method is not None)
            delta, params = (lambda: d), [d]
        h = self.h

        def forward(x, s):
            a = base[s] + (s == h).to(x.dtype)[:, None, None] * delta()[None]
            return torch.einsum('bji,bit->bjt', a, x)

        layer.forward = forward
        for p in list(self.model.parameters()) + list(self.criterion.parameters()):
            p.requires_grad_(False)
        if method is None:
            return None
        if method == 'ft_all':
            net = [p for n, p in self.model.named_parameters() if not n.endswith('subject_layer.A')]
            for p in net:
                p.requires_grad_(True)
            params = params + net
        return torch.optim.Adam(params, lr=lr)

    def run(self, method, x_cal, ids_cal, lr, steps, x_eval, eval_ids, checkpoints, generator):
        """Train on (x_cal, ids_cal) for max(checkpoints) steps; evaluate at each checkpoint."""
        optimizer = self.reset(self.mean_ws, method, lr)
        self.model.eval()  # BatchNorm statistics and dropout stay frozen
        audio = DATA.audio_index[ids_cal]
        subjects = torch.full((len(ids_cal),), self.h, dtype=torch.long, device=DEVICE)
        batch = min(BATCH, len(ids_cal))
        order, cursor, out = None, len(ids_cal), {}
        started = time.perf_counter()
        if 0 in checkpoints:  # no training: the mean_Ws starting point
            out[0] = dict(self.score(x_eval, eval_ids), cal_loss=float('nan'), calibration_seconds=0.0)
        for step in range(1, max(checkpoints) + 1):
            if cursor + batch > len(ids_cal):
                order, cursor = torch.randperm(len(ids_cal), device=DEVICE, generator=generator), 0
            b = order[cursor:cursor + batch]
            cursor += batch
            optimizer.zero_grad(set_to_none=True)
            _, predicted = self.model((x_cal[b], subjects[b]))
            _, loss = self.criterion(predicted, DATA.y[audio[b]], audio[b])
            loss.backward()
            optimizer.step()
            if step in checkpoints:
                out[step] = dict(self.score(x_eval, eval_ids), cal_loss=loss.item(),
                                 calibration_seconds=time.perf_counter() - started)
        return out

    def score(self, x, ids):
        split = DATA.windows.split.to_numpy()[ids]
        subjects = torch.full((len(ids),), self.h, dtype=torch.long, device=DEVICE)
        return evaluate(self.model, self.criterion, x, subjects, DATA.audio_index[ids], bank(split[0]))


def holdout_views(holdout, block, transform_from_block):
    """Holdout windows (all splits) scaled by the budget block (or population scaler if empty)."""
    rows = DATA.windows[DATA.windows.subject == holdout]
    ids = rows.window_id.to_numpy()
    if len(block):
        params = fit_scaler(DATA.unscaled[np.sort(block)])
    else:
        params = population_scaler(holdout)
    x = torch.from_numpy(apply_scaler(DATA.unscaled[ids], params)).to(DEVICE)
    position = {w: i for i, w in enumerate(ids)}
    transform = None
    if transform_from_block and len(block):
        transform = ea_matrix(x[[position[w] for w in block]])
        x = align(x, transform)
    return x, position


def population_scaler(holdout):
    """Per-channel median of the source subjects' affine scalers, expressed as scaler parameters."""
    offsets, scales = [], []
    for s in DATA.protocol['subjects']:
        if s == holdout:
            continue
        p = {k: np.asarray(v) for k, v in DATA.scalers[f'sub-{s:02d}'].items()}
        offsets.append(p['robust_center'] + p['standard_mean'] * p['robust_scale'])
        scales.append(p['robust_scale'] * p['standard_scale'])
    zeros = np.zeros(len(offsets[0]))
    return {'robust_center': np.median(offsets, axis=0), 'robust_scale': np.median(scales, axis=0),
            'standard_mean': zeros, 'standard_scale': zeros + 1}


def blocks(holdout, seed, budget):
    pool = DATA.pool(holdout)
    n = budget // DATA.protocol['window_seconds']
    if n == 0:
        return [np.array([], dtype=int)]
    if n >= len(pool):
        return [pool]
    rng = np.random.default_rng(seed * 1000 + holdout)
    return [pool[start:start + n] for start in rng.integers(0, len(pool) - n + 1, size=3)]


def select(x, position, ids):
    return x[[position[w] for w in ids]]


def tune(holdout, method):
    if holdout not in DATA.protocol['dev_subjects']:
        raise ValueError('Tuning is only allowed on dev subjects')
    seed = 0
    fix_seed(seed, torch_deterministic=False)
    kind = 'shared_ea' if method.startswith('ea_') else 'shared'
    cal = Calibrator(kind, holdout, seed)
    val_ids = np.sort(DATA.ids('validation', [holdout]))
    records = []
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    for budget in TUNE_BUDGETS:
        for r, block in enumerate(blocks(holdout, seed, budget)):
            x, position = holdout_views(holdout, block, method.startswith('ea_'))
            x_cal, x_val = select(x, position, block), select(x, position, val_ids)
            for lr in TUNE_LRS[method]:
                scores = cal.run(method, x_cal, block, lr, max(TUNE_STEPS), x_val, val_ids, TUNE_STEPS, generator)
                records += [{'holdout': holdout, 'method': method, 'budget_seconds': budget, 'repeat': r,
                             'lr': lr, 'steps': step, **score} for step, score in scores.items()]
    path = RES / 'tuning' / f'{method}_h{holdout:02d}.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(path, index=False)
    print(pd.DataFrame(records).groupby(['budget_seconds', 'lr', 'steps']).loss.mean().unstack().round(4).to_string())


def choose_hyperparameters():
    """Per method: lr minimising the mean validation loss; steps per budget group at that lr."""
    table = pd.concat([pd.read_csv(p) for p in sorted((RES / 'tuning').glob('*.csv'))])
    choice = {}
    for method, rows in table.groupby('method'):
        by_lr = rows.groupby(['lr', 'budget_seconds', 'steps']).loss.mean()
        best_per_budget = by_lr.groupby(['lr', 'budget_seconds']).min()
        lr = float(best_per_budget.groupby('lr').mean().idxmin())
        steps = {int(b): int(by_lr.loc[lr].loc[b].idxmin()) for b in TUNE_BUDGETS}
        choice[method] = {'lr': lr, 'steps_by_tuning_budget': steps,
                          'holdouts': sorted(rows.holdout.unique().tolist())}
    choice['pool_ft_Ws'] = choice['ft_Ws']
    (RES / 'tuning.json').write_text(json.dumps(choice, indent=2) + '\n')
    print(json.dumps(choice, indent=2))


def calibrate(holdout, seed):
    fix_seed(seed, torch_deterministic=False)
    tuning = json.loads((RES / 'tuning.json').read_text())
    shared, shared_ea = Calibrator('shared', holdout, seed), Calibrator('shared_ea', holdout, seed)
    test_ids = np.sort(DATA.ids('test', [holdout]))
    pool_x = torch.from_numpy(DATA.scaled[np.sort(DATA.windows[DATA.windows.subject == holdout].window_id)]).to(DEVICE)
    pool_pos = {w: i for i, w in enumerate(np.sort(DATA.windows[DATA.windows.subject == holdout].window_id))}
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    records = []

    def add(method, budget, repeat, block, score, **extra):
        records.append({'holdout': holdout, 'seed': seed, 'method': method, 'budget_seconds': budget,
                        'repeat': repeat, 'calibration_windows': len(block), **extra, **score})

    for budget in DATA.protocol['calibration_budgets_seconds']:
        for repeat, block in enumerate(blocks(holdout, seed, budget)):
            x, position = holdout_views(holdout, block, False)
            views = {'strict': (x, position), 'pool': (pool_x, pool_pos)}
            if len(block):
                views['ea'] = holdout_views(holdout, block, True)

            def rows(view, ids):
                return [views[view][1][w] for w in ids]

            def run(method, cal, view):
                hp = tuning[method]
                steps = hp['steps_by_tuning_budget'][str(STEP_GROUP[budget])]
                xv = views[view][0]
                score = cal.run(method, xv[rows(view, block)], block, hp['lr'], steps,
                                xv[rows(view, test_ids)], test_ids, (steps,), generator)[steps]
                add(method, budget, repeat, block, score, lr=hp['lr'], steps=steps)

            def zero_shot(method, cal, view, init):
                cal.reset(init)
                xv = views[view][0]
                add(method, budget, repeat, block, cal.score(xv[rows(view, test_ids)], test_ids))

            if budget == 0:
                cal = shared
                cal.reset(None)
                add('untrained_slot', 0, 0, block, cal.score(x[rows('strict', test_ids)], test_ids))
                zero_shot('ea_mean_Ws', shared_ea, 'strict', shared_ea.mean_ws)  # no data: no alignment
            zero_shot('mean_Ws', shared, 'strict', shared.mean_ws)
            if repeat == 0 and (budget == 0 or len(block) == len(DATA.pool(holdout))):
                zero_shot('pool_mean_Ws', shared, 'pool', shared.mean_ws)
            if not len(block):
                continue
            zero_shot('ea_mean_Ws', shared_ea, 'ea', shared_ea.mean_ws)
            run('ft_Ws', shared, 'strict')
            run('ft_lowrank', shared, 'strict')
            run('ft_all', shared, 'strict')
            run('ea_ft_Ws', shared_ea, 'ea')
            run('pool_ft_Ws', shared, 'pool')
        print(f'h{holdout:02d} s{seed} budget {budget} done', flush=True)
    path = RES / 'calibration' / f'h{holdout:02d}_s{seed}.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(path, index=False)
    table = pd.DataFrame(records).groupby(['method', 'budget_seconds']).top10.mean().unstack()
    print(table.round(3).to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['fit', 'tune', 'choose', 'calibrate'])
    parser.add_argument('--model', choices=['shared', 'shared_ea', 'ceiling'])
    parser.add_argument('--holdout', type=int)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--method', choices=sorted(TUNE_LRS))
    parser.add_argument('--deterministic', action='store_true')
    args = parser.parse_args()
    torch.backends.cudnn.benchmark = not args.deterministic
    if args.mode == 'choose':
        choose_hyperparameters()
    else:
        DATA = Data()
        if args.mode == 'fit':
            fit(args.model, args.holdout, args.seed, args.deterministic)
        elif args.mode == 'tune':
            tune(args.holdout, args.method)
        else:
            calibrate(args.holdout, args.seed)
