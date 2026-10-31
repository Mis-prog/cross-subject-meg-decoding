"""Mini-LOSO pilot: shared model on source subjects, ceiling on all, and subject-layer calibrators.

Usage (from code/scripts/exp05_mini_loso):
  python train_mini_loso.py shared     # fit on source subjects, select on their validation
  python train_mini_loso.py ceiling    # fit on all subjects incl. holdout (reference only)
  python train_mini_loso.py calibrate  # holdout calibrators on top of the shared checkpoint
"""
import argparse
import json
import os
import platform
import resource
import time
import sys
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/fqw-mpl')

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from lisa.cli.parse_experiment_args import parse_experiment_arguments
from lisa.model.load_model import load_from_config
from lisa.training.criteria import CLIPLoss, get_clip_similarity, retrieval_topk
from lisa.utils.determinism import fix_seed
# Support direct execution from any working directory.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.exp04_small_baseline.prepare_baseline import sha256
from scripts.exp05_mini_loso.prepare_mini_loso import OUT, PROTOCOL, RESULTS, ROOT

RUNS = ROOT / 'data/pilot/mini_loso_runs'
EPOCHS, BATCH, EVAL_BATCH = 50, 64, 32
LR, TEMP_LR = 3e-4, 1e-3
CAL_STEPS, CAL_LR, CAL_BATCH = 200, 1e-2, 64


def load_data():
    protocol = json.loads(PROTOCOL.read_text())
    prep = json.loads((RESULTS / 'mini_loso_preparation.json').read_text())
    if sha256(PROTOCOL) != prep['protocol_sha256']:
        raise ValueError('Protocol changed since preparation')
    for name, expected in prep['prepared_sha256'].items():
        if sha256(OUT / name) != expected:
            raise ValueError(f'Prepared file changed: {name}')
    windows = pd.read_csv(OUT / 'meg_windows.csv')
    x = torch.from_numpy(np.load(OUT / 'meg_windows.npy'))
    y = torch.from_numpy(np.load(OUT / 'audio_embeddings.npy')).transpose(1, 2)
    return protocol, windows, x, y


def build_model(protocol):
    config = parse_experiment_arguments([
        '--run-group', protocol['name'], '--device', 'cpu', '--seed', str(protocol['seed']),
        '--spatial-dropout-number', '0', '--dirprocess', str(ROOT / 'code/LISA/data/preprocessed'),
    ])
    model, _ = load_from_config(config, None, torch.device('cpu'))
    return model, config


def ids_of(windows, split, subjects):
    rows = windows[(windows.split == split) & windows.subject.isin(subjects)]
    return torch.as_tensor(rows.window_id.to_numpy(), dtype=torch.long)


def evaluate(model, criterion, windows, x, y, query_ids):
    """Rank every query against all unique audio windows of its split (one shared bank)."""
    model.eval()
    rows = windows.set_index('window_id').loc[query_ids.numpy()]
    if rows.split.nunique() != 1:
        raise ValueError('Queries must come from one split')
    bank_ids = torch.as_tensor(np.sort(windows[windows.split == rows.split.iloc[0]].audio_id.unique()))
    candidates = y[bank_ids]
    subjects = torch.as_tensor(rows.model_subject_index.to_numpy(), dtype=torch.long)
    labels_all = torch.as_tensor(rows.audio_id.to_numpy(), dtype=torch.long)
    records = []
    with torch.no_grad():
        for chunk in torch.arange(len(query_ids)).split(EVAL_BATCH):
            _, predicted = model((x[query_ids[chunk]], subjects[chunk]))
            labels = labels_all[chunk]
            target = (labels[:, None] == bank_ids[None, :]).long().argmax(dim=1)
            _, _, similarity = get_clip_similarity(predicted, candidates)
            losses = F.cross_entropy(similarity / criterion.temperature.exp(), target, reduction='none')
            ranks = retrieval_topk(predicted, candidates, labels, bank_ids, top_k=len(bank_ids))['rank_of_true'] + 1
            records += [{'window_id': int(w), 'subject': int(s), 'loss': float(l), 'rank': int(r)}
                        for w, s, l, r in zip(query_ids[chunk], rows.subject.to_numpy()[chunk.numpy()], losses, ranks)]
    frame = pd.DataFrame(records)
    summary = {'loss': float(frame.loss.mean()), 'top1': float(frame['rank'].eq(1).mean()),
               'top10': float(frame['rank'].le(10).mean()), 'mean_rank': float(frame['rank'].mean()),
               'queries': len(frame), 'candidates': len(bank_ids),
               'chance_top10': min(10 / len(bank_ids), 1.0)}
    return summary, frame


def fit(mode):
    protocol, windows, x, y = load_data()
    fix_seed(protocol['seed'], torch_deterministic=True)
    torch.set_num_threads(4)
    holdout = protocol['holdout_subject']
    sources = [s for s in protocol['subjects'] if s != holdout]
    fit_subjects = sources if mode == 'shared' else protocol['subjects']
    model, config = build_model(protocol)
    criterion = CLIPLoss(1.0)
    optimizer = torch.optim.AdamW([
        {'params': model.parameters(), 'lr': LR},
        {'params': criterion.parameters(), 'lr': TEMP_LR},
    ], weight_decay=0.0)
    train_ids = ids_of(windows, 'train', fit_subjects)
    val_ids = ids_of(windows, 'validation', fit_subjects)
    subject_index = torch.as_tensor(windows.model_subject_index.to_numpy(), dtype=torch.long)
    audio_index = torch.as_tensor(windows.audio_id.to_numpy(), dtype=torch.long)
    run = RUNS / mode
    run.mkdir(parents=True, exist_ok=True)

    def save(epoch, metrics):
        torch.save({'model_state_dict': model.state_dict(), 'criterion_state_dict': criterion.state_dict(),
                    'config': config, 'protocol_sha256': sha256(PROTOCOL), 'mode': mode,
                    'fit_subjects': fit_subjects, 'epoch': epoch, 'validation': metrics}, run / 'best.pt')

    started = time.perf_counter()
    val, _ = evaluate(model, criterion, windows, x, y, val_ids)
    best, best_epoch = val['loss'], 0
    save(0, val)
    history = [{'epoch': 0, **{f'val_{k}': v for k, v in val.items()}}]
    for epoch in range(1, EPOCHS + 1):
        model.train()
        batches = list(train_ids[torch.randperm(len(train_ids))].split(BATCH))
        if len(batches[-1]) == 1:
            batches[-2] = torch.cat((batches[-2], batches.pop()))
        total = 0.0
        for b in batches:
            optimizer.zero_grad(set_to_none=True)
            _, predicted = model((x[b], subject_index[b]))
            _, loss = criterion(predicted, y[audio_index[b]], audio_index[b])
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite loss at epoch {epoch}')
            loss.backward()
            optimizer.step()
            total += loss.item() * len(b)
        val, _ = evaluate(model, criterion, windows, x, y, val_ids)
        if val['loss'] < best:
            best, best_epoch = val['loss'], epoch
            save(epoch, val)
        history.append({'epoch': epoch, 'train_loss': total / len(train_ids),
                        **{f'val_{k}': v for k, v in val.items()},
                        'temperature': criterion.temperature.exp().item()})
        if epoch == 1 or epoch % 5 == 0:
            print(json.dumps({'epoch': epoch, 'train_loss': round(total / len(train_ids), 4),
                              'val_loss': round(val['loss'], 4), 'val_top10': round(val['top10'], 4),
                              'best_epoch': best_epoch}), flush=True)
    pd.DataFrame(history).to_csv(RESULTS / f'mini_loso_{mode}_history.csv', index=False)
    saved = torch.load(run / 'best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(saved['model_state_dict'])
    criterion.load_state_dict(saved['criterion_state_dict'])
    tests = {}
    for subject in protocol['subjects']:
        tests[f'sub-{subject:02d}'], _ = evaluate(model, criterion, windows, x, y, ids_of(windows, 'test', [subject]))
    result = {'mode': mode, 'fit_subjects': fit_subjects, 'holdout_subject': holdout,
              'best_epoch': best_epoch, 'selected_validation': saved['validation'], 'test_by_subject': tests,
              'note': 'shared: holdout row uses its untrained subject-layer slot (sanity floor)' if mode == 'shared' else 'ceiling: holdout train windows were used for fitting',
              'seconds': time.perf_counter() - started, 'checkpoint_sha256': sha256(run / 'best.pt'),
              'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'python': platform.python_version(), 'torch': torch.__version__}
    (RESULTS / f'mini_loso_{mode}.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['test_by_subject'], indent=2))


def calibrate():
    protocol, windows, x, y = load_data()
    fix_seed(protocol['seed'], torch_deterministic=True)
    torch.set_num_threads(4)
    holdout = protocol['holdout_subject']
    saved = torch.load(RUNS / 'shared' / 'best.pt', map_location='cpu', weights_only=False)
    if holdout in saved['fit_subjects']:
        raise ValueError('Shared checkpoint saw the holdout subject')
    model, _ = build_model(protocol)
    criterion = CLIPLoss(1.0)
    rows = windows[windows.subject == holdout]
    h_index = int(rows.model_subject_index.iloc[0])
    src_index = sorted(windows[windows.subject != holdout].model_subject_index.unique().tolist())
    pool = rows[rows.split == 'train'].sort_values(['task', 'meg_start_sample_1000hz'])
    test_ids = ids_of(windows, 'test', [holdout])
    audio_index = torch.as_tensor(windows.audio_id.to_numpy(), dtype=torch.long)

    def reset():
        model.load_state_dict(saved['model_state_dict'])
        criterion.load_state_dict(saved['criterion_state_dict'])
        return model.spatial_module.subject_layer.A

    results, curves = [], []
    A = reset()
    summary, _ = evaluate(model, criterion, windows, x, y, test_ids)
    results.append({'condition': 'untrained_slot', 'budget_seconds': 0, **summary})
    with torch.no_grad():
        mean_ws = A[src_index].mean(dim=0).clone()

    for budget in protocol['calibration_budgets_seconds']:
        A = reset()
        with torch.no_grad():
            A[h_index] = mean_ws
        if budget == 0:
            summary, _ = evaluate(model, criterion, windows, x, y, test_ids)
            results.append({'condition': 'mean_Ws', 'budget_seconds': 0, **summary})
            continue
        n_windows = budget // protocol['window_seconds']
        cal_ids = torch.as_tensor(pool.window_id.to_numpy()[:n_windows], dtype=torch.long)
        before = A.detach().clone()
        for p in list(model.parameters()) + list(criterion.parameters()):
            p.requires_grad_(False)
        A.requires_grad_(True)
        optimizer = torch.optim.Adam([A], lr=CAL_LR)
        model.eval()  # freeze BatchNorm statistics; only the holdout slice is updated
        subjects = torch.full((len(cal_ids),), h_index, dtype=torch.long)
        started = time.perf_counter()
        for step in range(1, CAL_STEPS + 1):
            order = torch.randperm(len(cal_ids))
            for chunk in order.split(CAL_BATCH):
                if len(chunk) < 2:
                    continue
                optimizer.zero_grad(set_to_none=True)
                _, predicted = model((x[cal_ids[chunk]], subjects[chunk]))
                _, loss = criterion(predicted, y[audio_index[cal_ids[chunk]]], audio_index[cal_ids[chunk]])
                loss.backward()
                A.grad[torch.arange(A.shape[0]) != h_index] = 0
                optimizer.step()
            if step % 20 == 0:
                curves.append({'budget_seconds': budget, 'step': step, 'cal_loss': loss.item()})
        with torch.no_grad():
            others = torch.arange(A.shape[0]) != h_index
            if not torch.equal(A[others], before[others]):
                raise RuntimeError('Calibration changed other subjects')
        A.requires_grad_(False)
        summary, _ = evaluate(model, criterion, windows, x, y, test_ids)
        results.append({'condition': 'finetune_Ws_from_mean', 'budget_seconds': budget,
                        'calibration_windows': len(cal_ids), 'calibration_seconds': time.perf_counter() - started,
                        **summary})
        print(json.dumps(results[-1]), flush=True)
    table = pd.DataFrame(results)
    table.to_csv(RESULTS / 'mini_loso_calibration.csv', index=False)
    pd.DataFrame(curves).to_csv(RESULTS / 'mini_loso_calibration_curves.csv', index=False)
    (RESULTS / 'mini_loso_calibration.json').write_text(json.dumps({
        'holdout_subject': holdout, 'shared_checkpoint_sha256': sha256(RUNS / 'shared' / 'best.pt'),
        'steps': CAL_STEPS, 'lr': CAL_LR, 'batch': CAL_BATCH,
        'pool_order': 'holdout train-split windows sorted by task then recording time; first budget/3 windows',
        'tuning': 'none; fixed schedule, test used once per condition',
        'results': results}, indent=2) + '\n')
    print(table[['condition', 'budget_seconds', 'top1', 'top10', 'mean_rank']].to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['shared', 'ceiling', 'calibrate'])
    args = parser.parse_args()
    calibrate() if args.mode == 'calibrate' else fit(args.mode)
