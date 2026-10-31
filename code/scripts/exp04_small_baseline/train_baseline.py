"""Fixed-budget, one-subject LISA baseline with validation-only selection."""
import json
import os
import platform
import resource
import time
import sys
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/fqw-mpl')

import matplotlib.pyplot as plt
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

from scripts.exp04_small_baseline.prepare_baseline import OUT, PROTOCOL, ROOT, sha256, validate_windows

RESULTS = ROOT / 'experiments/exp04_small_baseline/results'
RUN = ROOT / 'data/pilot/baseline_run'


def score_embeddings(predicted, candidates, labels, candidate_ids, temperature):
    """Score against the entire split bank, including ranks beyond Top-10."""
    matches = labels[:, None] == candidate_ids[None, :]
    if not (matches.sum(dim=1) == 1).all():
        raise ValueError('Each query must have exactly one target in the bank')
    _, _, similarity = get_clip_similarity(predicted, candidates)
    losses = F.cross_entropy(similarity / temperature, matches.long().argmax(dim=1), reduction='none')
    ranking = retrieval_topk(predicted, candidates, labels, candidate_ids, top_k=len(candidate_ids))
    ranks = ranking['rank_of_true'] + 1
    return losses, ranks, ranking['top_k_wav_indices'][:, 0]


def main():
    started = time.perf_counter()
    protocol = json.loads(PROTOCOL.read_text())
    preparation = json.loads((RESULTS / 'baseline_preparation.json').read_text())
    if sha256(PROTOCOL) != preparation['protocol_sha256']:
        raise ValueError('Protocol changed since preparation')
    for name, expected in preparation['prepared_sha256'].items():
        if sha256(OUT / name) != expected:
            raise ValueError(f'Prepared file changed: {name}')
    torch.set_num_threads(4)
    fix_seed(protocol['seed'], torch_deterministic=True)
    windows = pd.read_csv(OUT / 'windows.csv')
    validate_windows(windows)
    scalers = json.loads((OUT / 'scalers.json').read_text())
    split_ids = {name: rows.window_id.to_numpy() for name, rows in windows.groupby('split')}
    if scalers['fit_window_ids'] != split_ids['train'].tolist():
        raise ValueError('Scaler fit rows disagree with train split')
    x = torch.from_numpy(np.load(OUT / 'meg_windows.npy'))
    y = torch.from_numpy(np.load(OUT / 'audio_embeddings.npy')).transpose(1, 2)
    subjects = torch.from_numpy(windows.model_subject_index.to_numpy(dtype=np.int64))
    config = parse_experiment_arguments([
        '--run-group', protocol['name'], '--device', 'cpu',
        '--seed', str(protocol['seed']), '--spatial-dropout-number', '0',
        '--dirprocess', str(ROOT / 'code/LISA/data/preprocessed'),
    ])
    model, _ = load_from_config(config, None, torch.device('cpu'))
    criterion = CLIPLoss(1.0)
    optimizer = torch.optim.AdamW([
        {'params': model.parameters(), 'lr': protocol['learning_rate']},
        {'params': criterion.parameters(), 'lr': protocol['temperature_learning_rate']},
    ], weight_decay=protocol['weight_decay'])
    RUN.mkdir(parents=True, exist_ok=True)
    checkpoint = RUN / 'best.pt'

    def evaluate(split):
        model.eval()
        ids = torch.as_tensor(split_ids[split], dtype=torch.long)
        candidates = y[ids]
        records = []
        with torch.no_grad():
            for batch_ids in ids.split(protocol['evaluation_batch_size']):
                _, predicted = model((x[batch_ids], subjects[batch_ids]))
                losses, ranks, top_ids = score_embeddings(predicted, candidates, batch_ids, ids, criterion.temperature.exp())
                records.extend({
                    'window_id': int(idx), 'loss': float(loss),
                    'true_rank': int(rank), 'predicted_window_id': int(pred),
                } for idx, loss, rank, pred in zip(batch_ids, losses, ranks, top_ids))
        frame = pd.DataFrame(records)
        if not np.isfinite(frame.loss).all():
            raise RuntimeError('Non-finite evaluation loss')
        return {
            'loss': float(frame.loss.mean()),
            'top1': float(frame.true_rank.eq(1).mean()),
            'top10': float(frame.true_rank.le(10).mean()),
            'mean_rank': float(frame.true_rank.mean()),
            'median_rank': float(frame.true_rank.median()),
            'candidate_count': len(ids), 'query_count': len(ids),
            'uniform_random_top1': 1 / len(ids), 'uniform_random_top10': min(10 / len(ids), 1),
            'uniform_random_mean_rank': (len(ids) + 1) / 2,
        }, frame

    def save_checkpoint(epoch, metrics):
        torch.save({
            'model_state_dict': model.state_dict(), 'criterion_state_dict': criterion.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 'config': config,
            'protocol': protocol, 'protocol_sha256': sha256(PROTOCOL),
            'epoch': epoch, 'validation': metrics,
        }, checkpoint)

    validation, _ = evaluate('validation')
    initial_validation = validation.copy()
    best_loss, best_epoch = validation['loss'], 0
    save_checkpoint(0, validation)
    history = [{'epoch': 0, **{f'validation_{k}': v for k, v in validation.items()}, 'temperature': 1.0}]
    train_started = time.perf_counter()
    for epoch in range(1, protocol['epochs'] + 1):
        epoch_started = time.perf_counter()
        model.train()
        ids = torch.as_tensor(split_ids['train'], dtype=torch.long)
        shuffled = ids[torch.randperm(len(ids))]
        batches = list(shuffled.split(protocol['batch_size']))
        if len(batches[-1]) == 1:
            batches[-2] = torch.cat((batches[-2], batches[-1]))
            batches.pop()
        total_loss = 0.0
        for batch_ids in batches:
            optimizer.zero_grad(set_to_none=True)
            _, predicted = model((x[batch_ids], subjects[batch_ids]))
            _, loss = criterion(predicted, y[batch_ids], batch_ids)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite loss in epoch {epoch}')
            loss.backward()
            gradients = [p.grad for group in optimizer.param_groups for p in group['params'] if p.grad is not None]
            if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                raise RuntimeError(f'Non-finite gradients in epoch {epoch}')
            optimizer.step()
            total_loss += loss.item() * len(batch_ids)
        validation, _ = evaluate('validation')
        if validation['loss'] < best_loss:
            best_loss, best_epoch = validation['loss'], epoch
            save_checkpoint(epoch, validation)
        row = {
            'epoch': epoch, 'train_batch_loss': total_loss / len(ids),
            **{f'validation_{k}': v for k, v in validation.items()},
            'temperature': criterion.temperature.exp().item(),
            'epoch_seconds': time.perf_counter() - epoch_started,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(RESULTS / 'baseline_history.csv', index=False)
        if epoch == 1 or epoch % 5 == 0:
            print(json.dumps({'epoch': epoch, 'train_batch_loss': row['train_batch_loss'], 'validation': validation, 'best_epoch': best_epoch}), flush=True)
    training_seconds = time.perf_counter() - train_started
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(saved['model_state_dict'])
    criterion.load_state_dict(saved['criterion_state_dict'])
    selected_validation, _ = evaluate('validation')
    if selected_validation != saved['validation']:
        raise RuntimeError('Reloaded checkpoint validation disagrees with saved metrics')
    # This is the first access to test metrics, after checkpoint selection is final.
    test, test_predictions = evaluate('test')
    train, _ = evaluate('train')
    test_predictions = windows.merge(test_predictions, on='window_id', how='inner', validate='one_to_one')
    test_predictions.to_csv(RESULTS / 'baseline_test_predictions.csv', index=False)
    fragment_metrics = []
    for fname, rows in test_predictions.groupby('audio_file'):
        fragment_metrics.append({
            'audio_file': fname, 'queries': len(rows), 'candidate_count': test['candidate_count'],
            'top1': float(rows.true_rank.eq(1).mean()), 'top10': float(rows.true_rank.le(10).mean()),
            'mean_rank': float(rows.true_rank.mean()),
        })
    result = {
        'status': 'completed', 'scope': protocol['scope'],
        'protocol_sha256': sha256(PROTOCOL), 'preparation_report_sha256': sha256(RESULTS / 'baseline_preparation.json'),
        'upstream_commit': preparation['upstream_commit'],
        'seed': protocol['seed'], 'device': 'cpu', 'threads': 4,
        'python': platform.python_version(), 'torch': torch.__version__,
        'epochs': protocol['epochs'], 'best_epoch': best_epoch,
        'checkpoint_selection': protocol['checkpoint_selection'], 'test_evaluations': 1,
        'initial_validation': initial_validation, 'selected_validation': selected_validation,
        'selected_train': train, 'test': test, 'test_by_fragment': fragment_metrics,
        'training_with_validation_seconds': training_seconds,
        'total_seconds': time.perf_counter() - started,
        'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == 'Darwin' else 1024),
        'checkpoint': str(checkpoint.relative_to(ROOT)), 'checkpoint_sha256': sha256(checkpoint),
        'checkpoint_reload_validation_exact_match': True,
        'preprocessing': protocol['preprocessing'],
        'model_config': {k: str(v) if isinstance(v, (torch.device, Path)) else v for k, v in config.items()},
    }
    (RESULTS / 'baseline_result.json').write_text(json.dumps(result, indent=2) + '\n')
    hist = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), layout='constrained')
    axes[0].plot(hist.epoch, hist.train_batch_loss)
    axes[0].set(xlabel='Epoch', ylabel='Training loss (batch candidates)')
    axes[1].plot(hist.epoch, hist.validation_loss)
    axes[1].axhline(np.log(len(split_ids['validation'])), color='gray', ls='--', label='Uniform predictions')
    axes[1].axvline(best_epoch, color='tab:orange', ls=':', label=f'Selected epoch {best_epoch}')
    axes[1].set(xlabel='Epoch', ylabel='Validation loss (complete bank)')
    axes[1].legend(fontsize=8)
    axes[2].plot(hist.epoch, hist.validation_top1, label='Validation Top-1')
    axes[2].axhline(initial_validation['uniform_random_top1'], color='gray', ls='--', label='Uniform random')
    axes[2].set(xlabel='Epoch', ylabel='Retrieval accuracy', ylim=(0, max(0.1, hist.validation_top1.max() * 1.2)))
    axes[2].legend(fontsize=8)
    fig.suptitle('LISA: one subject, held-out audio fragments; test used after selection')
    fig.savefig(RESULTS / 'baseline_training.png', dpi=160)
    plt.close(fig)
    print(json.dumps({k: v for k, v in result.items() if k != 'model_config'}, indent=2))


if __name__ == '__main__':
    main()
