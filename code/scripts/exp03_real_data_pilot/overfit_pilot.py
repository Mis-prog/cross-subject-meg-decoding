"""Memorization diagnostic on 12 real pairs; no held-out evaluation."""

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import time
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/fqw-mpl')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from lisa.cli.parse_experiment_args import parse_experiment_arguments
from lisa.model.load_model import load_from_config
from lisa.training.criteria import CLIPLoss, retrieval_topk

ROOT = Path(__file__).resolve().parents[3]
PREPARED = ROOT / 'data/pilot/prepared'
RESULTS = ROOT / 'experiments/exp03_real_data_pilot/results'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    started = time.perf_counter()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    x = torch.from_numpy(np.load(PREPARED / 'meg_windows.npy'))
    y = torch.from_numpy(np.load(PREPARED / 'audio_embeddings.npy')).transpose(1, 2)
    windows = pd.read_csv(PREPARED / 'windows.csv')
    if x.shape != (12, 208, 300) or y.shape != (12, 768, 149):
        raise ValueError(f'Unexpected prepared shapes: {x.shape}, {y.shape}')
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError('Non-finite input')
    ids = torch.from_numpy(windows.window_id.to_numpy(dtype=np.int64))
    subjects = torch.from_numpy(windows.model_subject_index.to_numpy(dtype=np.int64))
    if ids.tolist() != list(range(12)) or subjects.tolist() != [0] * 12:
        raise ValueError('Unexpected window order or subjects')
    config = parse_experiment_arguments([
        '--run-group', 'fqw-real-data-overfit', '--device', 'cpu',
        '--seed', str(args.seed), '--spatial-dropout-number', '0',
        '--dirprocess', str(ROOT / 'code/LISA/data/preprocessed'),
    ])
    model, _ = load_from_config(config, None, torch.device('cpu'))
    criterion = CLIPLoss(1.0)
    optimizer = torch.optim.AdamW([
        {'params': model.parameters(), 'lr': 3e-4},
        {'params': criterion.parameters(), 'lr': 1e-3},
    ], weight_decay=0.0)

    def evaluate(step):
        model.eval()
        with torch.no_grad():
            _, predicted = model((x, subjects))
            unscaled, scaled = criterion(predicted, y, ids)
            # Request all candidates so ranks include misses outside Top-10.
            metrics = retrieval_topk(predicted, y, ids, ids, top_k=len(ids))
            ranks = metrics['rank_of_true'] + 1
        return {
            'step': step, 'eval_loss': scaled.item(),
            'eval_unscaled_loss': unscaled.item(),
            'top1': (ranks == 1).float().mean().item(),
            'top10': (ranks <= 10).float().mean().item(),
            'mean_rank': ranks.float().mean().item(),
            'temperature': criterion.temperature.exp().item(),
        }, ranks.tolist(), metrics['top_k_wav_indices'][:, 0].tolist()

    initial, _, _ = evaluate(0)
    history = [initial]
    first_perfect_step = None
    train_started = time.perf_counter()
    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        _, predicted = model((x, subjects))
        _, loss = criterion(predicted, y, ids)
        if not torch.isfinite(loss):
            raise RuntimeError(f'Non-finite loss at step {step}')
        loss.backward()
        gradients = [p.grad for group in optimizer.param_groups for p in group['params'] if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise RuntimeError(f'Invalid gradients at step {step}')
        optimizer.step()
        row, ranks, predictions = evaluate(step)
        row['train_loss_before_update'] = loss.item()
        history.append(row)
        if row['top1'] == 1.0 and first_perfect_step is None:
            first_perfect_step = step
        if step == 1 or step % 25 == 0 or step == args.steps:
            print(json.dumps(row), flush=True)
    training_seconds = time.perf_counter() - train_started
    final = history[-1]
    passed = final['top1'] == 1.0 and final['eval_loss'] < initial['eval_loss']
    checkpoint = ROOT / 'data/pilot/overfit_checkpoint.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'criterion_state_dict': criterion.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config, 'step': args.steps, 'seed': args.seed,
        'scope': 'Same-batch memorization diagnostic; not a transferable baseline',
    }, checkpoint)
    # Verify the saved weights reproduce the recorded evaluation.
    restored, _ = load_from_config(config, checkpoint, torch.device('cpu'))
    restored.eval()
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored((x, subjects))[1], model((x, subjects))[1], rtol=0, atol=0)
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(RESULTS / 'real_data_overfit_history.csv', index=False)
    windows.assign(true_rank=ranks, predicted_window_id=predictions).to_csv(
        RESULTS / 'real_data_overfit_predictions.csv', index=False)
    checksums = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(PREPARED.iterdir()) if p.is_file()
    }
    result = {
        'status': 'passed' if passed else 'not_converged',
        'scope': 'Training and evaluation use the same 12 real pairs. Memorization only; no generalization or alignment proof.',
        'success_criterion': 'Final same-batch Top-1 = 1.0 and evaluation loss below initialization',
        'upstream_commit': subprocess.check_output(['git', '-C', str(ROOT / 'code/LISA'), 'rev-parse', 'HEAD'], text=True).strip(),
        'python': platform.python_version(), 'torch': torch.__version__,
        'device': 'cpu', 'threads': 4, 'seed': args.seed,
        'deterministic_algorithms': True, 'steps': args.steps,
        'batch_size': len(ids), 'candidate_count': len(ids),
        'spatial_dropout_number': 0, 'model_lr': 3e-4,
        'temperature_lr': 1e-3, 'weight_decay': 0.0,
        'uniform_random_top1': 1 / len(ids), 'uniform_random_top10': 10 / len(ids),
        'initial': initial, 'final': final,
        'first_perfect_evaluated_step': first_perfect_step,
        'training_with_evaluation_seconds': training_seconds,
        'total_seconds': time.perf_counter() - started,
        'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == 'Darwin' else 1024),
        'checkpoint': str(checkpoint.relative_to(ROOT)),
        'checkpoint_reload_exact_match': True,
        'prepared_sha256': checksums,
        'model_config': {k: str(v) if isinstance(v, (torch.device, Path)) else v for k, v in config.items()},
    }
    (RESULTS / 'real_data_overfit.json').write_text(json.dumps(result, indent=2) + '\n')
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout='constrained')
    steps = [r['step'] for r in history]
    axes[0].plot(steps, [r['eval_loss'] for r in history])
    axes[0].set(xlabel='Optimizer step', ylabel='CLIP loss (evaluation mode)')
    axes[1].plot(steps, [r['top1'] for r in history], label='Top-1 on training pairs')
    axes[1].axhline(1 / len(ids), linestyle='--', color='gray', label='Uniform random: 1/12')
    axes[1].set(xlabel='Optimizer step', ylabel='Retrieval accuracy', ylim=(0, 1.05))
    axes[1].legend(loc='lower right', fontsize=8)
    fig.suptitle('LISA: memorization of 12 real MEG/audio pairs — no held-out test')
    fig.savefig(RESULTS / 'real_data_overfit.png', dpi=160)
    plt.close(fig)
    print(json.dumps({k: v for k, v in result.items() if k not in ('model_config', 'prepared_sha256')}, indent=2))


if __name__ == '__main__':
    main()
