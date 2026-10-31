"""Offline CPU check of the upstream LISA model; synthetic data only."""

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import torch

from lisa.cli.parse_experiment_args import parse_experiment_arguments
from lisa.model.load_model import load_from_config
from lisa.training.criteria import CLIPLoss, retrieval_topk


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    upstream = Path(__file__).resolve().parents[2] / 'LISA'
    torch.set_num_threads(2)
    torch.manual_seed(42)
    config = parse_experiment_arguments([
        '--run-group', 'fqw-offline-smoke', '--device', 'cpu',
        '--dirprocess', str(upstream / 'data/preprocessed'),
    ])
    start = time.perf_counter()
    model, _ = load_from_config(config, None, torch.device('cpu'))
    meg = torch.randn(12, 208, 300)
    subjects = torch.arange(12, dtype=torch.long) % 3
    candidate_ids = torch.arange(12, dtype=torch.long)
    model.eval()
    with torch.no_grad():
        interpretable, predicted = model((meg, subjects))
    # Random targets exercise the retrieval plumbing, not neural decoding quality.
    targets = torch.randn_like(predicted)
    criterion = CLIPLoss(config['clip_temperature'])
    optimizer = torch.optim.AdamW([
        {'params': model.parameters(), 'lr': config['lr_fe']},
        {'params': criterion.parameters(), 'lr': config['clip_temperature_lr']},
    ], weight_decay=0.0)
    losses = []
    for _ in range(3):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        _, embedding = model((meg, subjects))
        _, loss = criterion(embedding, targets, candidate_ids)
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite synthetic loss')
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise RuntimeError('Missing or non-finite model gradients')
        optimizer.step()
        losses.append(loss.item())
    model.eval()
    with torch.no_grad():
        _, embedding = model((meg, subjects))
        metrics = retrieval_topk(embedding, targets, candidate_ids, candidate_ids, 10)
        try:
            model((meg, torch.full_like(subjects, 27)))
        except ValueError as exc:
            unseen_error = str(exc)
        else:
            raise RuntimeError('Expected out-of-range subject ID to be rejected')
    result = {
        'scope': 'Offline synthetic CPU forward/backward/retrieval check; not MEG-MASC evaluation',
        'upstream_commit': subprocess.check_output(
            ['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True
        ).strip(),
        'python': platform.python_version(), 'platform': platform.platform(),
        'torch': torch.__version__, 'device': 'cpu', 'threads': 2, 'seed': 42,
        'input_shape': list(meg.shape),
        'interpretable_shape': list(interpretable.shape),
        'embedding_shape': list(embedding.shape),
        'parameters': sum(p.numel() for p in model.parameters()),
        'subject_matrix_shape': list(model.spatial_module.subject_layer.A.shape),
        'optimizer_steps': 3, 'synthetic_losses': losses,
        'synthetic_top1_among_12': metrics['is_in_top1'].float().mean().item(),
        'synthetic_top10_among_12': metrics['is_in_topk'].float().mean().item(),
        'unseen_id_27_error': unseen_error,
        'elapsed_seconds': time.perf_counter() - start,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
