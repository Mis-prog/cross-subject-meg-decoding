"""Measure synthetic LISA training steps on CPU and Apple GPU, without data downloads."""

import json
import time
from pathlib import Path

import torch

from lisa.cli.parse_experiment_args import parse_experiment_arguments
from lisa.model.load_model import load_from_config
from lisa.training.criteria import CLIPLoss


def measure(device_name, batch_size):
    device = torch.device(device_name)
    torch.manual_seed(42)
    config = parse_experiment_arguments([
        '--run-group', 'fqw-device-check', '--device', 'cpu',
        '--dirprocess', str(Path(__file__).resolve().parents[2] / 'LISA/data/preprocessed'),
    ])
    # The upstream CLI rejects MPS; test its model API directly.
    model, _ = load_from_config(config, None, device)
    model.train()
    criterion = CLIPLoss(1.0).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()), lr=3e-4,
        weight_decay=0.0,
    )
    meg = torch.randn(batch_size, 208, 300, device=device)
    subjects = torch.arange(batch_size, device=device) % 27
    labels = torch.arange(batch_size, device=device)
    targets = torch.randn(batch_size, 768, 149, device=device)

    def sync():
        if device_name == 'mps':
            torch.mps.synchronize()

    def step():
        optimizer.zero_grad(set_to_none=True)
        _, embedding = model((meg, subjects))
        _, loss = criterion(embedding, targets, labels)
        loss.backward()
        optimizer.step()
        sync()
        value = loss.item()
        if not torch.isfinite(loss).item():
            raise RuntimeError('Non-finite loss')
        return value

    for _ in range(2):
        step()
    start = time.perf_counter()
    losses = [step() for _ in range(3)]
    elapsed = time.perf_counter() - start
    row = {
        'device': device_name, 'batch_size': batch_size,
        'warmup_steps': 2, 'measured_steps': 3,
        'seconds_per_step': elapsed / 3,
        'windows_per_second': batch_size * 3 / elapsed,
        'losses': losses,
    }
    if device_name == 'mps':
        row['mps_current_allocated_bytes'] = torch.mps.current_allocated_memory()
        row['mps_driver_allocated_bytes'] = torch.mps.driver_allocated_memory()
    return row


def main():
    torch.set_num_threads(4)
    results = []
    for device in ('cpu', 'mps'):
        if device == 'mps' and not torch.backends.mps.is_available():
            results.append({'device': device, 'error': 'MPS unavailable in this process'})
            continue
        for batch_size in (12, 100):
            try:
                row = measure(device, batch_size)
            except Exception as exc:
                row = {'device': device, 'batch_size': batch_size,
                       'error': f'{type(exc).__name__}: {exc}'}
            print(json.dumps(row), flush=True)
            results.append(row)
            if device == 'mps':
                torch.mps.empty_cache()
    output = Path(__file__).resolve().parents[3] / 'experiments/exp00_lisa_readiness/results/lisa_device_benchmark.json'
    output.write_text(json.dumps({
        'scope': 'Synthetic repeated-batch training; excludes data loading, wav2vec, evaluation and preprocessing',
        'torch': torch.__version__, 'cpu_threads': 4, 'results': results,
    }, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
