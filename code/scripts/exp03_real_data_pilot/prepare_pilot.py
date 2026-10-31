"""Prepare 12 real MEG/audio pairs and check LISA training, without a scientific evaluation."""
import ast
import json
import os
import platform
import subprocess
import time
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/fqw-mpl')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import mne
import numpy as np
import pandas as pd
import torch

from lisa.cli.parse_experiment_args import parse_experiment_arguments
from lisa.data.meg_io import load_raw_meg, preprocess_meg
from lisa.data.generate_all_embeddings import load_wav2vec_model, extract_wav2vec_embedding
from lisa.utils.audio_io import load_audio_mono_16k
from lisa.model.load_model import load_from_config
from lisa.training.criteria import CLIPLoss, retrieval_topk

ROOT = Path(__file__).resolve().parents[3]
PILOT = ROOT / 'data/pilot'
OUT = PILOT / 'prepared'
REPORT = ROOT / 'experiments/exp03_real_data_pilot/results'


def main():
    start = time.perf_counter()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(PILOT / 'raw/sub-01/ses-0/meg/sub-01_ses-0_task-0_events.tsv', sep='\t')
    matches = []
    for row in events.itertuples():
        event = ast.literal_eval(row.trial_type)
        if event.get('kind') == 'sound' and event.get('sound_id') == 0:
            matches.append((float(row.onset), event))
    assert len(matches) == 1, matches
    onset, sound_event = matches[0]
    assert sound_event['sound'].endswith('lw1_0.0.wav')
    print('Loading one BIDS recording...', flush=True)
    raw, sfreq = load_raw_meg('bids', str(PILOT / 'raw'), 1, 0, 0)
    assert len(raw.ch_names) == 208 and sfreq == 1000
    original_samples = raw.n_times
    original_duration = raw.n_times / sfreq
    ica = mne.preprocessing.read_ica(PILOT / 'ica/sub-01/ses-0/meg/sub-01_ses-0_task-0_meg-ica.fif', verbose=False)
    assert raw.ch_names == ica.ch_names, 'ICA channel order mismatch'
    print('Applying author ICA; excluded components:', ica.exclude, flush=True)
    ica.apply(raw, verbose=False)
    print('Resampling and normalizing...', flush=True)
    # Infrastructure pilot only: normalization uses this recording, no held-out evaluation.
    meg, _ = preprocess_meg(raw, target_fs=100, first_onset=onset)
    assert meg.dtype == np.float32 and np.isfinite(meg).all()
    audio = load_audio_mono_16k(PILOT / 'raw/stimuli/audio/lw1_0.wav')
    assert len(audio) >= 36 * 16000 and np.isfinite(audio).all()
    meg_windows, audio_windows, metadata = [], [], []
    for i in range(12):
        audio_start = i * 3.0
        # Match the upstream onset discretization and 150 ms response offset.
        begin = int(onset * 100) + i * 300 + 15
        end = begin + 300
        meg_windows.append(meg[:, begin:end])
        audio_windows.append(audio[i * 48000:(i + 1) * 48000])
        metadata.append({
            'window_id': i, 'subject_id': 1, 'model_subject_index': 0,
            'session_id': 0, 'story_id': 0, 'sound_id': 0,
            'audio_file': 'lw1_0.wav', 'audio_start_seconds': audio_start,
            'audio_end_seconds': audio_start + 3, 'stimulus_onset_meg_seconds': onset,
            'meg_start_sample_100hz': begin, 'meg_stop_sample_100hz': end,
            'meg_start_seconds': begin / 100, 'response_offset_seconds': 0.15,
            'onset_rounding_error_seconds': int(onset * 100) / 100 - onset,
        })
    meg_windows = np.stack(meg_windows)
    audio_windows = np.stack(audio_windows)
    assert meg_windows.shape == (12, 208, 300)
    assert audio_windows.shape == (12, 48000)
    print('Extracting real wav2vec2 audio targets...', flush=True)
    extractor, audio_model = load_wav2vec_model(str(PILOT / 'wav2vec2-base-960h'), 'cpu')
    embeddings = []
    for i, window in enumerate(audio_windows):
        embedding, _ = extract_wav2vec_embedding(
            window, extractor, audio_model, 'cpu', last_n=4, return_all_layers=False,
        )
        embeddings.append(embedding)
        print('Audio target', i + 1, '/ 12', flush=True)
    embeddings = np.stack(embeddings).astype(np.float32)
    assert embeddings.shape == (12, 149, 768) and np.isfinite(embeddings).all()
    np.save(OUT / 'meg_windows.npy', meg_windows)
    np.save(OUT / 'audio_windows_16k.npy', audio_windows)
    np.save(OUT / 'audio_embeddings.npy', embeddings)
    pd.DataFrame(metadata).to_csv(OUT / 'windows.csv', index=False)
    (OUT / 'channels.json').write_text(json.dumps(raw.ch_names, indent=2) + '\n')
    config = parse_experiment_arguments([
        '--run-group', 'fqw-real-data-pilot', '--device', 'cpu',
        '--dirprocess', str(ROOT / 'code/LISA/data/preprocessed'),
    ])
    model, _ = load_from_config(config, None, torch.device('cpu'))
    criterion = CLIPLoss(1.0)
    optimizer = torch.optim.AdamW([
        {'params': model.parameters(), 'lr': 3e-4},
        {'params': criterion.parameters(), 'lr': 1e-3},
    ], weight_decay=0.0)
    x = torch.from_numpy(meg_windows)
    y = torch.from_numpy(embeddings).transpose(1, 2)
    subject_ids = torch.zeros(12, dtype=torch.long)
    ids = torch.arange(12)
    losses = []
    model.train()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        _, output = model((x, subject_ids))
        _, loss = criterion(output, y, ids)
        assert torch.isfinite(loss)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    with torch.no_grad():
        _, output = model((x, subject_ids))
        metrics = retrieval_topk(output, y, ids, ids, top_k=10)
    results = {
        'status': 'passed', 'scope': 'Real-data pipeline smoke check; same 12 windows used for optimization and retrieval. Not a baseline or held-out evaluation.',
        'upstream_commit': subprocess.check_output(['git', '-C', str(ROOT / 'code/LISA'), 'rev-parse', 'HEAD'], text=True).strip(),
        'python': platform.python_version(), 'torch': torch.__version__,
        'device': 'cpu', 'threads': 4, 'seed': 42,
        'audio_model': 'facebook/wav2vec2-base-960h',
        'audio_model_revision': '22aad52d435eb6dbaf354bdad9b0da84ce7d6156',
        'raw_samples': int(original_samples), 'raw_duration_seconds': float(original_duration),
        'raw_sample_rate': sfreq, 'meg_channels': len(raw.ch_names),
        'ica_excluded_components': [int(component) for component in ica.exclude],
        'preprocessing': 'Author ICA on full task-0 recording; author full-recording normalization and downsample to 100 Hz. No held-out calibration budget claim.',
        'forced_alignment': 'Not run; stimulus onset obtained from original BIDS events. No word-level TextGrid or canonical full dataset generated.',
        'audio_seconds_selected': 36, 'windows': 12,
        'meg_window_shape': list(meg_windows.shape), 'audio_embedding_shape': list(embeddings.shape),
        'optimizer_steps': 3, 'training_losses': losses,
        'same_batch_top1_among_12': metrics['is_in_top1'].float().mean().item(),
        'same_batch_top10_among_12': metrics['is_in_topk'].float().mean().item(),
        'elapsed_seconds': time.perf_counter() - start,
        'prepared_array_bytes': sum(p.stat().st_size for p in OUT.glob('*.npy')),
    }
    (REPORT / 'real_data_pilot.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
