"""Prepare windows for the 27-subject LOSO series (same fragments and splits as exp04/exp05).

Stores unscaled windows (for budget-limited scaling at calibration time) and windows scaled
with each subject's own train-split scaler. Subjects are processed in parallel.
"""
import argparse
import concurrent.futures
import json
import os
import platform
import resource
import subprocess
import time

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/fqw-mpl')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('OMP_NUM_THREADS', '4')

import mne
import numpy as np
import torch
from sklearn.preprocessing import RobustScaler, StandardScaler

from lisa.data.generate_all_embeddings import extract_wav2vec_embedding, load_wav2vec_model
from lisa.data.meg_io import load_raw_meg
from lisa.utils.audio_io import load_audio_mono_16k
import sys
from pathlib import Path

# Support direct execution from any working directory.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.exp04_small_baseline.prepare_baseline import sha256
from scripts.exp05_mini_loso.prepare_mini_loso import PILOT, ROOT, build_tables, validate

OUT = PILOT / 'loso'
PROTOCOL = ROOT / 'experiments/exp06_loso/protocol.json'
RESULTS = ROOT / 'experiments/exp06_loso/results'
SHAPE = (208, 300)


def fit_scaler(windows):
    """Robust then standard scaling per channel, as in exp04/exp05; returns plain parameters."""
    data = np.asarray(windows).transpose(0, 2, 1).reshape(-1, SHAPE[0]).astype(np.float64)
    robust = RobustScaler(quantile_range=(25, 75)).fit(data)
    standard = StandardScaler().fit(robust.transform(data))
    return {'robust_center': robust.center_.tolist(), 'robust_scale': robust.scale_.tolist(),
            'standard_mean': standard.mean_.tolist(), 'standard_scale': standard.scale_.tolist()}


def apply_scaler(windows, params):
    """Same transform as sklearn's pair, written as one per-channel affine map; clipped at 20 SD."""
    rc, rs = np.asarray(params['robust_center']), np.asarray(params['robust_scale'])
    sm, ss = np.asarray(params['standard_mean']), np.asarray(params['standard_scale'])
    offset, scale = rc + sm * rs, rs * ss
    scaled = (np.asarray(windows, dtype=np.float64) - offset[None, :, None]) / scale[None, :, None]
    return np.clip(scaled, -20, 20).astype(np.float32)


def process_subject(subject, rows, fragments, channels):
    """ICA, crop, resample and window one subject; writes rows into the shared memmaps."""
    unscaled = np.load(OUT / 'meg_unscaled.npy', mmap_mode='r+')
    scaled = np.load(OUT / 'meg_windows.npy', mmap_mode='r+')
    ica_metadata = []
    for task in sorted({f['task'] for f in fragments}):
        raw, sr = load_raw_meg('bids', str(PILOT / 'raw'), subject, 0, task)
        if sr != 1000 or raw.ch_names != channels:
            raise ValueError(f'sub-{subject:02d} task-{task}: unexpected MEG layout')
        ica = mne.preprocessing.read_ica(
            PILOT / f'ica/sub-{subject:02d}/ses-0/meg/sub-{subject:02d}_ses-0_task-{task}_meg-ica.fif', verbose=False)
        if ica.ch_names != channels:
            raise ValueError(f'sub-{subject:02d}: ICA channel mismatch')
        ica_metadata.append({'subject': subject, 'task': task,
                             'excluded_components': [int(c) for c in ica.exclude],
                             'raw_duration_seconds': raw.n_times / sr})
        ica.apply(raw, verbose=False)
        for fragment in [f for f in fragments if f['task'] == task]:
            if fragment['raw_crop_stop'] > raw.n_times:
                raise ValueError(f'sub-{subject:02d}: fragment exceeds recording')
            data = raw.get_data(start=fragment['raw_crop_start'], stop=fragment['raw_crop_stop'])
            downsampled = mne.filter.resample(data, down=10, npad='auto', verbose=False)
            for row in rows[rows.audio_file == fragment['audio_file']].itertuples():
                window = downsampled[:, row.resampled_start_in_fragment:row.resampled_start_in_fragment + SHAPE[1]]
                if window.shape != SHAPE or not np.isfinite(window).all():
                    raise ValueError(f'sub-{subject:02d}: invalid MEG window')
                unscaled[row.window_id] = window
        del raw, ica
    ids = rows.window_id.to_numpy()
    params = fit_scaler(unscaled[rows.loc[rows.split == 'train', 'window_id'].to_numpy()])
    scaled[ids] = apply_scaler(unscaled[ids], params)
    unscaled.flush()
    scaled.flush()
    return subject, params, ica_metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    started = time.perf_counter()
    protocol = json.loads(PROTOCOL.read_text())
    manifest = json.loads((PILOT / 'loso_downloads.json').read_text())
    missing = {f'sub-{s:02d}' for s in protocol['subjects']} - set(manifest['subjects'])
    if missing:
        raise ValueError(f'Not downloaded: {sorted(missing)}')
    for item in manifest['files']:
        path = PILOT / item['path']
        if path.stat().st_size != item['size']:
            raise ValueError(f'Source changed: {path}')
    OUT.mkdir(parents=True, exist_ok=True)
    audio, windows, fragments = build_tables(dict(protocol, holdout_subject=None))
    windows = windows.drop(columns='role')
    validate(audio, windows, protocol)
    audio.to_csv(OUT / 'audio_windows.csv', index=False)
    windows.to_csv(OUT / 'meg_windows.csv', index=False)
    (OUT / 'fragments.json').write_text(json.dumps(fragments, indent=2) + '\n')
    (OUT / 'protocol.json').write_text(PROTOCOL.read_text())
    print('Windows:', len(windows), 'MEG /', len(audio), 'audio', flush=True)

    for name in ('meg_unscaled.npy', 'meg_windows.npy'):
        np.lib.format.open_memmap(OUT / name, mode='w+', dtype='float32', shape=(len(windows), *SHAPE)).flush()
    raw, _ = load_raw_meg('bids', str(PILOT / 'raw'), protocol['subjects'][0], 0, protocol['tasks'][0])
    channels = raw.ch_names.copy()
    del raw
    scalers, ica_metadata = {}, []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(process_subject, s, windows[windows.subject == s],
                            [f for f in fragments if f['subject'] == s], channels)
                for s in protocol['subjects']]
        for job in concurrent.futures.as_completed(jobs):
            subject, params, meta = job.result()
            scalers[f'sub-{subject:02d}'] = params
            ica_metadata += meta
            print(f'sub-{subject:02d} done, {time.perf_counter() - started:.0f} s', flush=True)
    (OUT / 'scalers.json').write_text(json.dumps(dict(sorted(scalers.items()))) + '\n')
    (OUT / 'channels.json').write_text(json.dumps(channels, indent=2) + '\n')

    embeddings = np.lib.format.open_memmap(OUT / 'audio_embeddings.npy', mode='w+', dtype='float32',
                                           shape=(len(audio), 149, 768))
    extractor, audio_model = load_wav2vec_model(str(PILOT / 'wav2vec2-base-960h'), 'cpu')
    for fname, rows in audio.groupby('audio_file'):
        waveform = load_audio_mono_16k(PILOT / 'raw/stimuli/audio' / fname)
        for row in rows.itertuples():
            clip = waveform[row.audio_start_seconds * 16000:row.audio_stop_seconds * 16000]
            embedding, _ = extract_wav2vec_embedding(clip, extractor, audio_model, 'cpu', last_n=4, return_all_layers=False)
            if clip.shape != (48000,) or embedding.shape != (149, 768) or not np.isfinite(embedding).all():
                raise ValueError('Invalid audio window or embedding')
            embeddings[row.audio_id] = embedding
    embeddings.flush()
    for name in ('meg_unscaled.npy', 'meg_windows.npy'):
        if not np.isfinite(np.load(OUT / name, mmap_mode='r')).all():
            raise ValueError(f'Non-finite values in {name}')
    # exp05 used the same fragments and scaling; its four subjects must match exactly.
    reference = PILOT / 'mini_loso'
    agreement = {}
    if reference.exists():
        ref_emb = np.load(reference / 'audio_embeddings.npy')
        agreement['audio_embeddings_max_abs_diff'] = float(np.abs(ref_emb - embeddings).max())
        ref_x = np.load(reference / 'meg_windows.npy', mmap_mode='r')
        n = ref_x.shape[0]
        agreement['exp05_meg_max_abs_diff'] = float(np.abs(ref_x[:] - np.load(OUT / 'meg_windows.npy', mmap_mode='r')[:n]).max())
        print('Agreement with exp05:', agreement, flush=True)

    result = {
        'status': 'passed', 'protocol_sha256': sha256(PROTOCOL),
        'upstream_commit': subprocess.check_output(['git', '-C', str(ROOT / 'code/LISA'), 'rev-parse', 'HEAD'], text=True).strip(),
        'audio_model': 'facebook/wav2vec2-base-960h',
        'subjects': protocol['subjects'], 'meg_windows': len(windows), 'audio_windows': len(audio),
        'split_counts_per_subject': {str(k): int(v) for k, v in audio.groupby('split').size().items()},
        'ica': sorted(ica_metadata, key=lambda m: (m['subject'], m['task'])),
        'agreement_with_exp05': agreement,
        'prepared_sha256': {p.name: sha256(p) for p in sorted(OUT.iterdir()) if p.is_file()},
        'prepared_bytes': sum(p.stat().st_size for p in OUT.iterdir() if p.is_file()),
        'python': platform.python_version(), 'torch': torch.__version__, 'workers': args.workers,
        'elapsed_seconds': time.perf_counter() - started,
        'peak_rss_bytes_main': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        'peak_rss_bytes_worker': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024,
    }
    (RESULTS / 'loso_preparation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('prepared_sha256', 'ica')}, indent=2))


if __name__ == '__main__':
    main()
