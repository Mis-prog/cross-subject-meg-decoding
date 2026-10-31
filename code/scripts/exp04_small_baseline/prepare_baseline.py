"""Prepare a fixed ~24-minute subset with fragment-level train/validation/test."""
import ast
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
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import mne
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from sklearn.preprocessing import RobustScaler, StandardScaler

from lisa.data.meg_io import load_raw_meg
from lisa.data.generate_all_embeddings import load_wav2vec_model, extract_wav2vec_embedding
from lisa.utils.audio_io import load_audio_mono_16k

ROOT = Path(__file__).resolve().parents[3]
PILOT = ROOT / 'data/pilot'
OUT = PILOT / 'baseline'
PROTOCOL = ROOT / 'experiments/exp04_small_baseline/protocol.json'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_windows(windows):
    if set(windows.split) != {'train', 'validation', 'test'}:
        raise ValueError('Missing split')
    if windows.groupby('audio_file').split.nunique().max() != 1:
        raise ValueError('Audio fragment spans multiple splits')
    if windows.window_id.tolist() != list(range(len(windows))):
        raise ValueError('Noncanonical window order')
    for _, rows in windows.groupby('task'):
        rows = rows.sort_values('meg_start_sample_1000hz')
        if np.any(rows.meg_start_sample_1000hz.values[1:] < rows.meg_stop_sample_1000hz.values[:-1]):
            raise ValueError('Overlapping raw MEG windows')
    if not ((windows.meg_stop_sample_1000hz - windows.meg_start_sample_1000hz) == 3000).all():
        raise ValueError('Wrong raw window duration')


def main():
    started = time.perf_counter()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    protocol = json.loads(PROTOCOL.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    sources = json.loads((PILOT / 'baseline_downloads.json').read_text())
    for item in sources:
        path = PILOT / item['path']
        if path.stat().st_size != item['size'] or sha256(path) != item['local_sha256']:
            raise ValueError(f'Source changed: {path}')
    print('Verified all', len(sources), 'source files', flush=True)
    fragments, rows = [], []
    for task in protocol['tasks']:
        event_path = PILOT / f'raw/sub-01/ses-0/meg/sub-01_ses-0_task-{task}_events.tsv'
        sound_events = []
        for event_row in pd.read_csv(event_path, sep='\t').itertuples():
            event = ast.literal_eval(event_row.trial_type)
            if event.get('kind') == 'sound':
                fname = Path(event['sound']).name.removesuffix('.0.wav') + '.wav'
                sound_events.append((float(event_row.onset), fname, int(event['sound_id'])))
        sound_events.sort()
        for i, (onset, fname, sound_id) in enumerate(sound_events):
            info = sf.info(PILOT / 'raw/stimuli/audio' / fname)
            duration = info.frames / info.samplerate
            raw_start = round(onset * 1000)
            raw_stop = raw_start + int(duration * 1000) + 150
            if i + 1 < len(sound_events) and raw_stop > round(sound_events[i + 1][0] * 1000):
                raise ValueError('Stimulus preprocessing contexts overlap')
            fragment = {
                'task': task, 'sound_id': sound_id, 'audio_file': fname,
                'split': protocol['splits'][fname], 'duration_seconds': duration,
                'onset_seconds': onset, 'raw_crop_start': raw_start,
                'raw_crop_stop': raw_stop,
            }
            fragments.append(fragment)
            margin = protocol['edge_margin_seconds']
            count = int((duration - 2 * margin) // protocol['window_seconds'])
            for index in range(count):
                audio_start = margin + index * 3
                meg_start = raw_start + audio_start * 1000 + 150
                rows.append({
                    'window_id': len(rows), 'subject': 1, 'model_subject_index': 0,
                    'session': 0, 'task': task, 'sound_id': sound_id,
                    'audio_file': fname, 'split': fragment['split'],
                    'audio_start_seconds': audio_start, 'audio_stop_seconds': audio_start + 3,
                    'meg_start_sample_1000hz': meg_start, 'meg_stop_sample_1000hz': meg_start + 3000,
                    'resampled_start_in_fragment': audio_start * 100 + 15,
                    'response_offset_seconds': 0.15,
                })
    if {f['audio_file'] for f in fragments} != set(protocol['splits']):
        raise ValueError('Protocol and available fragments disagree')
    windows = pd.DataFrame(rows)
    validate_windows(windows)
    windows.to_csv(OUT / 'windows.csv', index=False)
    (OUT / 'fragments.json').write_text(json.dumps(fragments, indent=2) + '\n')
    # Save the split before feature extraction or model training.
    (OUT / 'protocol.json').write_text(PROTOCOL.read_text())
    n = len(windows)
    meg = np.lib.format.open_memmap(OUT / 'meg_windows.npy', mode='w+', dtype='float32', shape=(n, 208, 300))
    audio_windows = np.lib.format.open_memmap(OUT / 'audio_windows_16k.npy', mode='w+', dtype='float32', shape=(n, 48000))
    embeddings = np.lib.format.open_memmap(OUT / 'audio_embeddings.npy', mode='w+', dtype='float32', shape=(n, 149, 768))
    channels, ica_metadata = None, []
    for task in protocol['tasks']:
        print('Loading and applying ICA to task', task, flush=True)
        raw, sr = load_raw_meg('bids', str(PILOT / 'raw'), 1, 0, task)
        if sr != 1000 or len(raw.ch_names) != 208:
            raise ValueError('Unexpected MEG layout')
        if channels is None:
            channels = raw.ch_names.copy()
        elif channels != raw.ch_names:
            raise ValueError('Channel ordering differs between recordings')
        ica = mne.preprocessing.read_ica(PILOT / f'ica/sub-01/ses-0/meg/sub-01_ses-0_task-{task}_meg-ica.fif', verbose=False)
        if ica.ch_names != channels:
            raise ValueError('ICA channel mismatch')
        ica_metadata.append({'task': task, 'excluded_components': [int(c) for c in ica.exclude], 'raw_duration_seconds': raw.n_times / sr})
        ica.apply(raw, verbose=False)
        for fragment in [f for f in fragments if f['task'] == task]:
            if fragment['raw_crop_stop'] > raw.n_times:
                raise ValueError('Fragment exceeds recording')
            # Resample each stimulus independently: FFT context stays within its split.
            data = raw.get_data(start=fragment['raw_crop_start'], stop=fragment['raw_crop_stop'])
            downsampled = mne.filter.resample(data, down=10, npad='auto', verbose=False)
            for row in windows[windows.audio_file == fragment['audio_file']].itertuples():
                start = row.resampled_start_in_fragment
                window = downsampled[:, start:start + 300]
                if window.shape != (208, 300) or not np.isfinite(window).all():
                    raise ValueError('Invalid MEG window')
                meg[row.window_id] = window
            del data, downsampled
        del raw, ica
    # Statistics see only actual training windows, never validation/test or gaps.
    train_ids = windows.loc[windows.split == 'train', 'window_id'].to_numpy()
    fit_data = np.asarray(meg[train_ids]).transpose(0, 2, 1).reshape(-1, 208).astype(np.float64)
    robust = RobustScaler(quantile_range=(25, 75)).fit(fit_data)
    robust_train = robust.transform(fit_data)
    standard = StandardScaler().fit(robust_train)
    for i in range(n):
        scaled = standard.transform(robust.transform(meg[i].T.astype(np.float64)))
        meg[i] = np.clip(scaled.T, -20, 20).astype(np.float32)
    scalers = {
        'fit_window_ids': train_ids.tolist(), 'fit_sample_count': len(fit_data),
        'robust_center': robust.center_.tolist(), 'robust_scale': robust.scale_.tolist(),
        'standard_mean': standard.mean_.tolist(), 'standard_scale': standard.scale_.tolist(),
    }
    (OUT / 'scalers.json').write_text(json.dumps(scalers, indent=2) + '\n')
    del fit_data, robust_train
    extractor, audio_model = load_wav2vec_model(str(PILOT / 'wav2vec2-base-960h'), 'cpu')
    for fragment in fragments:
        audio = load_audio_mono_16k(PILOT / 'raw/stimuli/audio' / fragment['audio_file'])
        for row in windows[windows.audio_file == fragment['audio_file']].itertuples():
            clip = audio[row.audio_start_seconds * 16000:row.audio_stop_seconds * 16000]
            if clip.shape != (48000,) or not np.isfinite(clip).all():
                raise ValueError('Invalid audio window')
            embedding, _ = extract_wav2vec_embedding(clip, extractor, audio_model, 'cpu', last_n=4, return_all_layers=False)
            if embedding.shape != (149, 768) or not np.isfinite(embedding).all():
                raise ValueError('Invalid audio embedding')
            audio_windows[row.window_id] = clip
            embeddings[row.window_id] = embedding
        print('Prepared', fragment['audio_file'], fragment['split'], flush=True)
    for array in (meg, audio_windows, embeddings):
        array.flush()
        if not np.isfinite(array).all():
            raise ValueError('Non-finite prepared data')
    (OUT / 'channels.json').write_text(json.dumps(channels, indent=2) + '\n')
    result = {
        'status': 'passed', 'protocol_sha256': sha256(PROTOCOL),
        'upstream_commit': subprocess.check_output(['git', '-C', str(ROOT / 'code/LISA'), 'rev-parse', 'HEAD'], text=True).strip(),
        'audio_model': 'facebook/wav2vec2-base-960h',
        'audio_model_revision': '22aad52d435eb6dbaf354bdad9b0da84ce7d6156',
        'source_manifest_sha256': sha256(PILOT / 'baseline_downloads.json'),
        'source_files': len(sources), 'source_bytes': sum(i['size'] for i in sources),
        'windows': n, 'selected_audio_seconds': n * 3,
        'source_audio_seconds': sum(f['duration_seconds'] for f in fragments),
        'split_counts': {str(k): int(v) for k, v in windows.groupby('split').size().items()},
        'split_fragments': {str(k): int(v) for k, v in windows.groupby('split').audio_file.nunique().items()},
        'preprocessing': protocol['preprocessing'], 'ica': ica_metadata,
        'meg_shape': list(meg.shape), 'audio_shape': list(audio_windows.shape), 'embedding_shape': list(embeddings.shape),
        'prepared_sha256': {p.name: sha256(p) for p in sorted(OUT.iterdir()) if p.is_file()},
        'prepared_bytes': sum(p.stat().st_size for p in OUT.iterdir() if p.is_file()),
        'python': platform.python_version(), 'torch': torch.__version__, 'device': 'cpu',
        'elapsed_seconds': time.perf_counter() - started,
        'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == 'Darwin' else 1024),
    }
    (ROOT / 'experiments/exp04_small_baseline/results/baseline_preparation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('prepared_sha256',)}, indent=2))


if __name__ == '__main__':
    main()
