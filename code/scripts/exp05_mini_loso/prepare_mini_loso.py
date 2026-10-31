"""Prepare four-subject windows (same fragments and splits as exp04) for the mini-LOSO pilot."""
import ast
import json
import os
import platform
import resource
import subprocess
import time
import sys
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
# Support direct execution from any working directory.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.exp04_small_baseline.prepare_baseline import sha256

ROOT = Path(__file__).resolve().parents[3]
PILOT = ROOT / 'data/pilot'
OUT = PILOT / 'mini_loso'
PROTOCOL = ROOT / 'experiments/exp05_mini_loso/protocol.json'
RESULTS = ROOT / 'experiments/exp05_mini_loso/results'


def sound_events(subject, task):
    path = PILOT / f'raw/sub-{subject:02d}/ses-0/meg/sub-{subject:02d}_ses-0_task-{task}_events.tsv'
    events = []
    for row in pd.read_csv(path, sep='\t').itertuples():
        event = ast.literal_eval(row.trial_type)
        if event.get('kind') == 'sound':
            fname = Path(event['sound']).name.removesuffix('.0.wav') + '.wav'
            events.append((float(row.onset), fname, int(event['sound_id'])))
    return sorted(events)


def build_tables(protocol):
    """Audio windows are shared across subjects; MEG windows are per subject."""
    margin, length = protocol['edge_margin_seconds'], protocol['window_seconds']
    durations = {}
    for fname in protocol['splits']:
        info = sf.info(PILOT / 'raw/stimuli/audio' / fname)
        durations[fname] = info.frames / info.samplerate
    audio_rows = []
    for fname in protocol['splits']:
        for index in range(int((durations[fname] - 2 * margin) // length)):
            start = margin + index * length
            audio_rows.append({'audio_id': len(audio_rows), 'audio_file': fname,
                               'split': protocol['splits'][fname],
                               'audio_start_seconds': start, 'audio_stop_seconds': start + length})
    audio = pd.DataFrame(audio_rows)
    fragments, meg_rows = [], []
    for model_index, subject in enumerate(protocol['subjects']):
        for task in protocol['tasks']:
            events = sound_events(subject, task)
            for i, (onset, fname, sound_id) in enumerate(events):
                raw_start = round(onset * 1000)
                raw_stop = raw_start + int(durations[fname] * 1000) + 150
                if i + 1 < len(events) and raw_stop > round(events[i + 1][0] * 1000):
                    raise ValueError(f'sub-{subject:02d}: stimulus contexts overlap')
                fragments.append({'subject': subject, 'task': task, 'sound_id': sound_id,
                                  'audio_file': fname, 'split': protocol['splits'][fname],
                                  'onset_seconds': onset, 'raw_crop_start': raw_start,
                                  'raw_crop_stop': raw_stop})
                for a in audio[audio.audio_file == fname].itertuples():
                    meg_start = raw_start + a.audio_start_seconds * 1000 + 150
                    meg_rows.append({
                        'window_id': len(meg_rows), 'subject': subject,
                        'model_subject_index': model_index,
                        'role': 'holdout' if subject == protocol['holdout_subject'] else 'source',
                        'session': 0, 'task': task, 'sound_id': sound_id,
                        'audio_id': a.audio_id, 'audio_file': fname, 'split': a.split,
                        'audio_start_seconds': a.audio_start_seconds,
                        'meg_start_sample_1000hz': meg_start,
                        'meg_stop_sample_1000hz': meg_start + length * 1000,
                        'resampled_start_in_fragment': a.audio_start_seconds * 100 + 15,
                    })
        seen = {f['audio_file'] for f in fragments if f['subject'] == subject}
        if seen != set(protocol['splits']):
            raise ValueError(f'sub-{subject:02d}: fragments disagree with protocol: {sorted(seen ^ set(protocol["splits"]))}')
    return audio, pd.DataFrame(meg_rows), fragments


def validate(audio, windows, protocol):
    if audio.groupby('audio_file').split.nunique().max() != 1:
        raise ValueError('Audio fragment spans multiple splits')
    if windows.window_id.tolist() != list(range(len(windows))):
        raise ValueError('Noncanonical window order')
    if windows.duplicated(['subject', 'audio_id']).any():
        raise ValueError('Duplicate subject/audio window')
    for _, rows in windows.groupby(['subject', 'task']):
        rows = rows.sort_values('meg_start_sample_1000hz')
        if np.any(rows.meg_start_sample_1000hz.values[1:] < rows.meg_stop_sample_1000hz.values[:-1]):
            raise ValueError('Overlapping raw MEG windows')
    per_subject = windows.groupby('subject').size()
    if per_subject.nunique() != 1 or per_subject.iloc[0] != len(audio):
        raise ValueError('Subjects do not share the same audio windows')
    if set(windows.subject) != set(protocol['subjects']):
        raise ValueError('Subject set disagrees with protocol')


def main():
    started = time.perf_counter()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    protocol = json.loads(PROTOCOL.read_text())
    sources = [item for name in ('baseline_downloads.json', 'mini_loso_downloads.json')
               for item in json.loads((PILOT / name).read_text())]
    for item in sources:
        path = PILOT / item['path']
        if path.stat().st_size != item['size'] or sha256(path) != item['local_sha256']:
            raise ValueError(f'Source changed: {path}')
    print('Verified', len(sources), 'source files', flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    audio, windows, fragments = build_tables(protocol)
    validate(audio, windows, protocol)
    audio.to_csv(OUT / 'audio_windows.csv', index=False)
    windows.to_csv(OUT / 'meg_windows.csv', index=False)
    (OUT / 'fragments.json').write_text(json.dumps(fragments, indent=2) + '\n')
    (OUT / 'protocol.json').write_text(PROTOCOL.read_text())
    print('Windows:', len(windows), 'MEG /', len(audio), 'audio', flush=True)

    n = len(windows)
    meg = np.lib.format.open_memmap(OUT / 'meg_windows.npy', mode='w+', dtype='float32', shape=(n, 208, 300))
    channels, ica_metadata, scalers = None, [], {}
    for subject in protocol['subjects']:
        sub_rows = windows[windows.subject == subject]
        for task in protocol['tasks']:
            print(f'sub-{subject:02d} task-{task}: loading, ICA', flush=True)
            raw, sr = load_raw_meg('bids', str(PILOT / 'raw'), subject, 0, task)
            if sr != 1000 or len(raw.ch_names) != 208:
                raise ValueError('Unexpected MEG layout')
            if channels is None:
                channels = raw.ch_names.copy()
            elif channels != raw.ch_names:
                raise ValueError(f'sub-{subject:02d}: channel ordering differs')
            ica = mne.preprocessing.read_ica(
                PILOT / f'ica/sub-{subject:02d}/ses-0/meg/sub-{subject:02d}_ses-0_task-{task}_meg-ica.fif', verbose=False)
            if ica.ch_names != channels:
                raise ValueError(f'sub-{subject:02d}: ICA channel mismatch')
            ica_metadata.append({'subject': subject, 'task': task,
                                 'excluded_components': [int(c) for c in ica.exclude],
                                 'raw_duration_seconds': raw.n_times / sr})
            ica.apply(raw, verbose=False)
            for fragment in [f for f in fragments if f['subject'] == subject and f['task'] == task]:
                if fragment['raw_crop_stop'] > raw.n_times:
                    raise ValueError('Fragment exceeds recording')
                data = raw.get_data(start=fragment['raw_crop_start'], stop=fragment['raw_crop_stop'])
                downsampled = mne.filter.resample(data, down=10, npad='auto', verbose=False)
                for row in sub_rows[sub_rows.audio_file == fragment['audio_file']].itertuples():
                    window = downsampled[:, row.resampled_start_in_fragment:row.resampled_start_in_fragment + 300]
                    if window.shape != (208, 300) or not np.isfinite(window).all():
                        raise ValueError('Invalid MEG window')
                    meg[row.window_id] = window
                del data, downsampled
            del raw, ica
        # Per-subject scaling, fitted on that subject's train-split windows only.
        train_ids = sub_rows.loc[sub_rows.split == 'train', 'window_id'].to_numpy()
        fit_data = np.asarray(meg[train_ids]).transpose(0, 2, 1).reshape(-1, 208).astype(np.float64)
        robust = RobustScaler(quantile_range=(25, 75)).fit(fit_data)
        standard = StandardScaler().fit(robust.transform(fit_data))
        for i in sub_rows.window_id:
            scaled = standard.transform(robust.transform(meg[i].T.astype(np.float64)))
            meg[i] = np.clip(scaled.T, -20, 20).astype(np.float32)
        scalers[f'sub-{subject:02d}'] = {
            'fit_window_ids': train_ids.tolist(), 'fit_sample_count': len(fit_data),
            'robust_center': robust.center_.tolist(), 'robust_scale': robust.scale_.tolist(),
            'standard_mean': standard.mean_.tolist(), 'standard_scale': standard.scale_.tolist(),
        }
        del fit_data
    (OUT / 'scalers.json').write_text(json.dumps(scalers) + '\n')
    (OUT / 'channels.json').write_text(json.dumps(channels, indent=2) + '\n')

    audio_windows = np.lib.format.open_memmap(OUT / 'audio_windows_16k.npy', mode='w+', dtype='float32', shape=(len(audio), 48000))
    embeddings = np.lib.format.open_memmap(OUT / 'audio_embeddings.npy', mode='w+', dtype='float32', shape=(len(audio), 149, 768))
    extractor, audio_model = load_wav2vec_model(str(PILOT / 'wav2vec2-base-960h'), 'cpu')
    for fname, rows in audio.groupby('audio_file'):
        waveform = load_audio_mono_16k(PILOT / 'raw/stimuli/audio' / fname)
        for row in rows.itertuples():
            clip = waveform[row.audio_start_seconds * 16000:row.audio_stop_seconds * 16000]
            if clip.shape != (48000,) or not np.isfinite(clip).all():
                raise ValueError('Invalid audio window')
            embedding, _ = extract_wav2vec_embedding(clip, extractor, audio_model, 'cpu', last_n=4, return_all_layers=False)
            if embedding.shape != (149, 768) or not np.isfinite(embedding).all():
                raise ValueError('Invalid audio embedding')
            audio_windows[row.audio_id] = clip
            embeddings[row.audio_id] = embedding
        print('Embedded', fname, flush=True)
    for array in (meg, audio_windows, embeddings):
        array.flush()
        if not np.isfinite(array).all():
            raise ValueError('Non-finite prepared data')

    result = {
        'status': 'passed', 'protocol_sha256': sha256(PROTOCOL),
        'upstream_commit': subprocess.check_output(['git', '-C', str(ROOT / 'code/LISA'), 'rev-parse', 'HEAD'], text=True).strip(),
        'audio_model': 'facebook/wav2vec2-base-960h',
        'subjects': protocol['subjects'], 'holdout_subject': protocol['holdout_subject'],
        'meg_windows': n, 'audio_windows': len(audio),
        'split_counts_per_subject': {str(k): int(v) for k, v in audio.groupby('split').size().items()},
        'ica': ica_metadata, 'preprocessing': protocol['preprocessing'],
        'prepared_sha256': {p.name: sha256(p) for p in sorted(OUT.iterdir()) if p.is_file()},
        'prepared_bytes': sum(p.stat().st_size for p in OUT.iterdir() if p.is_file()),
        'python': platform.python_version(), 'torch': torch.__version__, 'device': 'cpu',
        'elapsed_seconds': time.perf_counter() - started,
        'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == 'Darwin' else 1024),
    }
    (RESULTS / 'mini_loso_preparation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'prepared_sha256'}, indent=2))


if __name__ == '__main__':
    main()
