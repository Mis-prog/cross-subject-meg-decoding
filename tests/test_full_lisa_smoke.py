"""End-to-end run of the exp08/exp09 pipeline on a tiny synthetic copy of the authors' preprocessed data.

The real lisa-train (one CPU epoch), our frozen evaluation and the summary all execute; only the data
are synthetic. Checks the row filtering, slot overrides and the agreement with the trainer's ranks.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'code/scripts/exp08_09_full_lisa'
sys.path.insert(0, str(ROOT / 'code'))
from scripts.exp08_09_full_lisa import full_lisa  # noqa: E402

SESSIONS = {1: (0, 1), 2: (0, 1), 3: (0,), 4: (0, 1)}
TRAIN = [(0, 0, 14), (3, 4, 12)]   # (story, sound, windows); 3:4 is the validation sound
TEST = [(3, 5, 12)]
ONSETS = {(0, 0): 0.5, (3, 4): 0.5, (3, 5): 12.0}


def windows(spec, start_index=0):
    rows, idx = [], start_index
    for story, sound, n in spec:
        for k in range(n):
            rows.append({'story_id': story, 'sound_id': sound, 'sound_fname': f's{story}_{sound}.wav',
                         'sound_length': 30.0, 'segment_start': k, 'words': "['a']",
                         'wav_start': k * 16000, 'wav_stop': k * 16000 + 48000, 'wav_index': idx})
            idx += 1
    return pd.DataFrame(rows)


def make_canon(path):
    rng = np.random.default_rng(0)
    (path / 'dataframe').mkdir(parents=True)
    (path / 'audio').mkdir()
    (path / 'meg').mkdir()
    shutil.copytree(ROOT / 'code/LISA/data/preprocessed/coords', path / 'coords')
    chunks = {'train': windows(TRAIN), 'test': windows(TEST)}
    chunks['test'].to_csv(path / 'dataframe' / 'chunks_info_test.csv', index=False)
    meg = {}
    for split, chunk in chunks.items():
        np.save(path / 'audio' / f'extract_features_{split}4.npy',
                rng.standard_normal((len(chunk), 149, 768)).astype(np.float32))
        rows = []
        for subject, sessions in SESSIONS.items():
            for session in sessions:
                for r in chunk.itertuples(index=False):
                    onset = ONSETS[(r.story_id, r.sound_id)]
                    start = int(onset * 100) + round(r.wav_start / 16000 * 100)
                    rows.append({'subject_id': subject, 'session_id': session, 'onset_meg': onset,
                                 'meg100_start': start, 'meg100_stop': start + 300, **r._asdict()})
                    meg.setdefault(f'subject{subject:02d}_session{session}_story{r.story_id}',
                                   rng.standard_normal((208, 3000)).astype(np.float32))
        pd.DataFrame(rows).to_csv(path / 'dataframe' / f'df_{split}27.csv', index=False)
    np.savez(path / 'meg' / 'meg27_sr100.npz', **meg)


def mini_protocol(path):
    protocol = full_lisa.load_protocol()
    exps = protocol['experiments']
    exps['exp08_reproduction']['seeds'] = [42]
    exps['exp08_cross_session']['seeds'] = [42]
    exps['exp09_new_subjects']['folds'] = [[2]]
    protocol['data']['subjects_without_session_1'] = [3]
    path.write_text(json.dumps(protocol))
    return protocol


def test_filters_and_tasks():
    protocol = full_lisa.load_protocol()
    tasks = full_lisa.train_tasks(protocol)
    assert len(tasks) == 3 + 6 + 9
    df = pd.DataFrame({'subject_id': [1, 1, 2, 3], 'session_id': [0, 1, 0, 1]})
    assert full_lisa.filter_rows(df, tasks['xsA_s42']).session_id.tolist() == [0, 0]
    held = protocol['experiments']['exp09_new_subjects']['folds'][1]
    assert not set(full_lisa.filter_rows(df, tasks['ns_f1_s42']).subject_id) & set(held)
    folds = sum(protocol['experiments']['exp09_new_subjects']['folds'], [])
    assert sorted(folds) == list(range(1, 28))


@pytest.mark.skipif(not (ROOT / 'code/LISA/data/preprocessed/coords').exists(), reason='LISA checkout missing')
def test_pipeline_end_to_end(tmp_path):
    from scripts.exp08_09_full_lisa.summarize_full import summarize
    canon, out = tmp_path / 'canon', tmp_path / 'out'
    make_canon(canon)
    protocol = mini_protocol(tmp_path / 'protocol.json')
    assert not full_lisa.check(canon, protocol, strict=True)['problems'] == []  # counts differ from the real data
    assert full_lisa.check(canon, protocol, strict=False)['problems'] == []

    cmd = [sys.executable, 'run_full_series.py', '--procs', '2', '--ignore-gate', '--device', 'cpu',
           '--protocol', str(tmp_path / 'protocol.json'), '--canon', str(canon), '--out', str(out),
           '--nepoch', '1', '--batch-size', '8', '--test-batch-size', '4', '--dl-n-workers', '0', '--save-test-top-k', '10']
    env = {**os.environ, 'MLFLOW_TRACKING_URI': ''}
    result = subprocess.run(cmd, cwd=SCRIPTS, capture_output=True, text=True, env=env, timeout=1800)
    assert result.returncode == 0, result.stdout + result.stderr

    ev = out / 'eval'
    assert sorted(p.name for p in ev.iterdir()) == ['ns_f0_s42', 'repro_s42', 'xsA_s42', 'xsB_s42']
    for task in ('repro_s42', 'xsA_s42', 'xsB_s42'):
        agreement = json.loads((ev / task / 'done.json').read_text())['trainer_agreement']
        assert agreement['identical_ranks'] >= 0.99
    xs_a = {c: pd.read_csv(ev / 'xsA_s42' / f'{c}.csv.gz') for c in ('own_same', 'own_cross', 'mean_cross', 'other_cross')}
    assert set(xs_a['own_same'].session_id) == {0} and set(xs_a['own_cross'].session_id) == {1}
    assert set(xs_a['mean_cross'].subject_id) == {1, 2, 4}
    assert len(xs_a['other_cross']) == 3 * 3 * 12  # 3 targets x 3 donors x 12 windows
    assert (xs_a['other_cross'].slot != xs_a['other_cross'].subject_id).all()
    xs_b = pd.read_csv(ev / 'xsB_s42' / 'own_cross.csv.gz')
    assert set(xs_b.subject_id) == {1, 2, 4} and set(xs_b.session_id) == {0}  # subject 3 has no session 1
    ns = {c: pd.read_csv(ev / 'ns_f0_s42' / f'{c}.csv.gz') for c in ('mean', 'mean_norm', 'untrained')}
    assert all(set(d.subject_id) == {2} and len(d) == 2 * 12 for d in ns.values())
    run = pd.read_csv(full_lisa.find_run_dir('ns_f0_s42', out) / 'final_test_per_window.csv')
    assert 2 not in set(run.subject_id)

    # the sharded CPU evaluation must reproduce the single-process ranks exactly
    reference = {c: d.sort_values(['subject_id', 'session_id', 'slot', 'wav_index']).reset_index(drop=True)
                 for c, d in xs_a.items()}
    (ev / 'xsA_s42' / 'done.json').unlink()
    full_lisa.evaluate('xsA_s42', protocol, canon, out, device='cpu', shards=3)
    for c, d in reference.items():
        sharded = pd.read_csv(ev / 'xsA_s42' / f'{c}.csv.gz')
        assert sharded[['subject_id', 'slot', 'wav_index', 'rank']].equals(d[['subject_id', 'slot', 'wav_index', 'rank']])

    summary = summarize(json.loads((tmp_path / 'protocol.json').read_text()), out, canon, tmp_path / 'results')
    conditions = set(summary.condition)
    assert {'exp08A_own_cross', 'exp08B_own_same', 'exp08A_ceiling', 'exp09_mean', 'exp09_own_seen'} <= conditions
    comparisons = pd.read_csv(tmp_path / 'results' / 'comparisons.csv')
    assert 'exp08: drop(A) - drop(B)' in set(comparisons.comparison)
