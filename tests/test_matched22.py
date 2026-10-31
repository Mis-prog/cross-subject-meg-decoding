"""Leakage/cohort/compatibility guards for reusing B in the matched22 control."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from scripts.exp08_09_full_lisa.full_lisa import filter_rows, load_protocol, slot_conditions, train_tasks
from scripts.exp08_09_full_lisa import full_lisa
from scripts.exp08_matched22 import run_matched22 as matched
from scripts.exp08_matched22.run_matched22 import (PROTOCOL, SUBJECTS, rank_agreement, reference_run,
                           signature, validate_eval, validate_rows)


def rows():
    return pd.DataFrame([{'subject_id': s, 'session_id': ses, 'story_id': 3,
                          'sound_id': 5, 'wav_index': w}
                         for s in range(1, 28) for ses in ([0, 1] if s in SUBJECTS else [0])
                         for w in range(3)])


def test_both_training_filters_and_donors():
    protocol = load_protocol(PROTOCOL)
    original = train_tasks(load_protocol())
    tasks = train_tasks(protocol)
    assert len(tasks) == 6
    assert original['xsA_s42']['exclude'] == []  # original protocol unchanged
    frame = rows()
    A = np.arange(27 * 6).reshape(27, 2, 3)
    for direction in 'AB':
        task = f'xs{direction}_s42'
        assert set(filter_rows(frame, tasks[task]).subject_id) == SUBJECTS
        conditions = list(slot_conditions(task, protocol, A, sorted(SUBJECTS), frame))
        for condition, part, overrides in conditions:
            assert set(part.subject_id) <= SUBJECTS
            if condition == 'mean_cross':
                s = part.subject_id.iloc[0]
                np.testing.assert_allclose(overrides[s], A[[t-1 for t in sorted(SUBJECTS) if t != s]].mean(0))
            if condition == 'other_cross':
                assert set(part.slot) <= SUBJECTS
                assert (part.slot != part.subject_id).all()
        donors = [part for c, part, _ in conditions if c == 'other_cross']
        assert len(pd.concat(donors)) == 22 * 21 * 3


def test_matching_audio_identity_not_only_count():
    frame = rows()
    protocol = load_protocol(PROTOCOL)
    assert validate_rows(frame, frame, protocol)['test_A'] == 66
    wrong = frame.copy()
    wrong.loc[(wrong.subject_id == 1) & (wrong.session_id == 1) & (wrong.wav_index == 0), 'wav_index'] = 999
    with pytest.raises(ValueError, match='audio windows differ'):
        validate_rows(wrong, frame, protocol)
    with pytest.raises(ValueError, match='duplicate'):
        validate_rows(pd.concat([frame, frame.iloc[[0]]]), frame, protocol)


def test_reference_eval_requires_exact_donors_and_cohort():
    frame = rows()
    own = frame[(frame.session_id == 0) & frame.subject_id.isin(SUBJECTS)].copy()
    own['slot'] = own.subject_id
    own['rank'] = 0
    validate_eval(own, frame, 'A', 'own_same')
    with pytest.raises(ValueError, match='wrong subjects'):
        validate_eval(own.iloc[:-1], frame, 'A', 'own_same')
    donors = pd.concat([own[own.subject_id != s].assign(slot=s) for s in sorted(SUBJECTS)])
    validate_eval(donors, frame, 'B', 'other_cross')
    donors.iloc[0, donors.columns.get_loc('slot')] = 3
    with pytest.raises(ValueError, match='donors'):
        validate_eval(donors, frame, 'B', 'other_cross')


def test_data_compatibility_rejects_different_predictions():
    frame = rows().assign(slot=lambda d: d.subject_id, rank=0)
    assert rank_agreement(frame, frame)['identical'] == 1
    with pytest.raises(ValueError, match='do not reproduce'):
        rank_agreement(frame.assign(rank=100), frame)
    with pytest.raises(ValueError, match='different evaluated rows'):
        rank_agreement(frame.iloc[:-1], frame)


def test_reference_cannot_escape_and_configs_keep_training_settings(tmp_path):
    folder = tmp_path / 'eval/xsB_s42'
    folder.mkdir(parents=True)
    (folder / 'done.json').write_text(json.dumps({'run_dir': '../../unrelated'}))
    with pytest.raises(ValueError, match='escapes'):
        reference_run(tmp_path, 42)
    assert signature({'device': 'cuda', 'run_name': 'A', 'batch_size': 100}) == {'batch_size': 100}
    assert signature({'batch_size': 100}) != signature({'batch_size': 50})


def test_provenance_includes_shared_pipeline_sources(tmp_path, monkeypatch):
    """Moving the driver must not silently omit shared code from resume checks."""
    protocol = tmp_path / 'protocol.json'
    protocol.write_text('{}')
    monkeypatch.setattr(matched, 'ROOT', tmp_path)
    monkeypatch.setattr(matched, 'reference_run', lambda reference, seed: reference / str(seed))
    manifest = matched.input_manifest(tmp_path / 'canon', tmp_path / 'reference', protocol)
    pipeline = Path(full_lisa.__file__).parent
    sources = [Path(matched.__file__), pipeline / 'full_lisa.py',
               pipeline / 'run_full_series.py', pipeline / 'summarize_full.py']
    for source in sources:
        assert manifest[str(source.resolve())] == matched.digest(source)
