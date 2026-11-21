"""Scientific invariants and a synthetic end-to-end exp10 run (no MEG downloads)."""
import copy
import json
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from scripts.exp10_shared_matrix import run_experiment as exp10
from scripts.exp10_shared_matrix.recordings import RecordingStore
from scripts.exp10_shared_matrix.subject_layer import SharedSubjectLayer


@pytest.fixture
def protocol():
    return exp10.read_json(exp10.PROTOCOL)


def test_folds_are_disjoint_and_cover_every_test_subject_once(protocol):
    specs = list(exp10.tasks(protocol).values())
    assert len(specs) == 27
    for spec in specs:
        train, val, test = [set(spec[k]) for k in ['train_subjects', 'val_subjects', 'test_subjects']]
        assert len(train) == 21 and len(val) == len(test) == 3
        assert not train & val and not train & test and not val & test
        assert train | val | test == set(range(1, 28))
    for variant in protocol['variants']:
        assert sorted(s for spec in specs if spec['variant'] == variant for s in spec['test_subjects']) == list(range(1, 28))
    broken = copy.deepcopy(protocol)
    broken['split']['folds'][1][0] = broken['split']['folds'][0][0]
    with pytest.raises(ValueError, match='partition'):
        exp10.tasks(broken)


def test_split_excludes_held_out_people_and_audio(protocol):
    spec = next(iter(exp10.tasks(protocol).values()))
    frame = pd.DataFrame([dict(subject_id=s, story_id=story, sound_id=sound, wav_index=w)
                          for s in range(1, 28) for story, sound, w in [(0, 0, 0), (3, 4, 1)]])
    train, val = exp10.split_rows(frame, spec, [3, 4])
    assert set(train.subject_id) == set(spec['train_subjects'])
    assert set(val.subject_id) == set(spec['val_subjects'])
    assert set(train.wav_index) == {0} and set(val.wav_index) == {1}


@pytest.mark.parametrize('p', [0.0, 1.0])
def test_endpoint_routes_and_gradients(p):
    initial = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4)
    layer = SharedSubjectLayer(initial, [0, 2], p)
    x = torch.randn(3, 4, 5)
    subjects = torch.tensor([0, 2, 0])
    actual = layer(x, subjects)
    expected = torch.bmm(initial[subjects], x) if p == 1 else layer.shared @ x
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    if p == 1:
        assert layer.shared.grad is None
        assert torch.count_nonzero(layer.A.grad[1]) == 0
        assert torch.count_nonzero(layer.A.grad[[0, 2]]) > 0
    else:
        assert layer.A.grad is None
        assert torch.count_nonzero(layer.shared.grad) > 0
    assert layer.route_counts.tolist() == ([3, 0] if p == 1 else [0, 3])


def test_independent_routing_and_deterministic_unseen_evaluation():
    layer = SharedSubjectLayer(torch.ones(3, 2, 4), [0, 2], 0.5)
    with torch.no_grad():
        layer.shared.fill_(2)
    x = torch.ones(1000, 4, 3)
    subjects = torch.zeros(1000, dtype=torch.long)
    torch.manual_seed(42)
    first = layer(x, subjects)
    torch.manual_seed(42)
    second = layer(x, subjects)
    torch.testing.assert_close(first, second)
    values = first[:, 0, 0]
    assert set(values.tolist()) == {4.0, 8.0}  # per-example routing, not one route for the batch
    assert 400 < int((values == 4).sum()) < 600
    first.sum().backward()
    personal_count = int((values == 4).sum())
    torch.testing.assert_close(layer.A.grad[0], torch.full((2, 4), personal_count * 3.0))
    torch.testing.assert_close(layer.shared.grad, torch.full((2, 4), (1000 - personal_count) * 3.0))
    assert torch.count_nonzero(layer.A.grad[1:]) == 0
    with pytest.raises(ValueError, match='held-out'):
        layer(torch.ones(1, 4, 3), torch.tensor([1]))
    layer.eval()
    held_out = torch.ones(2, 4, 3)
    expected = layer.shared @ held_out
    torch.testing.assert_close(layer(held_out, torch.tensor([1, 1])), expected)
    torch.testing.assert_close(layer(held_out, torch.tensor([0, 2])), expected)
    clone = SharedSubjectLayer(torch.zeros(3, 2, 4), [0, 2], 0.5)
    clone.load_state_dict(layer.state_dict())
    clone.eval()
    torch.testing.assert_close(clone(held_out, torch.tensor([1, 1])), expected)


def test_mean_never_uses_untrained_slots():
    initial = torch.tensor([[[2.]], [[10000.]], [[4.]]])
    layer = SharedSubjectLayer(initial, [0, 2], 1, mean_at_eval=True).eval()
    torch.testing.assert_close(layer(torch.ones(1, 1, 2), torch.tensor([1])), torch.full((1, 1, 2), 3.))


@pytest.mark.parametrize('p', [-1, 2, float('nan')])
def test_invalid_probability(p):
    with pytest.raises(ValueError, match='p_personal'):
        SharedSubjectLayer(torch.ones(3, 2, 4), [0], p)


def test_recording_store_spawn_and_reopen(tmp_path):
    path = tmp_path / 'meg.npz'
    np.savez(path, first=np.arange(12).reshape(3, 4))
    store = RecordingStore(path)
    np.testing.assert_array_equal(store['first'], np.arange(12).reshape(3, 4))
    clone = pickle.loads(pickle.dumps(store))
    assert clone._archive is None and not clone._cache
    np.testing.assert_array_equal(clone['first'], store['first'])
    store.close()
    clone.close()


def test_probability_selection_uses_only_validation(tmp_path, protocol):
    protocol['routing']['p_personal'] = [0.5, 0.9]
    for name, spec in exp10.tasks(protocol).items():
        if spec['fold'] != 0:
            continue
        run = tmp_path / 'runs' / name
        run.mkdir(parents=True)
        exp10.write_json(run / 'done.json', dict(val_top10=0.5, val_top1=0.1))
    names = exp10.select_tasks(protocol, tmp_path, {0})
    assert set(names) == {'individual_f0_s42', 'shared_only_f0_s42', 'switch_p0p5_f0_s42'}
    # No test directory exists; exact ties choose the smaller p specified by the protocol.
    assert not (tmp_path / 'eval').exists()


def test_rank_rows_supports_global_validation_ids():
    class IdentityModel(torch.nn.Module):
        def forward(self, xs):
            return xs[0], xs[0]

    hidden = np.eye(12, dtype=np.float32)[:, None, :]
    recordings = {'subject01_session0_story3': hidden[5].T}
    rows = pd.DataFrame([dict(subject_id=1, session_id=0, story_id=3, sound_id=4,
                              wav_index=5, meg100_start=0, meg100_stop=1)])
    ranks = exp10.rank_rows(IdentityModel(), rows, hidden, np.arange(2, 12), recordings,
                            dict(meg_sr=100, meg_offset=0), 'cpu', 1)
    assert ranks['rank'].tolist() == [0]
    assert ranks.top1_wav_index.tolist() == [5]


@pytest.mark.skipif(not (ROOT / 'code/LISA/data/preprocessed/coords').exists(), reason='LISA assets missing')
def test_synthetic_server_pipeline(tmp_path, protocol):
    from test_full_lisa_smoke import make_canon

    canon, out = tmp_path / 'canon', tmp_path / 'out'
    make_canon(canon)
    protocol['data'].update(subjects=[1, 2, 3, 4], validation_candidates=12, test_candidates=12)
    protocol['split']['folds'] = [[2], [3], [1, 4]]
    protocol['training']['lisa_args'] += ['--n-channels-attention', '8', '--n-spatial-harmonics', '2',
                                        '--n-channels-unmix', '4', '--n-channels-block', '4',
                                        '--n-temporal-module-blocks', '1']
    path = tmp_path / 'protocol.json'
    path.write_text(json.dumps(protocol))
    runner = ROOT / 'code/scripts/exp10_shared_matrix/run_experiment.py'
    common = ['--protocol', str(path), '--canon', str(canon), '--out', str(out), '--device', 'cpu',
              '--threads', '1', '--nepoch', '1', '--batch-size', '8', '--eval-batch-size', '8']
    upstream_hash = exp10.sha256(ROOT / 'code/LISA/src/lisa/model/nn_modules.py')
    for command, options in [('train', ['--all', '--fold', '0']), ('eval', ['--all', '--fold', '0']),
                              ('summarize', []), ('train', ['--all', '--fold', '0'])]:
        result = subprocess.run([sys.executable, str(runner), command, *common, *options],
                                cwd=tmp_path, capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, result.stdout + result.stderr
    assert exp10.sha256(ROOT / 'code/LISA/src/lisa/model/nn_modules.py') == upstream_hash
    for name, spec in exp10.tasks(protocol).items():
        if spec['fold'] != 0:
            continue
        run = out / 'runs' / name
        validation = pd.read_csv(run / 'validation.csv.gz')
        test = pd.read_csv(out / 'eval' / name / 'windows.csv.gz')
        assert set(validation.subject_id) == {3}
        assert set(test.subject_id) == {2}
        assert len(test) == 24 and test['rank'].between(0, 11).all()
        assert set(test.wav_index) == set(range(12))
        checkpoint = torch.load(run / 'best.pt', weights_only=False)
        original = checkpoint['model_state_dict']['spatial_module.subject_layer.A'][1].clone()
        from scripts.exp10_shared_matrix.model import build_model
        model = build_model(exp10.read_json(run / 'config.json'), spec, run / 'best.pt').eval()
        # Held-out individual slots have zero influence on new-subject predictions.
        x, subjects = torch.randn(2, 208, 300), torch.tensor([1, 1])
        with torch.no_grad():
            expected = model((x, subjects))[1]
            model.spatial_module.subject_layer.A[1].fill_(10000)
            actual = model((x, subjects))[1]
        torch.testing.assert_close(actual, expected)
        assert torch.isfinite(original).all()
    summary = pd.read_csv(out / 'summary/summary.csv')
    assert set(summary.variant) == {'individual', 'shared_only', 'switch'}
    assert not exp10.read_json(out / 'summary/coverage.json')['complete']
    # Reusing a completed directory with changed training settings is refused.
    changed = subprocess.run([sys.executable, str(runner), 'train', *common, '--nepoch', '2',
                              '--task', 'switch_p0p9_f0_s42'], capture_output=True, text=True, timeout=30)
    assert changed.returncode != 0 and 'provenance/settings changed' in changed.stderr
