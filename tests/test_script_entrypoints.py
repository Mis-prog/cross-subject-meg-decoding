"""CLI smoke checks from outside the repository; no downloads or real training."""
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'code/scripts'


@pytest.mark.parametrize('script', [
    'exp05_mini_loso/download_mini_loso.py',
    'exp06_loso/download_loso.py',
    'exp07_frozen_calibration/frozen_loso.py',
    'exp07_frozen_calibration/run_frozen_loso.py',
    'exp08_09_full_lisa/run_full_series.py',
    'exp08_matched22/run_matched22.py',
    'exp09_new_subjects/eval_mean_seen.py',
    'exp10_shared_matrix/run_experiment.py',
])
def test_help_from_unrelated_working_directory(script, tmp_path):
    result = subprocess.run([sys.executable, str(SCRIPTS / script), '--help'],
                            cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'usage:' in result.stdout.lower()


def test_full_pipeline_finds_default_protocol_outside_repository(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / 'exp08_09_full_lisa/full_lisa.py'), 'tasks'],
        cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    tasks = {line.split()[0] for line in result.stdout.splitlines()}
    assert {'repro_s42', 'xsA_s42', 'xsB_s42', 'ns_f0_s42'} <= tasks
