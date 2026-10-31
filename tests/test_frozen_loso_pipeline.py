"""Exercise the full worker flow on synthetic data, with no SciPy/GPU/real MEG.

The LISA constructor and full-dataset provenance checks are substituted; actual
Adapter operations, calibration, selection, receipts and summary all execute.
"""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from scripts.exp07_frozen_calibration import frozen_loso as worker
from scripts.exp07_frozen_calibration.frozen_loso_math import fit_standardizer, apply_standardizer
from scripts.exp07_frozen_calibration.summarize_frozen_loso import summarize
from test_frozen_loso_adapter import TinyLISA


class LoggedArray:
    def __init__(self, values):
        self.values = values
        self.reads = []

    def __getitem__(self, indices):
        self.reads.extend(list(indices))
        return self.values[indices]


class TinyData:
    ids = worker.Data.ids
    audio_ids = worker.Data.audio_ids
    bank_ids = worker.Data.bank_ids
    matched_source_ids = worker.Data.matched_source_ids

    def __init__(self):
        rng = np.random.default_rng(14)
        self.rows, self.audio_rows, self.by_subject_audio = [], [], {}
        for a in range(54):
            split = "train" if a < 48 else "validation" if a < 51 else "test"
            self.audio_rows.append({"audio_id": a, "split": split})
        for subject in range(1, 28):
            for a, audio in enumerate(self.audio_rows):
                i = len(self.rows)
                self.rows.append({"window_id": i, "audio_id": a, "subject": subject,
                                  "model_subject_index": subject - 1, "split": audio["split"],
                                  "audio_file": "easy_money_2.wav" if a < 48 else "lw1_3.wav",
                                  "audio_start_seconds": a * 3 + 1, "task": 2, "session": 0,
                                  "meg_start_sample_1000hz": a * 3000,
                                  "meg_stop_sample_1000hz": (a + 1) * 3000})
                self.by_subject_audio[subject, a] = i
        raw = rng.normal(size=(len(self.rows), 3, 11)).astype(np.float32)
        scaled = np.empty_like(raw)
        self.scalers = {}
        for subject in range(1, 28):
            ids = self.ids(subject, "train")
            params = fit_standardizer(raw[ids])
            self.scalers[f"sub-{subject:02d}"] = params
            all_ids = [i for i, r in enumerate(self.rows) if r["subject"] == subject]
            scaled[all_ids] = apply_standardizer(raw[all_ids], params)
        self.raw = LoggedArray(raw)
        self.scaled = LoggedArray(scaled)
        self.audio = LoggedArray(rng.normal(size=(54, 11, 4)).astype(np.float32))


def test_tune_choose_evaluate_summary_resume_and_split_isolation(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(12)
    model = TinyLISA()
    model.spatial_module.subject_layer.A = torch.nn.Parameter(torch.randn(27, 2, 4))
    data = TinyData()
    protocol_path = Path(worker.ROOT / "experiments/exp07_frozen_calibration/protocol.json")
    protocol = worker.read_json(protocol_path)
    args = SimpleNamespace(command="tune", holdout=1, seed=0, device="cpu", batch_size=16,
                           coords_dir=tmp_path, output_dir=tmp_path, runs_dir=tmp_path,
                           data_dir=tmp_path, protocol=protocol_path)
    manifest = {"fingerprint": "synthetic-manifest", "files": {"data/protocol.json": {"sha256": "original"}}}
    for subject in (1, 2, 3, 4):
        path = worker.checkpoint_path(args, subject, 0)
        torch.save({"holdout": subject, "seed": 0,
                    "fit_subjects": [s for s in range(1, 28) if s != subject],
                    "model_state_dict": model.state_dict(),
                    "criterion_state_dict": {"temperature": torch.tensor(-1.2)}}, path)
        manifest["files"][f"checkpoint/{subject}/0"] = {"path": str(path.resolve()), "sha256": worker.sha256(path)}
    actual_adapter = worker.Adapter

    def adapter_factory(config, saved):
        clone = deepcopy(model)
        clone.load_state_dict(saved["model_state_dict"])
        return actual_adapter(config, saved, model=clone)

    with (patch.object(worker, "load_manifest", return_value=manifest),
          patch.object(worker, "Data", return_value=data),
          patch.object(worker, "Adapter", side_effect=adapter_factory),
          patch.object(worker, "validate_checkpoint"), redirect_stdout(io.StringIO())):
        for subject in (1, 2, 3):
            args.holdout = subject
            data.raw.reads.clear()
            data.audio.reads.clear()
            worker.run_job(args, protocol)
            assert all(data.rows[i]["subject"] == subject for i in data.raw.reads)
            assert all(data.rows[i]["split"] in ("train", "validation") for i in data.raw.reads)
            assert all(data.audio_rows[i]["split"] in ("train", "validation") for i in data.audio.reads)
        args.command = "choose"
        worker.choose(args, protocol)
        tuning_bytes = (tmp_path / "tuning.json").read_bytes()
        args.command, args.holdout = "evaluate", 4
        data.raw.reads.clear()
        data.scaled.reads.clear()
        worker.run_job(args, protocol)
        receipt_path = worker.receipt_path(args, "evaluate", 4, 0)
        before = receipt_path.read_bytes()
        receipt = json.loads(before)
        assert len(receipt["records"]) == 51
        assert receipt["audit"]["neural_parameters_unchanged"]
        assert all(data.rows[i]["subject"] == 4 for i in data.raw.reads)
        assert all(data.rows[i]["subject"] != 4 and data.rows[i]["split"] == "train" for i in data.scaled.reads)
        for calibration in receipt["calibrations"]:
            assert all(data.rows[i]["split"] == "train" for i in calibration["window_ids"])
            if calibration["budget_seconds"]:
                assert calibration["selected_subject"] != 4
                assert 4 not in calibration["template_source_subjects"]
                assert set(calibration["maps"]) == {"scalar", "diagonal", "procrustes"}
        thirty = [c["window_ids"] for c in receipt["calibrations"] if c["budget_seconds"] == 30]
        longer = [c["window_ids"] for c in receipt["calibrations"] if c["budget_seconds"] == 120]
        assert thirty == [ids[:10] for ids in longer]
        summary = summarize(tmp_path, protocol_path, holdouts=[4], seeds=[0], draws=100)
        assert summary["requested_subset_complete"] and summary["record_count"] == 51
        assert summary["partial"]  # one synthetic person is never the full experiment
        worker.run_job(args, protocol)
        assert receipt_path.read_bytes() == before
        assert (tmp_path / "tuning.json").read_bytes() == tuning_bytes
        args.batch_size = 8
        with pytest.raises(ValueError, match="different provenance"):
            worker.run_job(args, protocol)


def test_real_checkpoint_provenance_rejects_heldout_source():
    # This test reads only metadata/tensors; it does not instantiate LISA.
    path = worker.ROOT / "data/pilot/loso_runs/shared_h01_s0.pt"
    if not path.exists():
        pytest.skip("Optional local original checkpoint unavailable")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    protocol = worker.read_json(worker.ROOT / "experiments/exp07_frozen_calibration/protocol.json")
    checksum = worker.sha256(worker.ROOT / "experiments/exp06_loso/protocol.json")
    worker.validate_checkpoint(saved, 1, 0, protocol, checksum)
    saved["fit_subjects"] = list(range(1, 27))
    with pytest.raises(ValueError, match="exclude exactly"):
        worker.validate_checkpoint(saved, 1, 0, protocol, checksum)


def test_receipt_rejects_changed_values_and_balancing_is_by_budget(tmp_path):
    path = tmp_path / "receipt.json"
    worker.finish_receipt(path, {"status": "complete", "fingerprint": "same", "records": [{"loss": 1.0}]})
    altered = worker.read_json(path)
    altered["records"][0]["loss"] = 0.01
    worker.atomic_json(path, altered)
    with pytest.raises(ValueError, match="integrity"):
        worker.try_resume(path, "same")
    receipts = [{"records": [{"method": "gain_mean_ws", "budget_seconds": 0, "loss": 10}]
                 + [{"method": "gain_mean_ws", "budget_seconds": 30, "loss": 2}] * 3}]
    assert worker.balanced_loss(receipts, "gain_mean_ws", lambda row: True) == 6
