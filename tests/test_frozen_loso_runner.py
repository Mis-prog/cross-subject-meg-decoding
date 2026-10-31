"""Scheduler and statistical-unit tests; no LISA, GPU, or real data required."""
from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from scripts.exp07_frozen_calibration import run_frozen_loso as runner
from scripts.exp07_frozen_calibration import summarize_frozen_loso as summary


class RunnerTests(unittest.TestCase):
    def test_dry_run_needs_no_data_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as temp:
            absent = Path(temp) / "absent"
            out = io.StringIO()
            with redirect_stdout(out):
                code = runner.main(["--stage", "all", "--dry-run", "--protocol", str(absent / "protocol.json"),
                                    "--data-dir", str(absent / "data"), "--output-dir", str(absent / "output")])
            self.assertEqual(code, 0)
            self.assertFalse(absent.exists())
            self.assertEqual(out.getvalue().count(" tune "), 3)
            self.assertEqual(out.getvalue().count(" evaluate "), 72)
            self.assertIn(" choose ", out.getvalue())

    def test_subset_only_affects_evaluation(self):
        args = runner.parser().parse_args(["--stage", "all", "--dry-run", "--holdouts", "4", "--seeds", "2"])
        stages = runner.build_plan(args)
        self.assertEqual([len(stage) for stage in stages], [1, 3, 1, 1, 1])
        self.assertEqual(stages[1][0].name, "tune_h01_s0")
        self.assertEqual(stages[3][0].name, "evaluate_h04_s2")

    def test_dev_subject_cannot_enter_evaluation(self):
        args = runner.parser().parse_args(["--stage", "evaluate", "--dry-run", "--holdouts", "1"])
        with self.assertRaises(ValueError):
            runner.build_plan(args)

    def test_failure_prevents_launching_pending_job(self):
        with tempfile.TemporaryDirectory() as temp:
            marker = Path(temp) / "should_not_exist"
            jobs = [runner.Job("bad", [sys.executable, "-c", "raise SystemExit(7)"]),
                    runner.Job("next", [sys.executable, "-c", f"open({str(marker)!r}, 'w').write('started')"])]
            with redirect_stdout(io.StringIO()):
                success = runner.run_stage(jobs, 1, Path(temp))
            self.assertFalse(success)
            self.assertFalse(marker.exists())
            self.assertTrue((Path(temp) / "bad.log").exists())

    def test_failure_stops_running_children(self):
        with tempfile.TemporaryDirectory() as temp:
            jobs = [runner.Job("bad", [sys.executable, "-c", "raise SystemExit(7)"]),
                    runner.Job("long", [sys.executable, "-c", "import time; time.sleep(30)"])]
            start = time.monotonic()
            with redirect_stdout(io.StringIO()):
                success = runner.run_stage(jobs, 2, Path(temp))
            self.assertFalse(success)
            self.assertLess(time.monotonic() - start, 10)

    def test_generated_commands_match_actual_worker_and_summary_cli(self):
        args = runner.parser().parse_args(["--stage", "all", "--dry-run", "--holdouts", "4", "--seeds", "0"])
        for stage in runner.build_plan(args):
            job = stage[0]
            result = subprocess.run([*job.command, "--help"], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, f"{job.name}: {result.stderr}")
            self.assertIn("usage:", result.stdout)


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.protocol = {"subjects": [1, 2, 3, 4, 5], "dev_subjects": [1, 2, 3], "seeds": [0, 1],
                         "budgets_seconds": [0, 30, 120], "repeats": 3}
        self.path = self.folder / "protocol.json"
        self.path.write_text(json.dumps(self.protocol))
        self.hash = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.tuning_path = self.folder / "tuning.json"
        self.tuning_path.write_text(json.dumps({"settings": "fixed on dev"}))
        self.tuning_hash = hashlib.sha256(self.tuning_path.read_bytes()).hexdigest()

    def receipt(self, person, seed):
        records = []
        for method, budget, repeat in sorted(summary.expected_record_keys([0, 30, 120], 3)):
            top10 = .2 + .1 * (person - 4) + .01 * seed + .005 * repeat
            if method == "align_procrustes":
                top10 += .02
            records.append({"method": method, "setting": "chosen", "budget_seconds": budget, "repeat": repeat,
                            "top1": .1, "top10": top10, "mean_rank": 40., "loss": 3., "queries": 99, "candidates": 99})
        result = {"schema_version": 1, "status": "complete", "stage": "evaluate", "holdout": person,
                  "seed": seed, "fingerprint": f"job-{person}-{seed}", "protocol_sha256": self.hash,
                  "tuning_sha256": self.tuning_hash, "records": records}
        result["receipt_sha256"] = hashlib.sha256(json.dumps(result, sort_keys=True, allow_nan=False).encode()).hexdigest()
        return result

    def write_receipt(self, person, seed):
        path = self.folder / "evaluate" / f"h{person:02d}_s{seed}.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(self.receipt(person, seed)))
        return path

    def test_people_are_unit_and_comparison_is_paired(self):
        for person in [4, 5]:
            for seed in [0, 1]:
                self.write_receipt(person, seed)
        report = summary.summarize(self.folder, self.path, draws=100)
        self.assertFalse(report["partial"])
        row = next(row for row in report["summary"] if row["method"] == "gain_mean_ws"
                   and row["budget_seconds"] == 30 and row["metric"] == "top10")
        self.assertEqual(row["subjects"], 2)  # Not 4 seeds or 12 blocks.
        self.assertAlmostEqual(row["mean"], .26)
        paired = next(row for row in report["paired_vs_gain_mean_ws"] if row["method"] == "align_procrustes"
                      and row["budget_seconds"] == 30 and row["metric"] == "top10")
        self.assertAlmostEqual(paired["mean_difference"], .02)
        self.assertEqual(paired["improved_fraction"], 1)

    def test_missing_seed_excludes_person_and_marks_partial(self):
        self.write_receipt(4, 0)
        self.write_receipt(4, 1)
        self.write_receipt(5, 0)
        report = summary.summarize(self.folder, self.path, draws=100)
        self.assertTrue(report["partial"])
        self.assertEqual(report["excluded_incomplete_subjects"], [5])
        self.assertEqual({row["subjects"] for row in report["summary"]}, {1})
        self.assertTrue(all(row["ci95_low"] is None for row in report["summary"]))

    def test_subset_is_explicitly_partial_even_when_all_requested_jobs_complete(self):
        self.write_receipt(4, 0)
        report = summary.summarize(self.folder, self.path, holdouts=[4], seeds=[0], draws=100)
        self.assertTrue(report["partial"])
        self.assertTrue(report["requested_subset_complete"])

    def test_rejects_duplicate_result_and_mixed_protocol(self):
        receipt = self.receipt(4, 0)
        expected = summary.expected_record_keys([0, 30, 120], 3)
        receipt["records"].append(receipt["records"][0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            summary.validate_receipt(receipt, self.path, 4, 0, expected, self.hash)
        receipt = self.receipt(4, 0)
        receipt["protocol_sha256"] = "other"
        with self.assertRaisesRegex(ValueError, "Protocol"):
            summary.validate_receipt(receipt, self.path, 4, 0, expected, self.hash)

    def test_rejects_missing_rows_in_claimed_complete_receipt(self):
        receipt = self.receipt(4, 0)
        receipt["records"].pop()
        with self.assertRaisesRegex(ValueError, "missing records"):
            summary.validate_receipt(receipt, self.path, 4, 0, summary.expected_record_keys([0, 30, 120], 3), self.hash)

    def test_tuning_hash_must_match_saved_selection(self):
        self.write_receipt(4, 0)
        self.tuning_path.write_text(json.dumps({"settings": "changed"}))
        with self.assertRaisesRegex(ValueError, "Tuning hash mismatch"):
            summary.summarize(self.folder, self.path, holdouts=[4], seeds=[0], draws=100)

    def test_tuning_hash_must_agree_between_receipts_without_tuning_file(self):
        self.write_receipt(4, 0)
        path = self.write_receipt(4, 1)
        receipt = json.loads(path.read_text())
        receipt["tuning_sha256"] = "a" * 64
        path.write_text(json.dumps(receipt))
        self.tuning_path.unlink()
        with self.assertRaisesRegex(ValueError, "Tuning hash mismatch"):
            summary.summarize(self.folder, self.path, holdouts=[4], draws=100)

    def test_missing_tuning_hash_rejected(self):
        receipt = self.receipt(4, 0)
        receipt.pop("tuning_sha256")
        with self.assertRaisesRegex(ValueError, "tuning_sha256"):
            summary.validate_receipt(receipt, self.path, 4, 0, summary.expected_record_keys([0, 30, 120], 3), self.hash)

    def test_corrupted_metric_rejected_by_receipt_integrity(self):
        receipt = self.receipt(4, 0)
        receipt["records"][0]["top10"] = .9
        with self.assertRaisesRegex(ValueError, "integrity"):
            summary.validate_receipt(receipt, self.path, 4, 0, summary.expected_record_keys([0, 30, 120], 3), self.hash)


if __name__ == "__main__":
    unittest.main()
