"""Scientific invariants for calibration helpers; run with unittest discovery."""
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from scripts.exp07_frozen_calibration.frozen_loso_math import (apply_branch_map, apply_standardizer, effective_scaler,
                              fit_branch_map, fit_standardizer, population_standardizer,
                              sample_blocks, source_norm_gain)


class StandardizerTests(unittest.TestCase):
    def test_composed_robust_standard_scaling_matches_direct_fit(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(9, 4, 20)) * np.array([1, 2, 3, 4])[None, :, None] + 20
        x[:, 3, :] = 6
        flattened = x.transpose(1, 0, 2).reshape(4, -1)
        center = np.median(flattened, axis=1)
        robust_scale = np.diff(np.percentile(flattened, [25, 75], axis=1), axis=0)[0]
        robust_scale[robust_scale == 0] = 1
        robust = (flattened - center[:, None]) / robust_scale[:, None]
        standard_mean = robust.mean(axis=1)
        standard_scale = robust.std(axis=1)
        standard_scale[standard_scale == 0] = 1
        legacy = dict(robust_center=center, robust_scale=robust_scale,
                      standard_mean=standard_mean, standard_scale=standard_scale)
        new = fit_standardizer(x)
        for old, direct in zip(effective_scaler(legacy), effective_scaler(new)):
            np.testing.assert_allclose(old, direct, atol=1e-13)
        np.testing.assert_allclose(apply_standardizer(x, legacy), apply_standardizer(x, new), atol=1e-6)
        self.assertEqual(apply_standardizer(x, new).dtype, np.float32)
        json.dumps(new)

    def test_population_uses_only_explicit_sources(self):
        scalers = {"sub-01": {"offset": [1, 2], "scale": [3, 4]},
                   "sub-02": {"offset": [3, 6], "scale": [7, 8]},
                   "sub-03": {"offset": [100, 100], "scale": [100, 100]}}
        self.assertEqual(population_standardizer(scalers, [1, 2]),
                         {"offset": [2, 4], "scale": [5, 6]})
        with self.assertRaises(ValueError):
            population_standardizer(scalers, [])
        with self.assertRaises(ValueError):
            population_standardizer(scalers, [1, 1])

    def test_clipping_constant_channels_and_nonfinite_rejection(self):
        params = fit_standardizer(np.ones((2, 2, 3)))
        self.assertEqual(params["scale"], [1, 1])
        nonbinary_constant = fit_standardizer(np.full((9, 2, 300), 1.1))
        self.assertEqual(nonbinary_constant["scale"], [1, 1])
        np.testing.assert_array_equal(apply_standardizer(np.ones((1, 2, 1)) * 100, params), 20)
        np.testing.assert_array_equal(apply_standardizer(np.ones((1, 2, 1)) * -100, params), -20)
        with self.assertRaises(ValueError):
            fit_standardizer(np.full((1, 2, 3), np.nan))
        with self.assertRaises(ValueError):
            apply_standardizer(np.ones((1, 3, 1)), params)


class BranchMapTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(9)
        self.x = self.rng.normal(size=(8, 5, 30)) + np.arange(5)[None, :, None]
        self.bias = np.array([1, 3, -1, 2, 0.5])

    def test_scalar_recovers_signed_scale_and_translation(self):
        y = -2 * self.x + self.bias[None, :, None]
        fitted = fit_branch_map(self.x, y, "scalar")
        np.testing.assert_allclose(fitted["matrix"], -2 * np.eye(5), atol=1e-12)
        np.testing.assert_allclose(fitted["bias"], self.bias, atol=1e-12)
        np.testing.assert_allclose(apply_branch_map(self.x, fitted), y, atol=1e-12)

    def test_diagonal_recovers_distinct_scales_and_translation(self):
        gains = np.array([2, -1, 0, 0.5, 4])
        y = gains[None, :, None] * self.x + self.bias[None, :, None]
        fitted = fit_branch_map(self.x, y, "diagonal")
        np.testing.assert_allclose(fitted["matrix"], np.diag(gains), atol=1e-12)
        np.testing.assert_allclose(apply_branch_map(self.x, fitted), y, atol=1e-12)

    def test_procrustes_recovers_rotation_and_reflection_in_correct_direction(self):
        q, _ = np.linalg.qr(self.rng.normal(size=(5, 5)))
        if np.linalg.det(q) > 0:
            q[0] *= -1
        self.assertLess(np.linalg.det(q), 0)
        y = np.einsum("ij,njt->nit", q, self.x) + self.bias[None, :, None]
        fitted = fit_branch_map(self.x, y, "procrustes")
        np.testing.assert_allclose(fitted["matrix"], q, atol=1e-12)
        np.testing.assert_allclose(fitted["matrix"] @ fitted["matrix"].T, np.eye(5), atol=1e-12)
        np.testing.assert_allclose(apply_branch_map(self.x, fitted), y, atol=1e-12)

    def test_constant_target_has_safe_identity_and_center_translation(self):
        x, y = np.full((3, 5, 100), 1.1), np.full((3, 5, 100), 4.2)
        for kind in ("scalar", "diagonal", "procrustes"):
            with self.subTest(kind=kind):
                fitted = fit_branch_map(x, y, kind)
                np.testing.assert_array_equal(fitted["matrix"], np.eye(5))
                np.testing.assert_allclose(apply_branch_map(x, fitted), y)

    def test_invalid_shapes_and_nonfinite_inputs_fail(self):
        with self.assertRaises(ValueError):
            fit_branch_map(self.x, self.x[:, :, :-1], "scalar")
        with self.assertRaises(ValueError):
            fit_branch_map(self.x, self.x, "unknown")
        with self.assertRaises(ValueError):
            fit_branch_map(self.x, self.x * np.nan, "diagonal")

    def test_norm_gain_corrects_rms_source_weight_norm(self):
        ws = self.rng.normal(size=(7, 5, 8))
        corrected = ws.mean(axis=0) * source_norm_gain(ws)
        rms_norm = np.sqrt(np.square(ws).sum(axis=(1, 2)).mean())
        self.assertAlmostEqual(np.linalg.norm(corrected), rms_norm, places=12)
        with self.assertRaises(ValueError):
            source_norm_gain(np.array([np.eye(2), -np.eye(2)]))


def make_rows(n, *, first_id=0, file="a.wav", subject=1, split="train", task=0, session=0):
    return [dict(window_id=first_id + i, subject=subject, split=split,
                 audio_file=file, task=task, session=session,
                 audio_start_seconds=1 + 3 * i,
                 meg_start_sample_1000hz=2000 + 3000 * i) for i in range(n)]


class BlockSamplingTests(unittest.TestCase):
    def test_exact_budget_reproducible_source_and_split_exclusion(self):
        rows = make_rows(30) + make_rows(40, first_id=100, subject=2)
        rows += make_rows(40, first_id=200, split="validation")
        rows += make_rows(40, first_id=300, split="test")
        first = sample_blocks(rows, 1, 30, 3, 42)
        self.assertEqual(first, sample_blocks(list(reversed(rows)), 1, 30, 3, 42))
        self.assertEqual(len({tuple(block) for block in first}), 3)
        for block in first:
            self.assertEqual(len(block), 10)
            self.assertTrue(all(0 <= index < 30 for index in block))
            self.assertEqual(block, list(range(block[0], block[0] + 10)))
        self.assertEqual(sample_blocks(rows, 1, 0, 3, 42), [[]])

    def test_does_not_cross_fragment_task_session_or_raw_gap(self):
        for changed in ("audio_file", "task", "session", "raw_gap", "audio_gap"):
            rows = make_rows(20)
            for row in rows[10:]:
                if changed == "audio_file": row[changed] = "b.wav"
                elif changed in ("task", "session"): row[changed] = 1
                elif changed == "raw_gap": row["meg_start_sample_1000hz"] += 2000
                else: row["audio_start_seconds"] += 1
            with self.subTest(changed=changed):
                with self.assertRaisesRegex(ValueError, "longest available is 30s"):
                    sample_blocks(rows, 1, 33, 3, 0)
                for block in sample_blocks(rows, 1, 30, 3, 0):
                    self.assertIn(block, [list(range(10)), list(range(10, 20))])

    def test_metadata_string_rows_and_optional_raw_column(self):
        rows = [{key: str(value) for key, value in row.items()
                 if key != "meg_start_sample_1000hz"} for row in make_rows(12)]
        self.assertEqual(len(sample_blocks(rows, 1, 30, 1, 0)[0]), 10)

    def test_invalid_budgets_and_duplicate_ids(self):
        rows = make_rows(20)
        for budget in (-3, 31, float("nan"), float("inf")):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                sample_blocks(rows, 1, budget, 3, 0)
        with self.assertRaises(ValueError):
            sample_blocks(rows, 1, 30, 0, 0)
        with self.assertRaisesRegex(ValueError, "Duplicate window_id"):
            sample_blocks(rows + rows, 1, 30, 3, 0)


if __name__ == "__main__":
    unittest.main()
