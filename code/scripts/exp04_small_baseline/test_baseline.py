"""Scientific-integrity checks for fragment splits and complete-bank retrieval."""
import unittest

import pandas as pd
import torch

import sys
from pathlib import Path

# Support direct execution from any working directory.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.exp04_small_baseline.prepare_baseline import validate_windows
from scripts.exp04_small_baseline.train_baseline import score_embeddings


class BaselineChecks(unittest.TestCase):
    def windows(self):
        return pd.DataFrame({
            'window_id': [0, 1, 2], 'task': [0, 0, 0],
            'audio_file': ['a.wav', 'b.wav', 'c.wav'],
            'split': ['train', 'validation', 'test'],
            'meg_start_sample_1000hz': [0, 10000, 20000],
            'meg_stop_sample_1000hz': [3000, 13000, 23000],
        })

    def test_fragment_cannot_cross_splits(self):
        windows = self.windows()
        validate_windows(windows)
        windows.loc[2, 'audio_file'] = 'a.wav'
        with self.assertRaisesRegex(ValueError, 'multiple splits'):
            validate_windows(windows)

    def test_overlap_in_raw_time_rejected(self):
        windows = self.windows()
        windows.loc[2, 'meg_start_sample_1000hz'] = 12000
        windows.loc[2, 'meg_stop_sample_1000hz'] = 15000
        with self.assertRaisesRegex(ValueError, 'Overlapping'):
            validate_windows(windows)

    def test_full_rank_includes_miss_outside_top10(self):
        candidates = torch.cat((torch.eye(11), -torch.eye(11)[:1]))
        ids = torch.arange(12) + 100
        losses, ranks, predictions = score_embeddings(
            torch.eye(11)[:1], candidates, ids[-1:], ids, torch.tensor(1.0))
        self.assertEqual(ranks.item(), 12)
        self.assertEqual(predictions.item(), 100)
        self.assertTrue(torch.isfinite(losses).all())

    def test_candidate_order_and_query_batching_do_not_change_scores(self):
        candidates = torch.eye(12)
        ids = torch.arange(12) + 100
        labels = ids[[3, 8, 11]]
        predictions = candidates[[3, 8, 11]]
        original = score_embeddings(predictions, candidates, labels, ids, torch.tensor(0.5))
        order = torch.tensor([11, 0, 7, 3, 1, 2, 4, 5, 8, 6, 10, 9])
        reordered = score_embeddings(predictions, candidates[order], labels, ids[order], torch.tensor(0.5))
        for a, b in zip(original, reordered):
            torch.testing.assert_close(a, b)
        separate = [score_embeddings(predictions[i:i+1], candidates, labels[i:i+1], ids, torch.tensor(0.5)) for i in range(3)]
        for column, expected in enumerate(original):
            torch.testing.assert_close(torch.cat([r[column] for r in separate]), expected)


if __name__ == '__main__':
    unittest.main()
