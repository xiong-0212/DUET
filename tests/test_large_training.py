from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from duet_clean.large_training import weighted_order


class LargeTrainingTests(unittest.TestCase):
    def test_weighted_order_respects_group_weights(self) -> None:
        weights = {"a": 1.0, "b": 1.0, "c": 0.5, "d": 0.5}
        counts = {"a": 0, "b": 0, "c": 0, "d": 0}
        generator = torch.Generator().manual_seed(7)
        for _ in range(1000):
            index = weighted_order(list(weights), weights, generator)[0]
            counts[list(weights)[index]] += 1
        self.assertGreater(counts["a"], counts["c"] * 1.5)
        self.assertGreater(counts["b"], counts["d"] * 1.5)


if __name__ == "__main__":
    unittest.main()
