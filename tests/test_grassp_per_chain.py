from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from duet_clean.grassp import _per_chain


class GrasspPerChainTests(unittest.TestCase):
    def test_batched_predictions_are_split_by_chain(self) -> None:
        chains = [
            ("a", Data(y=torch.zeros(2))),
            ("b", Data(y=torch.zeros(3))),
        ]
        probabilities = [np.array([0.1, 0.2, 0.3, 0.4, 0.5])]
        labels = [np.array([0, 0, 1, 0, 1])]
        result = _per_chain(probabilities, labels, chains, 0.5)
        self.assertEqual(len(result["a"]["prob"]), 2)
        self.assertEqual(len(result["b"]["prob"]), 3)
        self.assertEqual(result["b"]["label"], [1, 0, 1])


if __name__ == "__main__":
    unittest.main()
