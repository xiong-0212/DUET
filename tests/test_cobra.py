from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from duet_clean.cobra import CoBRAHead
from duet_clean.comparison import holm_adjust


class CoBRAHeadTests(unittest.TestCase):
    def test_head_returns_two_logits(self) -> None:
        model = CoBRAHead()
        output = model(torch.zeros(7, 1280))
        self.assertEqual(tuple(output.shape), (7, 2))

    def test_holm_adjustment_is_monotone(self) -> None:
        adjusted = holm_adjust([0.01, 0.04, 0.03])
        self.assertAlmostEqual(adjusted[0], 0.03)
        self.assertAlmostEqual(adjusted[1], 0.06)
        self.assertAlmostEqual(adjusted[2], 0.06)


if __name__ == "__main__":
    unittest.main()
