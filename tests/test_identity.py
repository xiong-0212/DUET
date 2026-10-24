from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from duet_clean.identity import IdentityEngine


class IdentityEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = IdentityEngine(min_length_ratio=0.8)

    def test_exact_sequences_have_identity_one(self) -> None:
        result = self.engine.compare("ACGTACGT", "ACGTACGT")
        self.assertTrue(result.comparable)
        self.assertEqual(result.standard_identity, 1.0)
        self.assertEqual(result.containment_identity, 1.0)

    def test_single_mismatch(self) -> None:
        result = self.engine.compare("ACGTACGT", "ACGTTCGT")
        self.assertTrue(result.comparable)
        self.assertAlmostEqual(result.standard_identity, 7 / 8)
        self.assertAlmostEqual(result.containment_identity, 7 / 8)

    def test_containment_identity_is_not_used_for_selection(self) -> None:
        longer = "ACGT" * 10
        shorter = "ACGT" * 8
        result = self.engine.compare(longer, shorter)
        self.assertTrue(result.comparable)
        self.assertAlmostEqual(result.standard_identity, 0.8)
        self.assertAlmostEqual(result.containment_identity, 1.0)

    def test_length_gate(self) -> None:
        result = self.engine.compare("ACGT" * 10, "ACGT" * 7)
        self.assertFalse(result.comparable)
        self.assertIsNone(result.standard_identity)


if __name__ == "__main__":
    unittest.main()
