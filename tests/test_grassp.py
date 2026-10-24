from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from duet_clean.experiments import manifest_splits
from duet_clean.grassp import load_grassp_splits, to_grassp_candidates


class GrasspDataTests(unittest.TestCase):
    def test_id_candidates_handle_chainless_te18_names(self) -> None:
        self.assertEqual(to_grassp_candidates("1ddy_A"), ["1DDYA", "1DDY"])
        self.assertEqual(to_grassp_candidates("8hb1_B"), ["8HB1B"])

    def test_p80_graphs_match_manifest_lengths_and_labels(self) -> None:
        asset = (
            ROOT
            / "data"
            / "grassp"
            / "processed"
            / "HARI60NEW_TRAIN"
            / "processed"
            / "processed_data_hariboss.pt"
        )
        if not asset.exists():
            self.skipTest("requires the GRASSP processed-data release")
        raw, _ = manifest_splits("homology_p80")
        splits = load_grassp_splits(raw)
        self.assertEqual(len(splits["train"]), 79)
        self.assertEqual(len(splits["val"]), 22)
        self.assertEqual(len(splits["TE18"]), 17)
        self.assertEqual(len(splits["TL12"]), 12)
        self.assertEqual(len(splits["JL10"]), 10)


if __name__ == "__main__":
    unittest.main()
