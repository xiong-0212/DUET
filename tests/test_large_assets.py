from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class LargeAssetTests(unittest.TestCase):
    def test_large_graph_counts_and_features(self) -> None:
        expected = {
            "graph_large_p80_cv5.pt": 640,
            "dual_large_p80_cv5.pt": 1920,
            "graph_seq3d_large_p80_cv5.pt": 640,
            "graph_seqss3d_large_p80_cv5.pt": 640,
            "graph_seq3d_ernie_large_p80_cv5.pt": 768,
            "grassp_rnafm_large_p80_cv5.pt": 640,
            "grassp_ernie_large_p80_cv5.pt": 768,
        }
        if not all(
            (ROOT / "data" / "processed" / name).exists()
            for name in expected
        ):
            self.skipTest("requires generated graph assets")
        for name, dimension in expected.items():
            payload = torch.load(
                ROOT / "data" / "processed" / name,
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(len(payload["graphs"]["pool"]), 290)
            self.assertEqual(payload["graphs"]["pool"][0].x.size(1), dimension)

    def test_external_assets_are_complete(self) -> None:
        asset = ROOT / "data" / "processed" / "smartpocket_external_graphs.pt"
        if not asset.exists():
            self.skipTest("requires generated external graph assets")
        payload = torch.load(
            asset,
            map_location="cpu",
            weights_only=False,
        )
        self.assertEqual(len(payload["graphs"]["seqss3d_rnafm"]), 19)
        sets = json.loads(
            (ROOT / "data" / "external" / "external_sets.json").read_text(
                encoding="utf-8"
            )
        )["splits"]
        self.assertEqual(len(sets["external_jiang"]), 2)
        self.assertEqual(len(sets["time_dependent"]), 11)
        self.assertEqual(len(sets["hariboss_test"]), 6)

    def test_independent_external_sets_have_no_80pct_overlap(self) -> None:
        audit = json.loads(
            (
                ROOT
                / "results"
                / "large_protocol"
                / "smartpocket_audit.json"
            ).read_text(encoding="utf-8")
        )
        for split_name in ("external_jiang", "time_dependent"):
            for record in audit["splits"][split_name]["records"].values():
                if record["status"] != "retained":
                    continue
                self.assertLess(
                    record["max_identity_to_large_pool"]["identity"],
                    0.8,
                )
                self.assertLess(
                    record["max_identity_to_canonical_tests"]["identity"],
                    0.8,
                )


if __name__ == "__main__":
    unittest.main()
