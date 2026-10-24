from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from duet_clean.identity import IdentityEngine
from duet_clean.large_protocol import (
    LARGE_MANIFEST,
    connected_components,
    load_large_manifest,
    sampling_weights,
)
from duet_clean.models import Record


def record(rna_id: str, sequence: str) -> Record:
    return Record(
        split="pool",
        source="test",
        rna_id=rna_id,
        pdb_id=rna_id[:4],
        chain=rna_id[-1],
        sequence=sequence,
        labels="0" * len(sequence),
        ordinal=0,
    )


class LargeProtocolTests(unittest.TestCase):
    def test_connected_components_do_not_split_transitive_homologs(self) -> None:
        records = [
            record("1abc_A", "ACGT" * 25),
            record("2abc_A", "ACGT" * 25 + "A"),
            record("3abc_A", "TGCA" * 25),
        ]
        engine = IdentityEngine(min_length_ratio=0.8)
        clusters = connected_components(records, 0.8, engine)
        self.assertEqual(sorted(len(cluster) for cluster in clusters), [1, 2])

    def test_manifest_retains_duplicates_and_whole_cluster_folds(self) -> None:
        path = ROOT / "data" / "clean" / "manifests" / f"{LARGE_MANIFEST}.json"
        if not path.exists():
            self.skipTest("large protocol manifest has not been generated")
        _, manifest = load_large_manifest()
        self.assertGreater(
            manifest["audit"]["retained_exact_duplicate_records"],
            0,
        )
        fold_by_cluster = {}
        for cluster in manifest["clusters"]:
            folds = set()
            for fold in manifest["folds"]:
                members = set(fold["validation_ids"])
                if set(cluster["rna_ids"]) & members:
                    folds.add(int(fold["fold"]))
            self.assertEqual(len(folds), 1)
            fold_by_cluster[int(cluster["cluster_id"])] = folds.pop()
        for cluster_id, fold in fold_by_cluster.items():
            self.assertEqual(
                fold,
                int(manifest["clusters"][cluster_id]["fold"]),
            )
        weights = sampling_weights(manifest)
        self.assertTrue(all(0 < value <= 1 for value in weights.values()))


if __name__ == "__main__":
    unittest.main()
