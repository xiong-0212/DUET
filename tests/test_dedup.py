from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from duet_clean.dedup import (
    canonicalize_tests,
    cluster_homologs,
    collapse_exact_sequences,
)
from duet_clean.identity import IdentityEngine
from duet_clean.models import Record


def record(
    split: str,
    rna_id: str,
    sequence: str,
    labels: str | None = None,
    ordinal: int = 0,
) -> Record:
    return Record(
        split=split,
        source=split,
        rna_id=rna_id,
        pdb_id=rna_id[:4],
        chain=rna_id[-1],
        sequence=sequence,
        labels=labels or ("0" * len(sequence)),
        ordinal=ordinal,
    )


class DedupTests(unittest.TestCase):
    def test_test_priority_is_deterministic(self) -> None:
        records = [
            record("RB9", "2abc_A", "ACGT"),
            record("TE18", "1abc_A", "ACGT"),
        ]
        kept, removals, groups = canonicalize_tests(records, ["TE18", "RB9"])
        self.assertEqual([item.rna_id for item in kept], ["1abc_A"])
        self.assertEqual([item.record.rna_id for item in removals], ["2abc_A"])
        self.assertEqual(len(groups), 1)

    def test_conflicting_labels_drop_whole_sequence_family(self) -> None:
        records = [
            record("train", "1abc_A", "ACGT", labels="0101"),
            record("train", "2abc_A", "ACGT", labels="1010"),
        ]
        kept, removals, groups = collapse_exact_sequences(records, "exact_sequence_unique")
        self.assertEqual(kept, [])
        self.assertEqual(len(removals), 2)
        self.assertTrue(all(item.reason == "ambiguous_same_sequence_different_labels" for item in removals))
        self.assertTrue(groups[0].label_conflict)

    def test_same_label_duplicates_keep_one(self) -> None:
        records = [
            record("train", "2abc_A", "ACGT", labels="0101"),
            record("train", "1abc_A", "ACGT", labels="0101"),
        ]
        kept, removals, _ = collapse_exact_sequences(records, "exact_sequence_unique")
        self.assertEqual([item.rna_id for item in kept], ["1abc_A"])
        self.assertEqual([item.record.rna_id for item in removals], ["2abc_A"])

    def test_greedy_homology_cluster_keeps_longer_representative(self) -> None:
        records = [
            record("train", "short_A", "ACGT" * 8),
            record("train", "long_A", "ACGT" * 10),
        ]
        kept, removals, clusters = cluster_homologs(
            records, 0.8, IdentityEngine(min_length_ratio=0.8), "training_cluster"
        )
        self.assertEqual([item.rna_id for item in kept], ["long_A"])
        self.assertEqual([item.record.rna_id for item in removals], ["short_A"])
        self.assertEqual(clusters[0]["representative"], "long_A")


if __name__ == "__main__":
    unittest.main()
