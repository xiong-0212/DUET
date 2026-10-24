from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from duet_clean.dedup import (
    canonicalize_tests,
    collapse_exact_sequences,
    filter_homologs,
    remove_exact_overlap,
)
from duet_clean.identity import IdentityEngine
from duet_clean.io import load_config, load_raw_dataset, write_json
from duet_clean.pipeline import _manifest, _write_removal_ledger


NAME = "leakage_p80_keep_redundancy"


def main() -> int:
    config = load_config("configs/protocol.json")
    tests_order = list(config["test_order"])
    split_order = tests_order + ["val", "train"]
    data, _ = load_raw_dataset(
        config["data_dir"],
        tests_order,
        int(config["max_len"]),
    )
    engine = IdentityEngine.from_config(config)

    test_records, test_removals, _ = canonicalize_tests(
        [record for split in tests_order for record in data[split]],
        tests_order,
    )
    val_no_test, val_test_removals, _ = filter_homologs(
        data["val"],
        test_records,
        0.8,
        engine,
        "p80_test_deleak",
        "homology_to_test",
    )
    val_records, val_internal_removals, _ = collapse_exact_sequences(
        val_no_test,
        "validation_sequence_unique",
        conflict_policy="drop_all",
    )
    train_no_test, train_test_removals, _ = filter_homologs(
        data["train"],
        test_records,
        0.8,
        engine,
        "p80_test_deleak",
        "homology_to_test",
    )
    train_records, train_val_removals = remove_exact_overlap(
        train_no_test,
        val_records,
        "train_validation_exact_disjoint",
        "exact_sequence_overlap_with_validation",
    )

    records = test_records + val_records + train_records
    removals = (
        test_removals
        + val_test_removals
        + val_internal_removals
        + train_test_removals
        + train_val_removals
    )
    manifest = _manifest(
        NAME,
        policy={
            "identity": "global alignment matches / alignment columns",
            "min_length_ratio": 0.8,
            "test_duplicates": "exact-sequence canonicalisation",
            "train_test": "homology >= 0.8 removed",
            "validation_test": "homology >= 0.8 removed",
            "train_validation": "exact-sequence disjoint",
            "training_internal_deduplication": "none",
            "training_label_conflicts": "retained",
        },
        records=records,
        removals=removals,
        split_order=split_order,
    )
    write_json(f"data/clean/manifests/{NAME}.json", manifest)
    _write_removal_ledger(manifest)

    for split in tests_order + ["val", "train"]:
        print(f"{split}: {manifest['counts'][split]}")
    print(f"[written] data/clean/manifests/{NAME}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
