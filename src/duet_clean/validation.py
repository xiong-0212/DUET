from __future__ import annotations

from itertools import combinations
from typing import Any

from .identity import IdentityEngine
from .io import load_raw_dataset, project_path, write_json
from .models import Record
from .pipeline import EXACT_MANIFEST


def load_manifest(name: str) -> dict[str, Any]:
    import json

    path = project_path(f"data/clean/manifests/{name}.json")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _records(manifest: dict[str, Any], split: str) -> list[Record]:
    return [Record.from_dict(value) for value in manifest["splits"][split]]


def _assert_unique_sequences(records: list[Record], label: str, errors: list[str]) -> None:
    seen: dict[str, str] = {}
    for record in records:
        previous = seen.get(record.sequence)
        if previous is not None:
            errors.append(f"{label}: duplicate sequence {previous} and {record.rna_id}")
        else:
            seen[record.sequence] = record.rna_id


def _cross_split_exact_violations(
    first: list[Record],
    second: list[Record],
    first_name: str,
    second_name: str,
    errors: list[str],
) -> None:
    first_sequences = {record.sequence for record in first}
    for record in second:
        if record.sequence in first_sequences:
            errors.append(f"exact overlap: {first_name} and {second_name}/{record.rna_id}")


def _cross_split_homology_violations(
    first: list[Record],
    second: list[Record],
    first_name: str,
    second_name: str,
    threshold: float,
    engine: IdentityEngine,
    errors: list[str],
) -> dict[str, Any]:
    best = {"identity": -1.0, "first": None, "second": None}
    for left in first:
        for right in second:
            result = engine.compare(left.sequence, right.sequence)
            if not result.comparable:
                continue
            if result.standard_identity > best["identity"]:
                best = {
                    "identity": float(result.standard_identity),
                    "first": f"{first_name}/{left.rna_id}",
                    "second": f"{second_name}/{right.rna_id}",
                }
            if result.standard_identity >= threshold:
                errors.append(
                    f"homology >= {threshold:.2f}: "
                    f"{first_name}/{left.rna_id} vs {second_name}/{right.rna_id}"
                )
    return best


def _within_split_homology_violations(
    records: list[Record],
    name: str,
    threshold: float,
    engine: IdentityEngine,
    errors: list[str],
) -> dict[str, Any]:
    best = {"identity": -1.0, "first": None, "second": None}
    for left, right in combinations(records, 2):
        result = engine.compare(left.sequence, right.sequence)
        if not result.comparable:
            continue
        if result.standard_identity > best["identity"]:
            best = {
                "identity": float(result.standard_identity),
                "first": f"{name}/{left.rna_id}",
                "second": f"{name}/{right.rna_id}",
            }
        if result.standard_identity >= threshold:
            errors.append(
                f"within-{name} homology >= {threshold:.2f}: {left.rna_id} vs {right.rna_id}"
            )
    return best


def validate_manifests(config: dict[str, Any]) -> dict[str, Any]:
    engine = IdentityEngine.from_config(config)
    test_order = list(config["test_order"])
    errors: list[str] = []
    report: dict[str, Any] = {"manifests": {}, "errors": errors}
    raw, _ = load_raw_dataset(
        config["data_dir"],
        test_order,
        int(config["max_len"]),
    )
    universe = {
        (record.split, record.rna_id): record
        for split in test_order + ["val", "train"]
        for record in raw[split]
    }

    names = [EXACT_MANIFEST] + [
        f"homology_p{int(round(float(threshold) * 100))}"
        for threshold in config["strict_thresholds"]
    ]
    leakage_name = "leakage_p80_keep_redundancy"
    if project_path(f"data/clean/manifests/{leakage_name}.json").exists():
        names.append(leakage_name)
    for name in names:
        manifest_errors_before = len(errors)
        manifest = load_manifest(name)
        tests = [record for split in test_order for record in _records(manifest, split)]
        train = _records(manifest, "train")
        val = _records(manifest, "val")

        _assert_unique_sequences(tests, f"{name}/tests", errors)
        if name != leakage_name:
            _assert_unique_sequences(train, f"{name}/train", errors)
        _assert_unique_sequences(val, f"{name}/val", errors)
        _cross_split_exact_violations(tests, train, "tests", "train", errors)
        _cross_split_exact_violations(tests, val, "tests", "val", errors)
        if name == EXACT_MANIFEST:
            _cross_split_exact_violations(val, train, "val", "train", errors)

        if name != EXACT_MANIFEST and name != leakage_name:
            threshold = float(manifest["policy"]["threshold"])
            train_test_max = _cross_split_homology_violations(
                tests, train, "tests", "train", threshold, engine, errors
            )
            val_test_max = _cross_split_homology_violations(
                tests, val, "tests", "val", threshold, engine, errors
            )
            train_val_max = _cross_split_homology_violations(
                val, train, "val", "train", threshold, engine, errors
            )
            train_within_max = _within_split_homology_violations(
                train, "train", threshold, engine, errors
            )
            val_within_max = _within_split_homology_violations(
                val, "val", threshold, engine, errors
            )
        elif name == leakage_name:
            threshold = 0.8
            train_test_max = _cross_split_homology_violations(
                tests, train, "tests", "train", threshold, engine, errors
            )
            val_test_max = _cross_split_homology_violations(
                tests, val, "tests", "val", threshold, engine, errors
            )
            train_val_max = None
            train_within_max = None
            val_within_max = None
        else:
            train_test_max = val_test_max = train_val_max = None
            train_within_max = val_within_max = None

        retained = {
            (record.split, record.rna_id)
            for record in tests + val + train
        }
        removed = {
            (row["split"], row["rna_id"])
            for row in manifest["removed"]
        }
        overlap = retained & removed
        if overlap:
            errors.append(f"{name}: records both retained and removed: {sorted(overlap)}")

        covered = retained | removed
        missing = set(universe) - covered
        extra = covered - set(universe)
        if missing:
            errors.append(f"{name}: input records missing from manifest: {sorted(missing)}")
        if extra:
            errors.append(f"{name}: manifest contains unknown records: {sorted(extra)}")

        retained_records = tests + val + train
        for record in retained_records:
            original = universe.get((record.split, record.rna_id))
            if original is None:
                continue
            if record.sequence != original.sequence or record.labels != original.labels:
                errors.append(
                    f"{name}: content changed for {record.split}/{record.rna_id}"
                )

        report["manifests"][name] = {
            "counts": manifest["counts"],
            "error_count": len(errors) - manifest_errors_before,
            "maximum_identity": {
                "train_vs_test": train_test_max,
                "validation_vs_test": val_test_max,
                "train_vs_validation": train_val_max,
                "within_train": train_within_max,
                "within_validation": val_within_max,
            },
            "checks": {
                "homology_control": name != EXACT_MANIFEST,
                "record_accounting_complete": not missing and not extra,
                "retained_removed_disjoint": not overlap,
            },
        }

    report["ok"] = not errors
    write_json("results/validation_report.json", report)
    if errors:
        raise ValueError("\n".join(errors[:50]))
    return report
