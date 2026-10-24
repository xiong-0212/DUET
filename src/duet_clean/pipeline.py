from __future__ import annotations

import csv
import json
import platform
import sys
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from typing import Any

from .dedup import (
    audit_sequence_groups,
    canonicalize_tests,
    cluster_homologs,
    collapse_exact_sequences,
    filter_homologs,
    remove_exact_overlap,
)
from .identity import IdentityEngine
from .io import load_raw_dataset, project_path, write_json
from .models import Record, Removal


EXACT_MANIFEST = "exact_sequence_unique"


def _sort_records(records: list[Record], split_order: list[str]) -> list[Record]:
    return sorted(
        records,
        key=lambda item: (split_order.index(item.split), item.ordinal, item.rna_id),
    )


def _split_payload(records: list[Record], split_order: list[str]) -> dict[str, list[dict]]:
    payload: dict[str, list[dict]] = {}
    for split in split_order:
        payload[split] = [record.to_dict() for record in records if record.split == split]
    return payload


def _manifest_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _manifest(
    name: str,
    policy: dict[str, Any],
    records: list[Record],
    removals: list[Removal],
    split_order: list[str],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    removed_by_split: dict[str, dict[str, int]] = {}
    for split in split_order:
        counts = Counter(
            removal.reason for removal in removals if removal.record.split == split
        )
        removed_by_split[split] = dict(sorted(counts.items()))

    payload: dict[str, Any] = {
        "schema_version": 1,
        "name": name,
        "policy": policy,
        "split_order": split_order,
        "counts": {
            split: sum(record.split == split for record in records) for split in split_order
        },
        "splits": _split_payload(records, split_order),
        "removed": [removal.to_dict() for removal in removals],
        "removed_counts": dict(sorted(Counter(removal.reason for removal in removals).items())),
        "removed_by_split": removed_by_split,
    }
    if extra:
        payload["audit"] = extra
    payload["manifest_sha256"] = _manifest_hash(payload)
    return payload


def _write_manifest(name: str, payload: dict[str, Any]) -> Path:
    path = project_path(f"data/clean/manifests/{name}.json")
    write_json(path, payload)
    return path


def _write_removal_ledger(manifest: dict[str, Any]) -> None:
    path = project_path(f"results/removal_ledger_{manifest['name']}.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "split",
                "rna_id",
                "source",
                "sequence_sha256",
                "stage",
                "reason",
                "reference_ids",
            ),
        )
        writer.writeheader()
        for row in manifest["removed"]:
            writer.writerow({**row, "reference_ids": ";".join(row["reference_ids"])})


def _write_sequence_group_csv(path: str, groups: list[Any]) -> None:
    target = project_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "sequence_sha256",
                "length",
                "label_conflict",
                "representative",
                "members",
            ),
        )
        writer.writeheader()
        for group in groups:
            writer.writerow(
                {
                    "sequence_sha256": group.sequence_hash,
                    "length": group.length,
                    "label_conflict": group.label_conflict,
                    "representative": group.representative.rna_id
                    if group.representative
                    else "",
                    "members": ";".join(
                        f"{record.split}/{record.rna_id}" for record in group.records
                    ),
                }
            )


def _counts(records: list[Record], split_order: list[str]) -> dict[str, int]:
    return {
        split: sum(record.split == split for record in records)
        for split in split_order
    }


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifests(config: dict[str, Any]) -> dict[str, Any]:
    split_order = list(config["test_order"]) + ["val", "train"]
    dataset, inventory = load_raw_dataset(
        config["data_dir"],
        list(config["test_order"]),
        int(config["max_len"]),
    )
    engine = IdentityEngine.from_config(config)
    conflict_policy = config["policy"]["sequence_label_conflict"]

    raw_records = [record for split in split_order for record in dataset[split]]
    raw_sequence_groups = audit_sequence_groups(raw_records)
    raw_duplicate_groups = [group for group in raw_sequence_groups if group.is_duplicate]

    test_raw = [record for split in config["test_order"] for record in dataset[split]]
    test_records, test_removals, test_groups = canonicalize_tests(
        test_raw, list(config["test_order"])
    )

    val_no_test, val_test_removals = remove_exact_overlap(
        dataset["val"],
        test_records,
        stage="exact_deleak",
        reason="exact_sequence_overlap_with_test",
    )
    val_records, val_internal_removals, val_groups = collapse_exact_sequences(
        val_no_test,
        stage="exact_sequence_unique",
        conflict_policy=conflict_policy,
    )

    train_no_test, train_test_removals = remove_exact_overlap(
        dataset["train"],
        test_records,
        stage="exact_deleak",
        reason="exact_sequence_overlap_with_test",
    )
    train_no_val, train_val_removals = remove_exact_overlap(
        train_no_test,
        val_records,
        stage="exact_train_val_disjoint",
        reason="exact_sequence_overlap_with_validation",
    )
    train_records, train_internal_removals, train_groups = collapse_exact_sequences(
        train_no_val,
        stage="exact_sequence_unique",
        conflict_policy=conflict_policy,
    )

    exact_removals = (
        test_removals
        + val_test_removals
        + val_internal_removals
        + train_test_removals
        + train_val_removals
        + train_internal_removals
    )
    exact_records = _sort_records(test_records + val_records + train_records, split_order)
    exact_manifest = _manifest(
        EXACT_MANIFEST,
        policy={
            "identity": "exact sequence",
            "test_representative": "benchmark priority, then rna_id",
            "train_validation": "exact-sequence disjoint",
            "training_redundancy": "one unique sequence; conflicting labels removed",
            "sequence_label_conflict": conflict_policy,
        },
        records=exact_records,
        removals=exact_removals,
        split_order=split_order,
        extra={
            "test_sequence_groups": [
                group.to_dict() for group in test_groups if group.is_duplicate
            ],
            "validation_sequence_groups": [
                group.to_dict() for group in val_groups if group.is_duplicate
            ],
            "training_sequence_groups": [
                group.to_dict() for group in train_groups if group.is_duplicate
            ],
        },
    )
    _write_manifest(EXACT_MANIFEST, exact_manifest)
    _write_removal_ledger(exact_manifest)

    manifests: dict[str, dict[str, Any]] = {EXACT_MANIFEST: exact_manifest}
    threshold_audits: list[dict[str, Any]] = []
    for threshold in config["strict_thresholds"]:
        threshold = float(threshold)
        name = f"homology_p{int(round(threshold * 100))}"

        val_no_test_h, val_test_h_removals, val_test_h_best = filter_homologs(
            val_records,
            test_records,
            threshold,
            engine,
            stage=f"{name}_test_deleak",
            reason="homology_to_test",
        )
        val_h, val_cluster_removals, val_clusters = cluster_homologs(
            val_no_test_h,
            threshold,
            engine,
            stage=f"{name}_validation_cluster",
        )

        train_no_test_h, train_test_h_removals, train_test_h_best = filter_homologs(
            train_records,
            test_records,
            threshold,
            engine,
            stage=f"{name}_test_deleak",
            reason="homology_to_test",
        )
        train_no_val_h, train_val_h_removals, train_val_h_best = filter_homologs(
            train_no_test_h,
            val_h,
            threshold,
            engine,
            stage=f"{name}_train_val_disjoint",
            reason="homology_to_validation",
        )
        train_h, train_cluster_removals, train_clusters = cluster_homologs(
            train_no_val_h,
            threshold,
            engine,
            stage=f"{name}_training_cluster",
        )

        homology_removals = (
            val_test_h_removals
            + val_cluster_removals
            + train_test_h_removals
            + train_val_h_removals
            + train_cluster_removals
        )
        records = _sort_records(test_records + val_h + train_h, split_order)
        manifest = _manifest(
            name,
            policy={
                "identity": "global alignment matches / alignment columns",
                "min_length_ratio": config["identity"]["min_length_ratio"],
                "threshold": threshold,
                "split_order_for_removal": ["test", "validation", "training"],
                "test_representative": "exact-sequence benchmark priority, then rna_id",
                "validation_redundancy": "greedy representative clustering",
                "training_redundancy": "greedy representative clustering",
            },
            records=records,
            removals=exact_removals + homology_removals,
            split_order=split_order,
        )
        _write_manifest(name, manifest)
        _write_removal_ledger(manifest)
        manifests[name] = manifest
        threshold_audits.append(
            {
                "manifest": name,
                "threshold": threshold,
                "counts": _counts(records, split_order),
                "removed_counts": manifest["removed_counts"],
                "validation_max_identity_to_test": val_test_h_best,
                "train_max_identity_to_test": train_test_h_best,
                "train_max_identity_to_validation": train_val_h_best,
                "validation_clusters": val_clusters,
                "training_clusters": train_clusters,
            }
        )

    summary = {
        "raw_inventory": inventory,
        "raw_exact_sequence_audit": {
            "groups": len(raw_duplicate_groups),
            "extra_records": sum(len(group.records) - 1 for group in raw_duplicate_groups),
            "label_conflict_groups": sum(
                group.label_conflict for group in raw_duplicate_groups
            ),
            "label_conflict_records": sum(
                len(group.records) for group in raw_duplicate_groups if group.label_conflict
            ),
        },
        "post_filter_exact_group_audit": {
            "test_duplicate_groups": sum(group.is_duplicate for group in test_groups),
            "validation_candidate_groups": sum(group.is_duplicate for group in val_groups),
            "training_candidate_groups": sum(group.is_duplicate for group in train_groups),
        },
        "canonical_test_counts": _counts(test_records, split_order),
        "exact_counts": exact_manifest["counts"],
        "exact_removed_counts": exact_manifest["removed_counts"],
        "strict": threshold_audits,
        "policy": {
            "test_priority": config["test_order"],
            "test_duplicates": "keep first benchmark by priority, then rna_id",
            "sequence_label_conflict": conflict_policy,
            "identity": {
                "selection": "matches / alignment columns",
                "diagnostic": "matches / min(sequence lengths)",
                "min_length_ratio": config["identity"]["min_length_ratio"],
                "alignment": config["alignment"],
            },
            "homology_cluster": "longest-first, deterministic greedy representatives",
        },
        "manifests": {
            name: {
                "counts": manifest["counts"],
                "removed_counts": manifest["removed_counts"],
                "sha256": manifest["manifest_sha256"],
            }
            for name, manifest in manifests.items()
        },
    }
    raw_root = project_path(config["data_dir"])
    raw_hashes = {
        str(path.relative_to(raw_root)).replace("\\", "/"): _file_sha256(path)
        for path in sorted(raw_root.rglob("*"))
        if path.is_file()
    }
    try:
        import Bio

        biopython_version = Bio.__version__
    except ImportError:
        biopython_version = None
    environment = {
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "biopython": biopython_version,
    }
    write_json("results/raw_inventory.json", inventory)
    write_json("results/raw_input_sha256.json", raw_hashes)
    write_json("results/environment.json", environment)
    write_json("results/dedup_summary.json", summary)
    write_json(
        "results/exact_duplicate_groups.json",
        {
            "summary": {
                "raw_groups": len(raw_duplicate_groups),
                "raw_extra_records": sum(
                    len(group.records) - 1 for group in raw_duplicate_groups
                ),
            },
            "raw": [group.to_dict() for group in raw_duplicate_groups],
            "test": [group.to_dict() for group in test_groups if group.is_duplicate],
            "validation": [
                group.to_dict() for group in val_groups if group.is_duplicate
            ],
            "training_candidates": [
                group.to_dict() for group in train_groups if group.is_duplicate
            ],
        },
    )
    _write_sequence_group_csv("results/raw_exact_duplicate_groups.csv", raw_duplicate_groups)
    _write_sequence_group_csv(
        "results/conflicting_sequence_families.csv",
        [group for group in raw_duplicate_groups if group.label_conflict],
    )
    _write_audit_report(summary)
    return summary


def _write_audit_report(summary: dict[str, Any]) -> None:
    lines = [
        "# Sequence de-duplication audit",
        "",
        "This report is generated by `scripts/01_build_dedup.py`.",
        "",
        "## Raw input",
        "",
        "| Split | Listed | Kept |",
        "|:--|--:|--:|",
    ]
    for split in ("train", "val", "TE18", "TL12", "JL10", "RB9"):
        row = summary["raw_inventory"]["splits"][split]
        lines.append(f"| {split} | {row['listed_ids']} | {row['kept']} |")
    raw = summary["raw_exact_sequence_audit"]
    post = summary["post_filter_exact_group_audit"]
    lines += [
        "",
        f"Raw exact-sequence duplicate groups: **{raw['groups']}** "
        f"({raw['extra_records']} redundant records).",
        f"Label-conflicting groups: **{raw['label_conflict_groups']}** "
        f"({raw['label_conflict_records']} records).",
        "",
        "After canonicalising test sets and removing train/test and train/validation "
        f"overlap, {post['training_candidate_groups']} duplicate groups remained in "
        f"the training candidate pool and {post['validation_candidate_groups']} "
        "remained in validation.",
        "",
        "Detailed duplicate groups are in `results/exact_duplicate_groups.json`; "
        "flat tables are in `results/raw_exact_duplicate_groups.csv` and "
        "`results/conflicting_sequence_families.csv`.",
        "",
        "## Manifests",
        "",
        "| Manifest | TE18 | TL12 | JL10 | RB9 | Val | Train |",
        "|:--|--:|--:|--:|--:|--:|--:|",
    ]
    for name, item in summary["manifests"].items():
        counts = item["counts"]
        lines.append(
            f"| {name} | {counts['TE18']} | {counts['TL12']} | {counts['JL10']} | "
            f"{counts['RB9']} | {counts['val']} | {counts['train']} |"
        )
    lines += [
        "",
        "The exact-clean manifest is the reference split. Stricter manifests "
        "remove homologous train/validation records and then cluster each "
        "remaining split by global sequence identity.",
        "",
        "## Validation",
        "",
        "Every manifest must pass `scripts/02_validate.py`. The validator checks "
        "exact cross-split disjointness, thresholded homology disjointness, "
        "within-split homology clustering, unique sequences, and retained/removed "
        "record accounting.",
        "",
    ]
    path = project_path("results/AUDIT_REPORT.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
