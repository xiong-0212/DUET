from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from typing import Any

from .dedup import canonicalize_tests, filter_homologs
from .identity import IdentityEngine
from .io import load_raw_dataset, project_path, write_json
from .models import Record, Removal


LARGE_MANIFEST = "large_p80_cv5"


def _manifest_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


class UnionFind:
    def __init__(self, values: list[int]) -> None:
        self.parent = {value: value for value in values}
        self.rank = {value: 0 for value in values}

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, first: int, second: int) -> None:
        left = self.find(first)
        right = self.find(second)
        if left == right:
            return
        if self.rank[left] < self.rank[right]:
            left, right = right, left
        self.parent[right] = left
        if self.rank[left] == self.rank[right]:
            self.rank[left] += 1


def connected_components(
    records: list[Record],
    threshold: float,
    engine: IdentityEngine,
) -> list[list[int]]:
    """Cluster at an identity threshold without allowing cross-fold leakage."""

    union = UnionFind(list(range(len(records))))
    for left in range(len(records)):
        for right in range(left + 1, len(records)):
            result = engine.compare(
                records[left].sequence,
                records[right].sequence,
            )
            if result.comparable and float(result.standard_identity) >= threshold:
                union.union(left, right)

    grouped: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        grouped[union.find(index)].append(index)
    clusters = list(grouped.values())
    clusters.sort(
        key=lambda members: (
            -len(members),
            min(records[index].rna_id for index in members),
        )
    )
    return clusters


def assign_clusters(
    records: list[Record],
    clusters: list[list[int]],
    n_folds: int,
) -> list[list[int]]:
    """Assign whole homology clusters greedily to balanced folds."""

    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if len(clusters) < n_folds:
        raise ValueError(
            f"only {len(clusters)} homology clusters for {n_folds} folds"
        )

    cluster_stats = []
    for cluster_id, members in enumerate(clusters):
        cluster_stats.append(
            {
                "cluster_id": cluster_id,
                "members": members,
                "size": len(members),
                "positive": sum(
                    records[index].labels.count("1") for index in members
                ),
                "residues": sum(len(records[index].sequence) for index in members),
            }
        )
    cluster_stats.sort(
        key=lambda item: (
            -item["size"],
            -item["positive"],
            -item["residues"],
            item["cluster_id"],
        )
    )

    folds: list[dict[str, Any]] = [
        {"records": [], "positive": 0, "residues": 0} for _ in range(n_folds)
    ]
    for cluster in cluster_stats:
        best_fold = min(
            range(n_folds),
            key=lambda fold: (
                len(folds[fold]["records"]) + cluster["size"],
                folds[fold]["positive"] + cluster["positive"],
                folds[fold]["residues"] + cluster["residues"],
                fold,
            ),
        )
        folds[best_fold]["records"].extend(cluster["members"])
        folds[best_fold]["positive"] += cluster["positive"]
        folds[best_fold]["residues"] += cluster["residues"]
        cluster["fold"] = best_fold

    return [
        sorted(fold["records"], key=lambda index: records[index].rna_id)
        for fold in folds
    ]


def build_large_protocol(
    config: dict[str, Any],
    name: str = LARGE_MANIFEST,
    threshold: float = 0.8,
    n_folds: int = 5,
) -> dict[str, Any]:
    test_order = list(config["test_order"])
    data, inventory = load_raw_dataset(
        config["data_dir"],
        test_order,
        int(config["max_len"]),
    )
    engine = IdentityEngine.from_config(config)

    test_records, test_removals, _ = canonicalize_tests(
        [record for split in test_order for record in data[split]],
        test_order,
    )
    pool_raw = data["train"] + data["val"]
    pool_records, test_removals_pool, best_to_test = filter_homologs(
        pool_raw,
        test_records,
        threshold,
        engine,
        stage=f"{name}_test_deleak",
        reason="homology_to_canonical_test",
    )
    pool_records = sorted(
        pool_records,
        key=lambda record: (record.split, record.ordinal, record.rna_id),
    )
    identifiers = [record.rna_id for record in pool_records]
    if len(identifiers) != len(set(identifiers)):
        duplicate_ids = [
            key for key, count in Counter(identifiers).items() if count > 1
        ]
        raise ValueError(f"pool rna_id values are not unique: {duplicate_ids[:20]}")

    clusters = connected_components(pool_records, threshold, engine)
    folds = assign_clusters(pool_records, clusters, n_folds)

    cluster_by_record = {}
    cluster_payload = []
    for cluster_id, members in enumerate(clusters):
        member_ids = sorted(pool_records[index].rna_id for index in members)
        cluster_size = len(members)
        positive = sum(pool_records[index].labels.count("1") for index in members)
        residues = sum(len(pool_records[index].sequence) for index in members)
        fold_id = next(
            fold_index
            for fold_index, fold in enumerate(folds)
            if any(index in fold for index in members)
        )
        cluster_payload.append(
            {
                "cluster_id": cluster_id,
                "rna_ids": member_ids,
                "cluster_size": cluster_size,
                "positive_residues": positive,
                "residues": residues,
                "fold": fold_id,
                "sampling_weight_per_record": 1.0 / cluster_size,
            }
        )
        for member in members:
            cluster_by_record[pool_records[member].rna_id] = cluster_id

    fold_payload = []
    for fold_index, members in enumerate(folds):
        fold_ids = sorted(pool_records[index].rna_id for index in members)
        train_ids = sorted(set(identifiers) - set(fold_ids))
        fold_payload.append(
            {
                "fold": fold_index,
                "validation_ids": fold_ids,
                "training_ids": train_ids,
                "validation_cluster_ids": sorted(
                    {
                        cluster_by_record[rna_id]
                        for rna_id in fold_ids
                    }
                ),
            }
        )

    cluster_sizes = [item["cluster_size"] for item in cluster_payload]
    unique_sequences = len({record.sequence for record in pool_records})
    removed_payload = [
        removal.to_dict() for removal in test_removals + test_removals_pool
    ]
    split_order = test_order + ["pool"]
    payload: dict[str, Any] = {
        "schema_version": 2,
        "name": name,
        "policy": {
            "identity": "global alignment matches / alignment columns",
            "min_length_ratio": config["identity"]["min_length_ratio"],
            "threshold": threshold,
            "pool": "raw train + validation; exact duplicates retained",
            "test_deleak": "pool records at or above threshold to canonical test removed",
            "clustering": "connected components of the task-level identity graph",
            "fold_assignment": "whole clusters, greedy chain-count balance",
            "sampling": "record weight = 1 / homology-cluster size",
            "cv_folds": n_folds,
        },
        "split_order": split_order,
        "counts": {
            **{
                split: sum(record.split == split for record in test_records)
                for split in test_order
            },
            "pool": len(pool_records),
        },
        "splits": {
            **{
                split: [
                    record.to_dict()
                    for record in test_records
                    if record.split == split
                ]
                for split in test_order
            },
            "pool": [record.to_dict() for record in pool_records],
        },
        "removed": removed_payload,
        "clusters": cluster_payload,
        "folds": fold_payload,
        "audit": {
            "raw_inventory": inventory,
            "raw_pool_records": len(pool_raw),
            "removed_pool_records": len(test_removals_pool),
            "retained_pool_records": len(pool_records),
            "retained_unique_sequences": unique_sequences,
            "retained_exact_duplicate_records": len(pool_records) - unique_sequences,
            "cluster_count": len(clusters),
            "singleton_clusters": sum(size == 1 for size in cluster_sizes),
            "largest_cluster": max(cluster_sizes),
            "fold_sizes": [len(fold["validation_ids"]) for fold in fold_payload],
            "fold_positive_residues": [
                sum(
                    pool_records[index].labels.count("1")
                    for index in members
                )
                for members in folds
            ],
            "best_identity_to_test": best_to_test,
        },
    }
    payload["manifest_sha256"] = _manifest_hash(payload)
    return payload


def load_large_manifest(
    name: str = LARGE_MANIFEST,
) -> tuple[dict[str, list[Record]], dict[str, Any]]:
    path = project_path(f"data/clean/manifests/{name}.json")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    splits = {
        split: [Record.from_dict(value) for value in values]
        for split, values in manifest["splits"].items()
    }
    return splits, manifest


def write_large_protocol(
    manifest: dict[str, Any],
    ledger_path: str | Path | None = None,
) -> Path:
    name = manifest["name"]
    output = project_path(f"data/clean/manifests/{name}.json")
    write_json(output, manifest)
    write_json(f"results/large_protocol/{name}_audit.json", manifest["audit"])

    if ledger_path is None:
        ledger_path = f"results/removal_ledger_{name}.csv"
    ledger = project_path(ledger_path)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("w", encoding="utf-8", newline="") as handle:
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

    clusters = project_path(f"results/large_protocol/{name}_clusters.csv")
    clusters.parent.mkdir(parents=True, exist_ok=True)
    with clusters.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "cluster_id",
                "fold",
                "cluster_size",
                "positive_residues",
                "residues",
                "sampling_weight_per_record",
                "rna_ids",
            ),
        )
        writer.writeheader()
        for row in manifest["clusters"]:
            writer.writerow({**row, "rna_ids": ";".join(row["rna_ids"])})
    return output


def fold_membership(manifest: dict[str, Any]) -> dict[int, dict[str, set[str]]]:
    return {
        int(fold["fold"]): {
            "train": set(fold["training_ids"]),
            "validation": set(fold["validation_ids"]),
        }
        for fold in manifest["folds"]
    }


def sampling_weights(manifest: dict[str, Any]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for cluster in manifest["clusters"]:
        weight = 1.0 / int(cluster["cluster_size"])
        for rna_id in cluster["rna_ids"]:
            weights[str(rna_id)] = weight
    return weights


def cluster_leakage_audit(
    manifest: dict[str, Any],
    threshold: float,
    engine: IdentityEngine,
) -> dict[str, Any]:
    splits, _ = load_large_manifest(manifest["name"])
    pool = {record.rna_id: record for record in splits["pool"]}
    violations = []
    for fold in manifest["folds"]:
        validation = set(fold["validation_ids"])
        training = set(fold["training_ids"])
        for left_id in sorted(validation):
            for right_id in sorted(training):
                result = engine.compare(
                    pool[left_id].sequence,
                    pool[right_id].sequence,
                )
                if (
                    result.comparable
                    and float(result.standard_identity) >= threshold
                ):
                    violations.append(
                        {
                            "fold": int(fold["fold"]),
                            "validation_id": left_id,
                            "training_id": right_id,
                            "identity": float(result.standard_identity),
                        }
                    )
    return {
        "threshold": threshold,
        "n_violations": len(violations),
        "violations": violations[:100],
    }
