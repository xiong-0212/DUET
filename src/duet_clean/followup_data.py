from __future__ import annotations

import json
import random
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .identity import IdentityEngine
from .io import project_path


def _manifest_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _cluster_lookup(manifest: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        int(cluster["cluster_id"]): cluster
        for cluster in manifest["clusters"]
    }


def _positive_target(full_positive: int, full_count: int, target: int) -> int:
    return int(round(full_positive * target / full_count))


def select_cluster_subset(
    cluster_ids: Iterable[int],
    clusters: dict[int, dict[str, Any]],
    target_count: int,
    target_positive: int,
    k_best: int = 8,
) -> list[int]:
    """Dynamic-programming subset selection with exact chain count when possible."""

    ids = sorted(
        cluster_ids,
        key=lambda cluster_id: (
            -int(clusters[cluster_id]["cluster_size"]),
            cluster_id,
        ),
    )
    states: dict[int, list[tuple[int, tuple[int, ...]]]] = {0: [(0, ())]}
    for cluster_id in ids:
        size = int(clusters[cluster_id]["cluster_size"])
        positive = int(clusters[cluster_id]["positive_residues"])
        additions: dict[int, list[tuple[int, tuple[int, ...]]]] = {}
        for count, candidates in states.items():
            for old_positive, old_ids in candidates:
                new_count = count + size
                if new_count > target_count:
                    continue
                additions.setdefault(new_count, []).append(
                    (old_positive + positive, old_ids + (cluster_id,))
                )
        for count, candidates in additions.items():
            merged = states.get(count, []) + candidates
            merged.sort(
                key=lambda item: (
                    abs(
                        item[0]
                        - target_positive * count / max(target_count, 1)
                    ),
                    item[1],
                )
            )
            states[count] = merged[:k_best]
    if target_count in states:
        return list(states[target_count][0][1])
    nearest = min(states, key=lambda count: abs(count - target_count))
    return list(states[nearest][0][1])


def select_nested_cluster_subsets(
    cluster_ids: Iterable[int],
    clusters: dict[int, dict[str, Any]],
    first_target: int,
    second_target: int,
    first_positive_target: int,
    combined_positive_target: int,
    k_best: int = 12,
    first_max_cluster_size: int | None = None,
) -> tuple[list[int], list[int]]:
    """Choose disjoint subsets for the first and first+second size tiers."""

    ids = sorted(
        cluster_ids,
        key=lambda cluster_id: (
            -int(clusters[cluster_id]["cluster_size"]),
            cluster_id,
        ),
    )
    states: dict[tuple[int, int], list[tuple[int, int, tuple[int, ...], tuple[int, ...]]]] = {
        (0, 0): [(0, 0, (), ())]
    }
    for cluster_id in ids:
        size = int(clusters[cluster_id]["cluster_size"])
        positive = int(clusters[cluster_id]["positive_residues"])
        additions: dict[
            tuple[int, int],
            list[tuple[int, int, tuple[int, ...], tuple[int, ...]]],
        ] = {}
        for (first_size, second_size), candidates in states.items():
            for first_pos, second_pos, first_ids, second_ids in candidates:
                if (
                    first_size + size <= first_target
                    and (
                        first_max_cluster_size is None
                        or size <= first_max_cluster_size
                    )
                ):
                    additions.setdefault(
                        (first_size + size, second_size),
                        [],
                    ).append(
                        (
                            first_pos + positive,
                            second_pos,
                            first_ids + (cluster_id,),
                            second_ids,
                        )
                    )
                if second_size + size <= second_target:
                    additions.setdefault(
                        (first_size, second_size + size),
                        [],
                    ).append(
                        (
                            first_pos,
                            second_pos + positive,
                            first_ids,
                            second_ids + (cluster_id,),
                        )
                    )
        for state, candidates in additions.items():
            combined = states.get(state, []) + candidates
            combined.sort(
                key=lambda item: (
                    abs(
                        item[0]
                        - first_positive_target
                        * state[0]
                        / max(first_target, 1)
                    )
                    + abs(
                        item[0]
                        + item[1]
                        - combined_positive_target
                        * (state[0] + state[1])
                        / max(first_target + second_target, 1)
                    ),
                    item[2],
                    item[3],
                )
            )
            states[state] = combined[:k_best]
    target = (first_target, second_target)
    if target not in states:
        raise ValueError(
            f"no disjoint {first_target}+{second_target} subset for fold"
        )
    _, _, first_ids, second_ids = states[target][0]
    return list(first_ids), list(second_ids)


def _write_subset_manifest(
    base: dict[str, Any],
    name: str,
    selected_cluster_ids: set[int],
    fold_for_cluster: dict[int, int],
    purpose: str,
) -> dict[str, Any]:
    clusters = _cluster_lookup(base)
    pool_by_id = {
        record["rna_id"]: record
        for record in base["splits"]["pool"]
    }
    selected_clusters = []
    fold_validation_ids = {
        int(fold["fold"]): []
        for fold in base["folds"]
    }
    for cluster_id in sorted(selected_cluster_ids):
        cluster = clusters[cluster_id]
        fold = int(fold_for_cluster[cluster_id])
        selected_clusters.append(
            {
                **cluster,
                "fold": fold,
                "sampling_weight_per_record": 1.0
                / int(cluster["cluster_size"]),
            }
        )
        fold_validation_ids[fold].extend(cluster["rna_ids"])
    pool_ids = sorted(
        rna_id
        for cluster in selected_clusters
        for rna_id in cluster["rna_ids"]
    )
    folds = []
    for fold_index in sorted(fold_validation_ids):
        validation_ids = sorted(fold_validation_ids[fold_index])
        training_ids = sorted(set(pool_ids) - set(validation_ids))
        folds.append(
            {
                "fold": fold_index,
                "validation_ids": validation_ids,
                "training_ids": training_ids,
                "validation_cluster_ids": sorted(
                    cluster_id
                    for cluster_id in selected_cluster_ids
                    if fold_for_cluster[cluster_id] == fold_index
                ),
            }
        )
    payload = {
        "schema_version": 2,
        "name": name,
        "policy": {
            "base_manifest": base["name"],
            "purpose": purpose,
            "clustering": "whole connected components inherited from base",
            "sampling": "record weight = 1 / component size",
            "fold_assignment": "subset selected within existing homology folds",
        },
        "split_order": base["split_order"],
        "counts": {
            **{
                split: base["counts"][split]
                for split in base["split_order"]
                if split != "pool"
            },
            "pool": len(pool_ids),
        },
        "splits": {
            **{
                split: base["splits"][split]
                for split in base["split_order"]
                if split != "pool"
            },
            "pool": [pool_by_id[rna_id] for rna_id in pool_ids],
        },
        "clusters": selected_clusters,
        "folds": folds,
        "audit": {
            "base_manifest": base["name"],
            "base_manifest_sha256": base["manifest_sha256"],
            "selected_cluster_count": len(selected_clusters),
            "pool_records": len(pool_ids),
            "pool_unique_sequences": len(
                {pool_by_id[rna_id]["sequence"] for rna_id in pool_ids}
            ),
            "retained_exact_duplicate_records": len(pool_ids)
            - len({pool_by_id[rna_id]["sequence"] for rna_id in pool_ids}),
            "fold_sizes": [len(fold["validation_ids"]) for fold in folds],
            "fold_positive_residues": [
                sum(
                    pool_by_id[rna_id]["labels"].count("1")
                    for rna_id in fold["validation_ids"]
                )
                for fold in folds
            ],
        },
    }
    payload["manifest_sha256"] = _manifest_hash(payload)
    return payload


def build_size_curve_manifests(
    base: dict[str, Any],
    sizes: tuple[int, ...] = (80, 160),
) -> dict[int, dict[str, Any]]:
    clusters = _cluster_lookup(base)
    full_positive = int(
        sum(int(cluster["positive_residues"]) for cluster in clusters.values())
    )
    full_count = len(base["splits"]["pool"])
    if tuple(sizes) != (80, 160):
        raise ValueError("size-curve builder currently expects sizes (80, 160)")
    first_ids, second_ids = select_nested_cluster_subsets(
        list(clusters),
        clusters,
        first_target=80,
        second_target=80,
        first_positive_target=_positive_target(
            full_positive,
            full_count,
            80,
        ),
        combined_positive_target=_positive_target(
            full_positive,
            full_count,
            160,
        ),
        k_best=16,
        first_max_cluster_size=16,
    )
    selected_by_size = {
        80: set(first_ids),
        160: set(first_ids + second_ids),
    }
    output = {}
    for size, selected_ids in selected_by_size.items():
        fold_for_cluster = assign_clusters_balanced(
            [clusters[cluster_id] for cluster_id in selected_ids],
            seed=8000 + size,
        )
        manifest = _write_subset_manifest(
            base,
            f"followup_size_{size}_cv5",
            selected_ids,
            fold_for_cluster,
            f"fixed-composition training-size curve, nominal {size}",
        )
        output[size] = manifest
    return output


def _assignment_objective(
    loads: list[dict[str, int]],
    target_size: float,
    target_positive: float,
    size_weight: float = 8.0,
) -> float:
    return float(
        size_weight
        * np.mean(
            [
                ((fold["size"] - target_size) / target_size) ** 2
                for fold in loads
            ]
        )
        + np.mean(
            [
                ((fold["positive"] - target_positive) / target_positive) ** 2
                for fold in loads
            ]
        )
    )


def assign_clusters_balanced(
    clusters: list[dict[str, Any]],
    seed: int,
    n_folds: int = 5,
    iterations: int = 12000,
) -> dict[int, int]:
    rng = random.Random(seed)
    total_size = sum(int(cluster["cluster_size"]) for cluster in clusters)
    total_positive = sum(int(cluster["positive_residues"]) for cluster in clusters)
    target_size = total_size / n_folds
    target_positive = total_positive / n_folds
    ordered = sorted(
        clusters,
        key=lambda cluster: (
            -int(cluster["cluster_size"]),
            -int(cluster["positive_residues"]),
            rng.random(),
            int(cluster["cluster_id"]),
        ),
    )
    loads = [
        {"size": 0, "positive": 0}
        for _ in range(n_folds)
    ]
    assignment = {}
    for cluster in ordered:
        size = int(cluster["cluster_size"])
        positive = int(cluster["positive_residues"])
        best = min(
            range(n_folds),
            key=lambda fold: (
                (loads[fold]["size"] + size) / target_size
                + 0.65
                * (loads[fold]["positive"] + positive)
                / target_positive,
                loads[fold]["size"],
                fold,
            ),
        )
        assignment[int(cluster["cluster_id"])] = best
        loads[best]["size"] += size
        loads[best]["positive"] += positive

    by_id = {
        int(cluster["cluster_id"]): cluster
        for cluster in clusters
    }
    best_score = _assignment_objective(
        loads,
        target_size,
        target_positive,
    )
    ids = list(by_id)
    for _ in range(iterations):
        left, right = rng.sample(ids, 2)
        left_fold = assignment[left]
        right_fold = assignment[right]
        if left_fold == right_fold:
            continue
        left_size = int(by_id[left]["cluster_size"])
        right_size = int(by_id[right]["cluster_size"])
        left_pos = int(by_id[left]["positive_residues"])
        right_pos = int(by_id[right]["positive_residues"])
        candidate = [dict(load) for load in loads]
        candidate[left_fold]["size"] += right_size - left_size
        candidate[left_fold]["positive"] += right_pos - left_pos
        candidate[right_fold]["size"] += left_size - right_size
        candidate[right_fold]["positive"] += left_pos - right_pos
        score = _assignment_objective(
            candidate,
            target_size,
            target_positive,
        )
        if score + 1e-12 < best_score:
            assignment[left], assignment[right] = right_fold, left_fold
            loads = candidate
            best_score = score
    return assignment


def build_repeated_cv_manifests(
    base: dict[str, Any],
    seeds: tuple[int, ...] = (101, 102, 103),
) -> dict[int, dict[str, Any]]:
    output = {}
    for seed in seeds:
        assignment = assign_clusters_balanced(base["clusters"], seed)
        manifest = _write_subset_manifest(
            base,
            f"followup_cvrep_{seed}_cv5",
            set(assignment),
            assignment,
            f"repeated cluster-stratified assignment seed {seed}",
        )
        manifest["repeated_cv_seed"] = seed
        manifest["manifest_sha256"] = _manifest_hash(manifest)
        output[seed] = manifest
    return output


def build_exact_deduplicated_manifest(base: dict[str, Any]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in base["splits"]["pool"]:
        grouped.setdefault(record["sequence"], []).append(record)
    kept_ids = set()
    conflict_sequences = set()
    for sequence, records in grouped.items():
        labels = {record["labels"] for record in records}
        if len(labels) > 1:
            conflict_sequences.add(sequence)
            continue
        kept_ids.add(
            sorted(
                records,
                key=lambda record: (
                    record["split"],
                    int(record["ordinal"]),
                    record["rna_id"],
                ),
            )[0]["rna_id"]
        )
    pool_by_id = {
        record["rna_id"]: record
        for record in base["splits"]["pool"]
    }
    clusters = []
    fold_validation_ids = {fold: [] for fold in range(5)}
    for cluster in base["clusters"]:
        members = [
            rna_id
            for rna_id in cluster["rna_ids"]
            if rna_id in kept_ids
        ]
        if not members:
            continue
        updated = {
            **cluster,
            "rna_ids": members,
            "cluster_size": len(members),
            "positive_residues": sum(
                pool_by_id[rna_id]["labels"].count("1")
                for rna_id in members
            ),
            "residues": sum(
                len(pool_by_id[rna_id]["sequence"])
                for rna_id in members
            ),
            "sampling_weight_per_record": 1.0 / len(members),
        }
        clusters.append(updated)
        fold_validation_ids[int(cluster["fold"])].extend(members)
    pool_ids = sorted(kept_ids)
    folds = []
    for fold_index in range(5):
        validation_ids = sorted(fold_validation_ids[fold_index])
        folds.append(
            {
                "fold": fold_index,
                "validation_ids": validation_ids,
                "training_ids": sorted(set(pool_ids) - set(validation_ids)),
                "validation_cluster_ids": sorted(
                    int(cluster["cluster_id"])
                    for cluster in clusters
                    if int(cluster["fold"]) == fold_index
                ),
            }
        )
    payload = {
        "schema_version": 2,
        "name": "followup_dedup_290_cv5",
        "policy": {
            "base_manifest": base["name"],
            "purpose": "exact-duplicate removal for duplicate-weighting ablation",
            "exact_conflicts": "drop the entire conflicting sequence family",
            "fold_assignment": "inherited whole-cluster assignment",
        },
        "split_order": base["split_order"],
        "counts": {
            **{
                split: base["counts"][split]
                for split in base["split_order"]
                if split != "pool"
            },
            "pool": len(pool_ids),
        },
        "splits": {
            **{
                split: base["splits"][split]
                for split in base["split_order"]
                if split != "pool"
            },
            "pool": [pool_by_id[rna_id] for rna_id in pool_ids],
        },
        "clusters": clusters,
        "folds": folds,
        "audit": {
            "base_manifest": base["name"],
            "base_manifest_sha256": base["manifest_sha256"],
            "exact_duplicate_records_removed": len(base["splits"]["pool"])
            - len(pool_ids),
            "conflicting_sequence_families_removed": len(conflict_sequences),
            "pool_records": len(pool_ids),
            "pool_unique_sequences": len(pool_ids),
            "fold_sizes": [len(fold["validation_ids"]) for fold in folds],
            "fold_positive_residues": [
                sum(
                    pool_by_id[rna_id]["labels"].count("1")
                    for rna_id in fold["validation_ids"]
                )
                for fold in folds
            ],
        },
    }
    payload["manifest_sha256"] = _manifest_hash(payload)
    return payload


def validate_manifest(
    base: dict[str, Any],
    candidate: dict[str, Any],
    engine: IdentityEngine,
    threshold: float = 0.8,
) -> dict[str, Any]:
    errors = []
    pool = {
        record["rna_id"]: record
        for record in candidate["splits"]["pool"]
    }
    test_ids = {
        record["rna_id"]: record
        for split in ["TE18", "TL12", "JL10", "RB9"]
        for record in candidate["splits"][split]
    }
    for split in ["TE18", "TL12", "JL10", "RB9"]:
        if candidate["splits"][split] != base["splits"][split]:
            errors.append(f"{candidate['name']}: test split {split} changed")
    for fold in candidate["folds"]:
        validation = set(fold["validation_ids"])
        training = set(fold["training_ids"])
        if validation & training:
            errors.append(f"{candidate['name']}/fold{fold['fold']}: overlap")
        for left_id in sorted(validation):
            for right_id in sorted(training):
                result = engine.compare(
                    pool[left_id]["sequence"],
                    pool[right_id]["sequence"],
                )
                if (
                    result.comparable
                    and float(result.standard_identity) >= threshold
                ):
                    errors.append(
                        f"{candidate['name']}/fold{fold['fold']}: "
                        f"{left_id} vs {right_id} >= {threshold}"
                    )
    for pool_id in pool:
        for test_id in test_ids:
            result = engine.compare(pool[pool_id]["sequence"], test_ids[test_id]["sequence"])
            if result.comparable and float(result.standard_identity) >= threshold:
                errors.append(
                    f"{candidate['name']}: pool/test {pool_id} vs {test_id}"
                )
    return {
        "ok": not errors,
        "errors": errors[:100],
        "fold_sizes": [len(fold["validation_ids"]) for fold in candidate["folds"]],
        "fold_positive_residues": candidate["audit"]["fold_positive_residues"],
        "pool_records": len(pool),
    }


def write_manifest(manifest: dict[str, Any]) -> Path:
    path = project_path(
        f"data/clean/manifests/{manifest['name']}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
