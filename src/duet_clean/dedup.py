from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from .identity import IdentityEngine, IdentityResult
from .models import Record, Removal


@dataclass(frozen=True)
class Match:
    reference: Record
    identity: IdentityResult


@dataclass(frozen=True)
class SequenceGroup:
    sequence_hash: str
    length: int
    records: tuple[Record, ...]
    label_conflict: bool
    representative: Record | None

    @property
    def is_duplicate(self) -> bool:
        return len(self.records) > 1

    def to_dict(self) -> dict:
        return {
            "sequence_sha256": self.sequence_hash,
            "length": self.length,
            "records": [
                {
                    "split": record.split,
                    "rna_id": record.rna_id,
                    "task_sha256": record.task_hash,
                }
                for record in self.records
            ],
            "label_conflict": self.label_conflict,
            "representative": self.representative.rna_id if self.representative else None,
        }


def record_sort_key(record: Record, split_order: list[str]) -> tuple[int, str]:
    return split_order.index(record.split), record.rna_id


def group_by_sequence(records: Iterable[Record]) -> dict[str, list[Record]]:
    groups: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        groups[record.sequence].append(record)
    return dict(groups)


def audit_sequence_groups(records: Iterable[Record]) -> list[SequenceGroup]:
    groups = group_by_sequence(records)
    audit: list[SequenceGroup] = []
    for sequence in sorted(groups, key=lambda value: (len(value), value)):
        members = tuple(sorted(groups[sequence], key=lambda item: (item.split, item.rna_id)))
        audit.append(
            SequenceGroup(
                sequence_hash=members[0].sequence_hash,
                length=len(sequence),
                records=members,
                label_conflict=len({record.labels for record in members}) > 1,
                representative=members[0],
            )
        )
    return audit


def canonicalize_tests(
    records: list[Record],
    split_order: list[str],
) -> tuple[list[Record], list[Removal], list[SequenceGroup]]:
    """Keep one record per sequence across test sets.

    The policy is deterministic: earlier benchmark priority, then rna_id.
    """

    kept: list[Record] = []
    removals: list[Removal] = []
    groups = group_by_sequence(records)
    group_audit: list[SequenceGroup] = []
    for sequence in sorted(groups, key=lambda value: (len(value), value)):
        members = sorted(groups[sequence], key=lambda item: record_sort_key(item, split_order))
        representative = members[0]
        kept.append(representative)
        label_conflict = len({record.labels for record in members}) > 1
        group_audit.append(
            SequenceGroup(
                sequence_hash=representative.sequence_hash,
                length=len(sequence),
                records=tuple(members),
                label_conflict=label_conflict,
                representative=representative,
            )
        )
        for duplicate in members[1:]:
            removals.append(
                Removal(
                    record=duplicate,
                    stage="test_sequence_unique",
                    reason="exact_sequence_duplicate_across_test_sets",
                    reference_ids=(representative.rna_id,),
                )
            )
    kept.sort(key=lambda item: (split_order.index(item.split), item.ordinal))
    return kept, removals, group_audit


def remove_exact_overlap(
    records: Iterable[Record],
    references: Iterable[Record],
    stage: str,
    reason: str,
) -> tuple[list[Record], list[Removal]]:
    reference_map: dict[str, list[str]] = defaultdict(list)
    for record in references:
        reference_map[record.sequence].append(record.rna_id)

    kept: list[Record] = []
    removals: list[Removal] = []
    for record in records:
        matches = reference_map.get(record.sequence, [])
        if matches:
            removals.append(
                Removal(
                    record=record,
                    stage=stage,
                    reason=reason,
                    reference_ids=tuple(sorted(matches)),
                )
            )
        else:
            kept.append(record)
    return kept, removals


def collapse_exact_sequences(
    records: Iterable[Record],
    stage: str,
    split_priority: list[str] | None = None,
    conflict_policy: str = "drop_all",
) -> tuple[list[Record], list[Removal], list[SequenceGroup]]:
    """Collapse exact sequences to one deterministic representative.

    Same-sequence records with different labels are an irreducible target
    conflict for a sequence-only model. The default drops the whole family
    instead of choosing an arbitrary ligand context.
    """

    split_priority = split_priority or []
    kept: list[Record] = []
    removals: list[Removal] = []
    group_audit: list[SequenceGroup] = []
    groups = group_by_sequence(records)

    for sequence in sorted(groups, key=lambda value: (len(value), value)):
        members = sorted(
            groups[sequence],
            key=lambda item: (
                split_priority.index(item.split) if item.split in split_priority else 0,
                item.rna_id,
            ),
        )
        label_conflict = len({member.labels for member in members}) > 1
        representative: Record | None = None
        if label_conflict:
            if conflict_policy == "keep_first":
                representative = members[0]
                kept.append(representative)
                for duplicate in members[1:]:
                    removals.append(
                        Removal(
                            record=duplicate,
                            stage=stage,
                            reason="ambiguous_same_sequence_different_labels",
                            reference_ids=(representative.rna_id,),
                        )
                    )
            elif conflict_policy == "drop_all":
                for member in members:
                    removals.append(
                        Removal(
                            record=member,
                            stage=stage,
                            reason="ambiguous_same_sequence_different_labels",
                            reference_ids=tuple(
                                other.rna_id for other in members if other.rna_id != member.rna_id
                            ),
                        )
                    )
            else:
                raise ValueError(f"unknown conflict_policy: {conflict_policy}")
        else:
            representative = members[0]
            kept.append(representative)
            for duplicate in members[1:]:
                removals.append(
                    Removal(
                        record=duplicate,
                        stage=stage,
                        reason="exact_sequence_and_label_duplicate",
                        reference_ids=(representative.rna_id,),
                    )
                )

        group_audit.append(
            SequenceGroup(
                sequence_hash=members[0].sequence_hash,
                length=len(sequence),
                records=tuple(members),
                label_conflict=label_conflict,
                representative=representative,
            )
        )

    kept.sort(key=lambda item: (item.split, item.ordinal, item.rna_id))
    return kept, removals, group_audit


def best_match(
    record: Record,
    references: Iterable[Record],
    engine: IdentityEngine,
) -> Match | None:
    best: Match | None = None
    for reference in references:
        result = engine.compare(record.sequence, reference.sequence)
        if not result.comparable:
            continue
        if best is None or result.standard_identity > best.identity.standard_identity:
            best = Match(reference=reference, identity=result)
    return best


def filter_homologs(
    records: Iterable[Record],
    references: list[Record],
    threshold: float,
    engine: IdentityEngine,
    stage: str,
    reason: str,
) -> tuple[list[Record], list[Removal], dict[str, dict]]:
    kept: list[Record] = []
    removals: list[Removal] = []
    best_by_record: dict[str, dict] = {}
    for record in records:
        match = best_match(record, references, engine)
        if match is None:
            best_by_record[record.rna_id] = {"identity": None, "to": None}
            kept.append(record)
            continue
        best_by_record[record.rna_id] = {
            "identity": round(float(match.identity.standard_identity), 6),
            "to": match.reference.rna_id,
            "alignment_columns": match.identity.alignment_columns,
            "matches": match.identity.matches,
        }
        if match.identity.standard_identity >= threshold:
            removals.append(
                Removal(
                    record=record,
                    stage=stage,
                    reason=reason,
                    reference_ids=(match.reference.rna_id,),
                )
            )
        else:
            kept.append(record)
    return kept, removals, best_by_record


def cluster_homologs(
    records: Iterable[Record],
    threshold: float,
    engine: IdentityEngine,
    stage: str,
) -> tuple[list[Record], list[Removal], list[dict]]:
    """Greedy deterministic clustering.

    Longer sequences are representatives first, then rna_id. A record joins an
    existing cluster when its identity to the representative reaches threshold.
    This avoids connected-component chaining.
    """

    ordered = sorted(records, key=lambda item: (-len(item.sequence), item.rna_id))
    representatives: list[Record] = []
    removals: list[Removal] = []
    clusters: list[dict] = []

    for record in ordered:
        assigned: tuple[Record, IdentityResult] | None = None
        for representative in representatives:
            result = engine.compare(record.sequence, representative.sequence)
            if result.comparable and result.standard_identity >= threshold:
                assigned = representative, result
                break
        if assigned is None:
            representatives.append(record)
            clusters.append(
                {
                    "representative": record.rna_id,
                    "length": len(record.sequence),
                    "members": [record.rna_id],
                }
            )
            continue

        representative, result = assigned
        removals.append(
            Removal(
                record=record,
                stage=stage,
                reason="homology_cluster_duplicate",
                reference_ids=(representative.rna_id,),
            )
        )
        for cluster in clusters:
            if cluster["representative"] == representative.rna_id:
                cluster["members"].append(record.rna_id)
                break

    representatives.sort(key=lambda item: (item.split, item.ordinal, item.rna_id))
    return representatives, removals, clusters
