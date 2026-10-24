from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any


def _sha256(text: str) -> str:
    return sha256(text.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class Record:
    split: str
    source: str
    rna_id: str
    pdb_id: str
    chain: str
    sequence: str
    labels: str
    ordinal: int

    @property
    def sequence_hash(self) -> str:
        return _sha256(self.sequence)

    @property
    def task_hash(self) -> str:
        return _sha256(f"{self.sequence}\n{self.labels}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "source": self.source,
            "rna_id": self.rna_id,
            "pdb_id": self.pdb_id,
            "chain": self.chain,
            "sequence": self.sequence,
            "labels": self.labels,
            "ordinal": self.ordinal,
            "sequence_sha256": self.sequence_hash,
            "task_sha256": self.task_hash,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Record":
        return cls(
            split=value["split"],
            source=value["source"],
            rna_id=value["rna_id"],
            pdb_id=value["pdb_id"],
            chain=value["chain"],
            sequence=value["sequence"],
            labels=value["labels"],
            ordinal=int(value["ordinal"]),
        )


@dataclass(frozen=True)
class Removal:
    record: Record
    stage: str
    reason: str
    reference_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.record.split,
            "rna_id": self.record.rna_id,
            "source": self.record.source,
            "sequence_sha256": self.record.sequence_hash,
            "stage": self.stage,
            "reason": self.reason,
            "reference_ids": list(self.reference_ids),
        }
