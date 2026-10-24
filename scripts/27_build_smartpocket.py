from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from duet_clean.identity import IdentityEngine
from duet_clean.io import load_config, project_path
from duet_clean.large_protocol import load_large_manifest


SOURCE_ROOT = ROOT / "data" / "external"
DEST_ROOT = ROOT / "data" / "external"
EMBEDDING_ROOT = Path(
    os.environ.get("DUET_EXTERNAL_EMBEDDINGS", str(SOURCE_ROOT))
)
OUTPUT = DEST_ROOT / "external_sets.json"
AUDIT = ROOT / "results" / "large_protocol" / "smartpocket_audit.json"


def relative_path(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def copy_if_distinct(source: Path, destination: Path) -> None:
    if source.resolve() == destination.resolve():
        return
    if not source.exists():
        raise FileNotFoundError(
            f"Missing external asset {source.name}. "
            "Set DUET_EXTERNAL_EMBEDDINGS to the directory containing "
            "emb_rnafm.pt and emb_rinalmo.pt."
        )
    shutil.copy2(source, destination)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def max_identity(sequence: str, references: list[str], engine: IdentityEngine):
    best = {"identity": 0.0, "reference": None}
    for reference in references:
        result = engine.compare(sequence, reference)
        if not result.comparable:
            continue
        identity = float(result.standard_identity)
        if identity > best["identity"]:
            best = {"identity": identity, "reference": reference}
    return best


def main() -> int:
    source_sets = json.loads(
        (SOURCE_ROOT / "external_sets.json").read_text(encoding="utf-8")
    )
    config = load_config("configs/protocol.json")
    engine = IdentityEngine.from_config(config)
    benchmark, manifest = load_large_manifest()
    pool = benchmark["pool"]
    test_sequences = [
        record.sequence
        for split in config["test_order"]
        for record in benchmark[split]
    ]
    pool_sequences = [record.sequence for record in pool]

    cleaned: dict[str, dict[str, Any]] = {}
    audit: dict[str, Any] = {
        "source": relative_path(SOURCE_ROOT / "external_sets.json"),
        "source_sha256": sha256(SOURCE_ROOT / "external_sets.json"),
        "large_manifest": manifest["name"],
        "large_manifest_sha256": manifest["manifest_sha256"],
        "reference_pool_records": len(pool),
        "reference_test_records": len(test_sequences),
        "splits": {},
    }
    removed: dict[str, Any] = {}
    seen_sequences: dict[str, str] = {}
    for name, records in source_sets.items():
        kept: dict[str, Any] = {}
        split_audit: dict[str, Any] = {}
        for chain_id, record in records.items():
            sequence = str(record["sequence"]).upper()
            labels = [int(value) for value in record["labels"]]
            if len(sequence) != len(labels):
                raise ValueError(f"{name}/{chain_id}: sequence/label length mismatch")
            to_pool = max_identity(sequence, pool_sequences, engine)
            to_tests = max_identity(sequence, test_sequences, engine)
            duplicate_of = seen_sequences.get(sequence)
            status = "retained"
            if name not in {"hariboss_test"} and to_pool["identity"] >= 0.8:
                status = "removed_pool_homology"
            elif name not in {"hariboss_test"} and to_tests["identity"] >= 0.8:
                status = "removed_canonical_test_homology"
            elif duplicate_of is not None:
                status = "removed_internal_duplicate"
            if status == "retained":
                kept[chain_id] = {
                    "sequence": sequence,
                    "labels": labels,
                }
                seen_sequences[sequence] = f"{name}/{chain_id}"
            else:
                removed[chain_id] = {
                    "split": name,
                    "reason": status,
                    "duplicate_of": duplicate_of,
                    "pool_identity": to_pool,
                    "test_identity": to_tests,
                }
            split_audit[chain_id] = {
                "status": status,
                "length": len(sequence),
                "positive_residues": sum(labels),
                "max_identity_to_large_pool": to_pool,
                "max_identity_to_canonical_tests": to_tests,
            }
        cleaned[name] = kept
        audit["splits"][name] = {
            "listed": len(records),
            "retained": len(kept),
            "removed": len(records) - len(kept),
            "residues": sum(len(record["sequence"]) for record in kept.values()),
            "positive_residues": sum(
                sum(record["labels"]) for record in kept.values()
            ),
            "records": split_audit,
        }

    DEST_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "source_release": "SMARTPocket",
        "source_file": relative_path(SOURCE_ROOT / "external_sets.json"),
        "source_sha256": sha256(SOURCE_ROOT / "external_sets.json"),
        "policy": {
            "identity": "global alignment matches / alignment columns",
            "min_length_ratio": 0.8,
            "pool_identity_removal": ">= 0.8",
            "canonical_test_identity_removal": ">= 0.8",
            "hariboss_test": "source-matched control; overlap is reported, not hidden",
        },
        "splits": cleaned,
        "removed": removed,
    }
    OUTPUT.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    copy_if_distinct(SOURCE_ROOT / "ss_external.tsv", DEST_ROOT / "ss_external.tsv")
    copy_if_distinct(
        SOURCE_ROOT / "fasta_cache.json",
        DEST_ROOT / "fasta_cache.json",
    )
    for name in ("emb_rnafm.pt", "emb_rinalmo.pt"):
        copy_if_distinct(EMBEDDING_ROOT / name, DEST_ROOT / name)
    print(
        json.dumps(
            {
                name: {
                    "retained": value["retained"],
                    "residues": value["residues"],
                }
                for name, value in audit["splits"].items()
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
