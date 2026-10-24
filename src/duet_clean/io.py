from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from .models import Record


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: str | Path) -> dict[str, Any]:
    with project_path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def parse_bindingsite(path: Path) -> dict[str, dict[str, str]]:
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
        if line.strip()
    ]
    if len(lines) % 3:
        raise ValueError(f"{path}: expected groups of three non-empty lines")

    records: dict[str, dict[str, str]] = {}
    for index in range(0, len(lines), 3):
        header, sequence, labels = lines[index : index + 3]
        if not header.startswith(">"):
            raise ValueError(f"{path}: expected a FASTA-style header at line {index + 1}")
        rna_id = header[1:]
        if rna_id in records:
            raise ValueError(f"{path}: duplicate header {rna_id}")
        sequence = sequence.upper()
        if set(sequence) - set("ACGTUN"):
            raise ValueError(f"{path}: unexpected nucleotide code in {rna_id}")
        if len(sequence) != len(labels):
            raise ValueError(
                f"{path}: {rna_id} sequence length {len(sequence)} != label length {len(labels)}"
            )
        if set(labels) - set("01"):
            raise ValueError(f"{path}: {rna_id} labels must contain only 0 and 1")
        records[rna_id] = {"sequence": sequence, "labels": labels}
    return records


def read_split_ids(path: Path) -> list[str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"PDB_ID", "Chain"}
    missing = required - set(rows[0]) if rows else required
    if missing:
        raise ValueError(f"{path}: missing CSV columns {sorted(missing)}")
    return [f"{row['PDB_ID']}_{row['Chain']}" for row in rows]


def _record(
    split: str,
    source: str,
    rna_id: str,
    value: dict[str, str],
    ordinal: int,
) -> Record:
    pdb_id, _, chain = rna_id.partition("_")
    return Record(
        split=split,
        source=source,
        rna_id=rna_id,
        pdb_id=pdb_id,
        chain=chain,
        sequence=value["sequence"],
        labels=value["labels"],
        ordinal=ordinal,
    )


def load_raw_dataset(
    cobra_dir: str | Path,
    test_order: list[str],
    max_len: int = 161,
    include_internal_test: bool = False,
) -> tuple[dict[str, list[Record]], dict[str, Any]]:
    root = project_path(cobra_dir)
    hariboss = parse_bindingsite(root / "Hariboss_TR60" / "Hariboss_TR60-bindingsite.txt")
    dataset: dict[str, list[Record]] = {}
    audit: dict[str, Any] = {"max_len": max_len, "splits": {}, "missing_ids": {}}

    for split in ("train", "val"):
        ids = read_split_ids(root / "Hariboss_TR60" / f"{split}.csv")
        missing = [rna_id for rna_id in ids if rna_id not in hariboss]
        kept: list[Record] = []
        over_length: list[str] = []
        for ordinal, rna_id in enumerate(ids):
            if rna_id not in hariboss:
                continue
            value = hariboss[rna_id]
            if len(value["sequence"]) > max_len:
                over_length.append(rna_id)
                continue
            kept.append(_record(split, "HARIBOSS_TR60", rna_id, value, ordinal))
        dataset[split] = kept
        audit["missing_ids"][split] = missing
        audit["splits"][split] = {
            "listed_ids": len(ids),
            "kept": len(kept),
            "missing_from_bindingsite": len(missing),
            "over_max_len": len(over_length),
        }

    if include_internal_test:
        ids = read_split_ids(root / "Hariboss_TR60" / "test.csv")
        missing = [rna_id for rna_id in ids if rna_id not in hariboss]
        kept_internal: list[Record] = []
        over_length: list[str] = []
        for ordinal, rna_id in enumerate(ids):
            if rna_id not in hariboss:
                continue
            value = hariboss[rna_id]
            if len(value["sequence"]) > max_len:
                over_length.append(rna_id)
                continue
            kept_internal.append(
                _record("internal_test", "HARIBOSS_TR60", rna_id, value, ordinal)
            )
        dataset["internal_test"] = kept_internal
        audit["missing_ids"]["internal_test"] = missing
        audit["splits"]["internal_test"] = {
            "listed_ids": len(ids),
            "kept": len(kept_internal),
            "missing_from_bindingsite": len(missing),
            "over_max_len": len(over_length),
        }

    for split in test_order:
        values = parse_bindingsite(root / split / f"{split}-bindingsite.txt")
        kept = []
        over_length = []
        for ordinal, (rna_id, value) in enumerate(values.items()):
            if len(value["sequence"]) > max_len:
                over_length.append(rna_id)
                continue
            kept.append(_record(split, split, rna_id, value, ordinal))
        dataset[split] = kept
        audit["splits"][split] = {
            "listed_ids": len(values),
            "kept": len(kept),
            "missing_from_bindingsite": 0,
            "over_max_len": len(over_length),
        }

    return dataset, audit


def write_json(path: str | Path, value: Any) -> None:
    target = project_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            _json_safe(value),
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def load_manifest_records(name: str) -> tuple[dict[str, list[Record]], dict[str, Any]]:
    path = project_path(f"data/clean/manifests/{name}.json")
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    splits = {
        split: [Record.from_dict(value) for value in values]
        for split, values in manifest["splits"].items()
    }
    return splits, manifest
