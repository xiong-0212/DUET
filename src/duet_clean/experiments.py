from __future__ import annotations

from pathlib import Path

import torch

from .io import load_manifest_records, load_raw_dataset, project_path
from .models import Record


EXTERNAL_TESTS = ["TE18", "TL12", "JL10", "RB9"]


def load_rinalmo_embeddings(path: str | Path) -> dict[str, torch.Tensor]:
    payload = torch.load(
        project_path(path),
        map_location="cpu",
        weights_only=False,
    )
    if isinstance(payload, dict) and "embeddings" in payload:
        payload = payload["embeddings"]
    if not isinstance(payload, dict):
        raise TypeError("RiNALMo embedding file must contain a dictionary")
    return payload


def paper_splits() -> dict[str, list[Record]]:
    raw, _ = load_raw_dataset(
        "data/raw/cobra",
        EXTERNAL_TESTS,
        max_len=161,
        include_internal_test=True,
    )
    return {
        "train": raw["train"],
        "val": raw["val"],
        "internal_test": raw["internal_test"],
        **{split: raw[split] for split in EXTERNAL_TESTS},
    }


def old_exact_deleak_splits() -> dict[str, list[Record]]:
    raw, _ = load_raw_dataset(
        "data/raw/cobra",
        EXTERNAL_TESTS,
        max_len=161,
    )
    drop_rb9 = {"1nem_A", "1q8n_A", "2tob_A"}
    return {
        "train": raw["train"],
        "val": raw["val"],
        **{
            split: [
                record
                for record in raw[split]
                if not (split == "RB9" and record.rna_id in drop_rb9)
            ]
            for split in EXTERNAL_TESTS
        },
    }


def manifest_splits(name: str) -> tuple[dict[str, list[Record]], dict]:
    return load_manifest_records(name)
