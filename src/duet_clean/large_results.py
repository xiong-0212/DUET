from __future__ import annotations

from pathlib import Path
from typing import Any

from .comparison import pooled_from_per_chain
from .io import project_path, write_json


TEST_SPLITS = ["TE18", "TL12", "JL10", "RB9"]


def standardize_result(
    method: str,
    seed: int,
    threshold: float,
    selected_epoch: int,
    per_chain: dict[str, dict[str, Any]],
    test_splits: list[str] | None = None,
    fold: int | None = None,
    fixed_epoch: int | None = None,
    validation_threshold_score: float | None = None,
    oof_threshold: float | None = None,
) -> dict[str, Any]:
    test_splits = test_splits or [
        split for split in TEST_SPLITS if split in per_chain
    ]
    metrics = {}
    metrics_05 = {}
    selected_chains = {}
    for split in test_splits:
        selected, chains = pooled_from_per_chain(per_chain[split], threshold)
        fixed, _ = pooled_from_per_chain(per_chain[split], 0.5)
        metrics[split] = selected
        metrics_05[split] = fixed
        selected_chains[split] = chains
    return {
        "method": method,
        "seed": int(seed),
        "fold": fold,
        "threshold": float(threshold),
        "threshold_source": (
            "outer CV OOF" if oof_threshold is not None
            else "fold validation"
            if fold is not None
            else "fixed CV selection"
        ),
        "oof_threshold": oof_threshold,
        "validation_threshold_score": validation_threshold_score,
        "selected_epoch": int(selected_epoch),
        "fixed_epoch": fixed_epoch,
        "test": metrics,
        "test_at_0.5": metrics_05,
        "per_chain": {
            **selected_chains,
            **(
                {"val": per_chain["val"]}
                if "val" in per_chain and fold is not None
                else {}
            ),
        },
    }


def output_path(
    stage: str,
    method: str,
    seed: int,
    fold: int | None = None,
    results_root: str | Path = "results/large_protocol",
) -> Path:
    base = project_path(results_root) / stage / method
    if fold is None:
        return base / f"seed{seed}.json"
    return base / f"seed{seed}_fold{fold}.json"


def save_standard_result(
    stage: str,
    result: dict[str, Any],
    results_root: str | Path = "results/large_protocol",
) -> Path:
    path = output_path(
        stage,
        str(result["method"]),
        int(result["seed"]),
        int(result["fold"]) if result.get("fold") is not None else None,
        results_root=results_root,
    )
    write_json(path, result)
    return path


def load_selected(
    method: str,
    results_root: str | Path = "results/large_protocol",
) -> dict[str, Any]:
    import json

    path = project_path(
        Path(results_root) / "cv" / method / "selected.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))
