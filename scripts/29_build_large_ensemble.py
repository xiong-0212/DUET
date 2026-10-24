from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from duet_clean.large_results import (
    output_path,
    save_standard_result,
    standardize_result,
)
from duet_clean.large_training import select_cv_hyperparameters


METHOD = "duet_ensemble"
MEMBERS = ["duet_graph", "duet_tcn", "duet_dense"]
TEST_SPLITS = ["TE18", "TL12", "JL10", "RB9"]
SEEDS = [42, 0, 1, 2, 3, 7, 11, 13, 17, 23]


def merge(per_chain_members: list[dict[str, dict]], split: str) -> dict:
    references = [member[split] for member in per_chain_members]
    ids = sorted(references[0])
    merged = {}
    for rna_id in ids:
        labels = np.asarray(references[0][rna_id]["label"], dtype=int)
        probabilities = []
        for reference in references:
            payload = reference[rna_id]
            if labels.tolist() != payload["label"]:
                raise ValueError(f"{split}/{rna_id}: label mismatch")
            probabilities.append(np.asarray(payload["prob"], dtype=float))
        merged[rna_id] = {
            "label": labels.tolist(),
            "prob": np.mean(probabilities, axis=0).tolist(),
        }
    return merged


def build_cv() -> int:
    fold_results = []
    for fold in range(5):
        members = []
        for method in MEMBERS:
            path = output_path("cv", method, 42, fold)
            if not path.exists():
                print(f"[wait] missing {path.relative_to(ROOT)}")
                return 1
            members.append(json.loads(path.read_text(encoding="utf-8")))
        per_chain = {
            "val": merge(
                [member["per_chain"] for member in members],
                "val",
            )
        }
        result = {
            "method": METHOD,
            "seed": 42,
            "fold": fold,
            "selected_epoch": int(
                np.median([member["selected_epoch"] for member in members])
            ),
            "threshold": float(
                np.mean([member["threshold"] for member in members])
            ),
            "per_chain": per_chain,
            "member_methods": MEMBERS,
            "member_epochs": [member["selected_epoch"] for member in members],
        }
        save_standard_result("cv", result)
        fold_results.append(result)
    selected = select_cv_hyperparameters(fold_results)
    selected.update(
        {
            "method": METHOD,
            "cv_seed": 42,
            "folds": [0, 1, 2, 3, 4],
            "member_methods": MEMBERS,
        }
    )
    output = ROOT / "results" / "large_protocol" / "cv" / METHOD / "selected.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(selected, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[selected] {METHOD}: threshold={selected['threshold']:.3f} "
        f"OOF_MCC={selected['oof_metrics_at_selected_threshold']['MCC']:.4f}"
    )
    return 0


def build_final() -> int:
    selected_path = (
        ROOT / "results" / "large_protocol" / "cv" / METHOD / "selected.json"
    )
    if not selected_path.exists():
        print("[wait] run --stage cv first")
        return 1
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    threshold = float(selected["threshold"])
    for seed in SEEDS:
        members = []
        for method in MEMBERS:
            path = output_path("final", method, seed)
            if not path.exists():
                print(f"[wait] missing {path.relative_to(ROOT)}")
                return 1
            members.append(json.loads(path.read_text(encoding="utf-8")))
        per_chain = {
            split: merge([member["per_chain"] for member in members], split)
            for split in TEST_SPLITS
        }
        result = standardize_result(
            method=METHOD,
            seed=seed,
            threshold=threshold,
            selected_epoch=0,
            per_chain=per_chain,
            test_splits=TEST_SPLITS,
            oof_threshold=threshold,
        )
        result["member_methods"] = MEMBERS
        result["member_thresholds"] = [
            member["threshold"] for member in members
        ]
        save_standard_result("final", result)
        print(
            f"[final] {METHOD} seed={seed} "
            f"MCC={np.mean([result['test'][split]['MCC'] for split in TEST_SPLITS]):.4f}"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["cv", "final"], required=True)
    args = parser.parse_args()
    return build_cv() if args.stage == "cv" else build_final()


if __name__ == "__main__":
    raise SystemExit(main())
