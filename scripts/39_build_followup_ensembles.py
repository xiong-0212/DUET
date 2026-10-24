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


SEEDS = [42, 0, 1, 2, 3, 7, 11, 13, 17, 23]
FOUR_SPLITS = ["TE18", "TL12", "JL10", "RB9"]
THREE_SPLITS = ["TE18", "TL12", "JL10"]
VARIANTS = {
    "duet_ensemble": {
        "members": ["duet_graph", "duet_tcn", "duet_dense"],
        "tests": FOUR_SPLITS,
    },
    "graph_seq3d_ensemble": {
        "members": ["graph_seq3d", "duet_tcn", "duet_dense"],
        "tests": THREE_SPLITS,
    },
}


def merge(member_payloads: list[dict[str, dict]], split: str) -> dict[str, dict]:
    references = [payload[split] for payload in member_payloads]
    merged = {}
    for rna_id in sorted(references[0]):
        labels = np.asarray(references[0][rna_id]["label"], dtype=int)
        probabilities = []
        for payload in references:
            value = payload[rna_id]
            if value["label"] != labels.tolist():
                raise ValueError(f"{split}/{rna_id}: label mismatch")
            probabilities.append(np.asarray(value["prob"], dtype=float))
        merged[rna_id] = {
            "label": labels.tolist(),
            "prob": np.mean(probabilities, axis=0).tolist(),
        }
    return merged


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--stage", choices=["cv", "final", "all"], default="all")
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANTS), default=sorted(VARIANTS))
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = parser.parse_args()
    for variant in args.variants:
        spec = VARIANTS[variant]
        members = spec["members"]
        tests = spec["tests"]
        if args.stage in {"cv", "all"}:
            folds = []
            for fold in range(5):
                runs = [
                    json.loads(
                        output_path(
                            "cv",
                            member,
                            42,
                            fold,
                            results_root=args.results_root,
                        ).read_text(encoding="utf-8")
                    )
                    for member in members
                ]
                result = {
                    "method": variant,
                    "seed": 42,
                    "fold": fold,
                    "selected_epoch": int(
                        np.median([run["selected_epoch"] for run in runs])
                    ),
                    "threshold": float(np.mean([run["threshold"] for run in runs])),
                    "per_chain": {
                        "val": merge([run["per_chain"] for run in runs], "val")
                    },
                }
                save_standard_result(
                    "cv",
                    result,
                    results_root=args.results_root,
                )
                folds.append(result)
            selected = select_cv_hyperparameters(folds)
            selected["method"] = variant
            selected["member_methods"] = members
            selected_path = (
                ROOT / args.results_root / "cv" / variant / "selected.json"
            )
            selected_path.parent.mkdir(parents=True, exist_ok=True)
            selected_path.write_text(
                json.dumps(selected, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(
                f"[selected] {variant}: threshold={selected['threshold']:.3f}"
            )
        if args.stage in {"final", "all"}:
            selected = json.loads(
                (ROOT / args.results_root / "cv" / variant / "selected.json").read_text(
                    encoding="utf-8"
                )
            )
            for seed in args.seeds:
                runs = [
                    json.loads(
                        output_path(
                            "final",
                            member,
                            seed,
                            results_root=args.results_root,
                        ).read_text(encoding="utf-8")
                    )
                    for member in members
                ]
                per_chain = {
                    split: merge([run["per_chain"] for run in runs], split)
                    for split in tests
                }
                result = standardize_result(
                    method=variant,
                    seed=seed,
                    threshold=float(selected["threshold"]),
                    selected_epoch=0,
                    per_chain=per_chain,
                    test_splits=tests,
                    oof_threshold=float(selected["threshold"]),
                )
                result["member_methods"] = members
                save_standard_result(
                    "final",
                    result,
                    results_root=args.results_root,
                )
                print(f"[final] {variant} seed={seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
