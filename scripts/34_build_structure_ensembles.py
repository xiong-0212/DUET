from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from duet_clean.large_results import output_path, save_standard_result, standardize_result
from duet_clean.large_training import select_cv_hyperparameters
from duet_clean.comparison import pooled_from_per_chain


SEEDS = [42, 0, 1, 2, 3, 7, 11, 13, 17, 23]
TESTS = ["TE18", "TL12", "JL10"]
EXTERNAL_TESTS = ["external_jiang", "time_dependent", "hariboss_test"]
VARIANTS = {
    "graph_seq3d_ensemble": ["graph_seq3d", "duet_tcn", "duet_dense"],
    "graph_seqss3d_ensemble": ["graph_seqss3d", "duet_tcn", "duet_dense"],
}


def merge(member_payloads: list[dict[str, dict]], split: str) -> dict[str, dict]:
    references = [payload[split] for payload in member_payloads]
    merged = {}
    for rna_id in sorted(references[0]):
        labels = np.asarray(references[0][rna_id]["label"], dtype=int)
        probabilities = []
        for reference in references:
            payload = reference[rna_id]
            if payload["label"] != labels.tolist():
                raise ValueError(f"{split}/{rna_id}: label mismatch")
            probabilities.append(np.asarray(payload["prob"], dtype=float))
        merged[rna_id] = {
            "label": labels.tolist(),
            "prob": np.mean(probabilities, axis=0).tolist(),
        }
    return merged


def main() -> int:
    for method, members in VARIANTS.items():
        folds = []
        for fold in range(5):
            runs = [
                json.loads(output_path("cv", member, 42, fold).read_text(encoding="utf-8"))
                for member in members
            ]
            result = {
                "method": method,
                "seed": 42,
                "fold": fold,
                "selected_epoch": int(np.median([run["selected_epoch"] for run in runs])),
                "threshold": float(np.mean([run["threshold"] for run in runs])),
                "per_chain": {"val": merge([run["per_chain"] for run in runs], "val")},
            }
            save_standard_result("cv", result)
            folds.append(result)
        selected = select_cv_hyperparameters(folds)
        selected["method"] = method
        selected["member_methods"] = members
        selected_path = ROOT / "results" / "large_protocol" / "cv" / method / "selected.json"
        selected_path.write_text(
            json.dumps(selected, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        for seed in SEEDS:
            runs = [
                json.loads(output_path("final", member, seed).read_text(encoding="utf-8"))
                for member in members
            ]
            per_chain = {
                split: merge([run["per_chain"] for run in runs], split)
                for split in TESTS
            }
            result = standardize_result(
                method=method,
                seed=seed,
                threshold=float(selected["threshold"]),
                selected_epoch=0,
                per_chain=per_chain,
                test_splits=TESTS,
                oof_threshold=float(selected["threshold"]),
            )
            result["member_methods"] = members
            save_standard_result("final", result)

            external_runs = [
                json.loads(
                    (
                        ROOT
                        / "results"
                        / "large_protocol"
                        / "external"
                        / f"{member}_seed{seed}.json"
                    ).read_text(encoding="utf-8")
                )
                for member in members
            ]
            external = {
                "method": method,
                "seed": seed,
                "threshold": float(selected["threshold"]),
                "per_chain": {
                    split: merge(
                        [run["per_chain"] for run in external_runs],
                        split,
                    )
                    for split in EXTERNAL_TESTS
                },
            }
            external["test"] = {
                split: pooled_from_per_chain(
                    external["per_chain"][split],
                    float(selected["threshold"]),
                )[0]
                for split in EXTERNAL_TESTS
            }
            external["test_at_0.5"] = {
                split: pooled_from_per_chain(
                    external["per_chain"][split],
                    0.5,
                )[0]
                for split in EXTERNAL_TESTS
            }
            external_path = (
                ROOT
                / "results"
                / "large_protocol"
                / "external"
                / f"{method}_seed{seed}.json"
            )
            external_path.write_text(
                json.dumps(external, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        print(f"[built] {method}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
