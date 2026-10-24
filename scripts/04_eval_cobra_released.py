from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from duet_clean.cobra import CoBRAHead, attach_embeddings, evaluate
from duet_clean.experiments import (
    EXTERNAL_TESTS,
    load_rinalmo_embeddings,
    manifest_splits,
    old_exact_deleak_splits,
    paper_splits,
)
from duet_clean.io import project_path, write_json


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embeddings = load_rinalmo_embeddings("data/processed/rinalmo_embeddings.pt")
    model = CoBRAHead().to(device)
    checkpoint_path = project_path("data/processed/CoBRA_released.pt")
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()

    datasets = {
        "paper_released": paper_splits(),
        "old_exact_deleak": old_exact_deleak_splits(),
        "exact_sequence_unique": manifest_splits("exact_sequence_unique")[0],
    }
    output = {
        "checkpoint": str(checkpoint_path.relative_to(project_path("."))),
        "threshold": 0.5,
        "variants": {},
    }
    for name, raw_splits in datasets.items():
        embedded = attach_embeddings(raw_splits, embeddings)
        split_names = EXTERNAL_TESTS + (
            ["internal_test"] if "internal_test" in embedded else []
        )
        metrics, per_chain = evaluate(model, embedded, device, 0.5, split_names)
        external_mcc = [
            metrics[split]["MCC"] for split in EXTERNAL_TESTS if split in metrics
        ]
        output["variants"][name] = {
            "counts": {split: len(embedded.get(split, [])) for split in split_names},
            "metrics": metrics,
            "average_external_MCC": sum(external_mcc) / len(external_mcc),
            "per_chain": per_chain,
        }

    write_json("results/training/cobra_released_eval.json", output)
    print(json.dumps(
        {
            name: {
                "counts": {
                    split: len(data)
                    for split, data in attach_embeddings(splits, embeddings).items()
                },
                "average_external_MCC": output["variants"][name]["average_external_MCC"],
                "per_split_MCC": {
                    split: output["variants"][name]["metrics"][split]["MCC"]
                    for split in EXTERNAL_TESTS
                    if split in output["variants"][name]["metrics"]
                },
            }
            for name, splits in datasets.items()
        },
        indent=2,
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
