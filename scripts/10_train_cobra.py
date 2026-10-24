from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from duet_clean.cobra import attach_embeddings, train_one
from duet_clean.experiments import (
    EXTERNAL_TESTS,
    load_rinalmo_embeddings,
    manifest_splits,
    paper_splits,
)
from duet_clean.io import project_path, write_json


PROTOCOLS = {
    "paper": {
        "seeds": [42],
        "epochs": 100,
        "batch_size": 4,
        "learning_rate": 5e-4,
        "weight_decay": 1e-4,
        "patience": None,
        "test_splits": ["internal_test"] + EXTERNAL_TESTS,
        "description": (
            "CoBRA paper configuration: two-logit cross-entropy compatible with "
            "the official softmax head, batch 4, 100 epochs, seed 42, fixed 0.5 threshold"
        ),
    },
    "controlled": {
        "seeds": [42, 0, 1, 2, 3, 7, 11, 13, 17, 23],
        "epochs": 120,
        "batch_size": 8,
        "learning_rate": 5e-4,
        "weight_decay": 1e-4,
        "patience": 15,
        "test_splits": EXTERNAL_TESTS,
        "description": (
            "CoBRA-M3 architecture and two-logit cross-entropy under the DUET "
            "controlled protocol"
        ),
    },
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["paper", "exact_clean", "homology_p80", "all"],
        default="all",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embeddings = load_rinalmo_embeddings("data/processed/rinalmo_embeddings.pt")

    jobs = []
    if args.mode in ("paper", "all"):
        jobs.append(("cobra_paper", paper_splits(), "paper"))
    if args.mode in ("exact_clean", "all"):
        jobs.append(
            (
                "cobra_exact_clean",
                manifest_splits("exact_sequence_unique")[0],
                "controlled",
            )
        )
    if args.mode in ("homology_p80", "all"):
        jobs.append(
            (
                "cobra_homology_p80",
                manifest_splits("homology_p80")[0],
                "controlled",
            )
        )

    summaries = {}
    for tag, raw_splits, protocol_name in jobs:
        protocol = PROTOCOLS[protocol_name]
        data = attach_embeddings(raw_splits, embeddings)
        seeds = args.seeds or protocol["seeds"]
        epochs = args.epochs or protocol["epochs"]
        runs = []
        print(
            f"== {tag}: protocol={protocol_name} train={len(data['train'])} "
            f"val={len(data['val'])} seeds={seeds} ==",
            flush=True,
        )
        for seed in seeds:
            model, result = train_one(
                data=data,
                test_splits=protocol["test_splits"],
                seed=seed,
                protocol=protocol_name,
                device=device,
                epochs=epochs,
                batch_size=protocol["batch_size"],
                learning_rate=protocol["learning_rate"],
                weight_decay=protocol["weight_decay"],
                patience=protocol["patience"],
            )
            checkpoint = project_path(f"checkpoints/cobra/{tag}_seed{seed}.pt")
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "seed": seed,
                    "protocol": protocol["description"],
                    "threshold": result["threshold"],
                    "manifest": (
                        tag.removeprefix("cobra_")
                        if protocol_name == "controlled"
                        else None
                    ),
                },
                checkpoint,
            )
            result["checkpoint"] = str(checkpoint.relative_to(project_path(".")))
            runs.append(result)
            external = [
                result["test"][split]["MCC"]
                for split in EXTERNAL_TESTS
                if split in result["test"]
            ]
            internal = (
                result["test"]["internal_test"]["MCC"]
                if "internal_test" in result["test"]
                else None
            )
            print(
                f"  seed {seed}: threshold={result['threshold']:.2f} "
                f"external_avg={sum(external)/len(external):.4f}"
                + (f" internal={internal:.4f}" if internal is not None else ""),
                flush=True,
            )

        summary = {
            "tag": tag,
            "protocol": protocol,
            "runs": runs,
            "n_seeds": len(runs),
        }
        output_path = f"results/training/{tag}.json"
        write_json(output_path, summary)
        summaries[tag] = {
            "output": output_path,
            "n_seeds": len(runs),
        }
    print(json.dumps(summaries, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
