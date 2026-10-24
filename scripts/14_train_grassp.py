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

from duet_clean.experiments import manifest_splits
from duet_clean.grassp import load_grassp_splits, train_one
from duet_clean.io import write_json


SEEDS = [42, 0, 1, 2, 3, 7, 11, 13, 17, 23]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="homology_p80")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=86)
    parser.add_argument("--dropout", type=float, default=0.6)
    parser.add_argument("--features", choices=["ernie", "rnafm"], default="ernie")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_splits, manifest = manifest_splits(args.manifest)
    splits = load_grassp_splits(raw_splits, feature_source=args.features)
    print(
        f"manifest={args.manifest} train={len(splits['train'])} val={len(splits['val'])} "
        + " ".join(f"{split}={len(splits[split])}" for split in ["TE18", "TL12", "JL10"]),
        flush=True,
    )

    suffix = "" if args.features == "ernie" else "_rnafm"
    output_path = (
        ROOT
        / "results"
        / "training"
        / f"grassp_{args.manifest}{suffix}.json"
    )
    runs = []
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing.get("manifest") == args.manifest:
            runs = list(existing.get("runs", []))
    existing_seeds = {int(run["seed"]) for run in runs}
    for seed in args.seeds:
        if seed in existing_seeds:
            print(f"[skip] existing seed {seed}", flush=True)
            continue
        print(f"[GRASSP] seed {seed}", flush=True)
        model, result = train_one(
            splits=splits,
            seed=seed,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            hidden=args.hidden,
            dropout=args.dropout,
            batch_size=args.batch_size,
            device=device,
            rna_dim=768 if args.features == "ernie" else 640,
        )
        prefix = "p80" if args.manifest == "homology_p80" else "exact"
        if args.features == "rnafm":
            prefix += "_rnafm"
        checkpoint = ROOT / "checkpoints" / "grassp" / f"{prefix}_seed{seed}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "seed": seed,
                "manifest": args.manifest,
                "threshold": result["threshold"],
                "config": {
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate,
                    "weight_decay": args.weight_decay,
                    "hidden": args.hidden,
                    "dropout": args.dropout,
                    "features": args.features,
                },
            },
            checkpoint,
        )
        result["checkpoint"] = str(checkpoint.relative_to(ROOT))
        runs.append(result)
        avg = sum(result["test"][split]["MCC"] for split in ["TE18", "TL12", "JL10"]) / 3
        print(
            f"  threshold={result['threshold']:.2f} three-set mean={avg:.4f}",
            flush=True,
        )
        write_json(
            f"results/training/grassp_{args.manifest}{suffix}.json",
            {
                "manifest": args.manifest,
                "manifest_sha256": manifest["manifest_sha256"],
                "protocol": {
                    "architecture": "GRASSP HybridRNABindingSiteModel",
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate,
                    "weight_decay": args.weight_decay,
                    "hidden": args.hidden,
                    "dropout": args.dropout,
                    "features": args.features,
                    "threshold": "validation MCC",
                },
                "runs": sorted(runs, key=lambda run: int(run["seed"])),
            },
        )
    print(json.dumps({
        "runs": len(runs),
        "output": f"results/training/grassp_{args.manifest}{suffix}.json",
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
