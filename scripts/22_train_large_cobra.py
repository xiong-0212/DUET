from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from duet_clean.cobra import attach_embeddings, train_one
from duet_clean.comparison import EXTERNAL_TESTS
from duet_clean.experiments import load_rinalmo_embeddings
from duet_clean.io import write_json
from duet_clean.large_protocol import (
    LARGE_MANIFEST,
    load_large_manifest,
    sampling_weights,
)
from duet_clean.large_results import (
    load_selected,
    output_path,
    save_standard_result,
    standardize_result,
)


METHOD = "cobra"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["cv", "final"], required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--manifest", default=LARGE_MANIFEST)
    parser.add_argument("--results-root", default="results/large_protocol")
    parser.add_argument("--checkpoint-root", default="checkpoints/large")
    parser.add_argument(
        "--sample-weight-mode",
        choices=["group", "uniform", "none"],
        default="group",
    )
    args = parser.parse_args()
    if args.stage == "cv":
        seeds = args.seeds or [42]
    else:
        seeds = args.seeds or [42, 0, 1, 2, 3, 7, 11, 13, 17, 23]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records, manifest = load_large_manifest(args.manifest)
    if args.sample_weight_mode == "group":
        weights = sampling_weights(manifest)
    elif args.sample_weight_mode == "uniform":
        weights = {
            record.rna_id: 1.0
            for record in records["pool"]
        }
    else:
        weights = None
    embeddings = load_rinalmo_embeddings("data/processed/rinalmo_embeddings.pt")
    selected = (
        load_selected(METHOD, args.results_root)
        if args.stage == "final"
        else None
    )
    checkpoint_dir = ROOT / args.checkpoint_root / METHOD
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    jobs = (
        [(seed, fold) for seed in seeds for fold in args.folds]
        if args.stage == "cv"
        else [(seed, None) for seed in seeds]
    )
    for seed, fold in jobs:
        path = output_path(
            "cv" if fold is not None else "final",
            METHOD,
            seed,
            fold,
            results_root=args.results_root,
        )
        if path.exists():
            print(f"[skip] {path.relative_to(ROOT)}", flush=True)
            continue
        if fold is not None:
            fold_spec = manifest["folds"][fold]
            train_ids = set(fold_spec["training_ids"])
            val_ids = set(fold_spec["validation_ids"])
        else:
            train_ids = set(records["pool"][index].rna_id for index in range(len(records["pool"])))
            val_ids = set()
        splits = {
            "train": [record for record in records["pool"] if record.rna_id in train_ids],
            "val": [record for record in records["pool"] if record.rna_id in val_ids],
            **{split: records[split] for split in EXTERNAL_TESTS},
        }
        data = attach_embeddings(splits, embeddings)
        fixed_epoch = None
        threshold = None
        if args.stage == "final":
            fixed_epoch = int(selected["selected_epoch"])
            threshold = float(selected["threshold"])
        print(
            f"[{METHOD}] stage={args.stage} seed={seed} fold={fold} "
            f"train={len(data['train'])} val={len(data['val'])}",
            flush=True,
        )
        model, raw = train_one(
            data=data,
            test_splits=EXTERNAL_TESTS,
            seed=seed,
            protocol="controlled",
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=5e-4,
            weight_decay=1e-4,
            patience=args.patience if fold is not None else None,
            sample_weights=weights,
            fixed_epoch=fixed_epoch,
            threshold_override=threshold,
        )
        checkpoint = checkpoint_dir / (
            f"seed{seed}_fold{fold}.pt"
            if fold is not None
            else f"seed{seed}.pt"
        )
        torch.save(
            {
                "state_dict": model.state_dict(),
                "seed": seed,
                "fold": fold,
                "threshold": raw["threshold"],
                "selected_epoch": raw["selected_epoch"],
                "sample_weight": raw["sample_weight"],
            },
            checkpoint,
        )
        result = standardize_result(
            method=METHOD,
            seed=seed,
            threshold=float(raw["threshold"]),
            selected_epoch=int(raw["selected_epoch"]),
            per_chain={
                split: raw["per_chain"][split]
                for split in EXTERNAL_TESTS
                if split in raw["per_chain"]
            }
            | ({"val": raw["per_chain"]["val"]} if fold is not None else {}),
            fold=fold,
            fixed_epoch=fixed_epoch,
            validation_threshold_score=raw["validation_threshold_score"],
            oof_threshold=threshold,
        )
        result["checkpoint"] = str(checkpoint.relative_to(ROOT))
        save_standard_result(
            "cv" if fold is not None else "final",
            result,
            results_root=args.results_root,
        )
        print(
            f"  epoch={raw['selected_epoch']} threshold={raw['threshold']:.2f} "
            f"avg_MCC={sum(result['test'][s]['MCC'] for s in EXTERNAL_TESTS) / 4:.4f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
