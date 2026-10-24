from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch_geometric.data import Data

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from duet_clean.grassp import train_one
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


TEST_SPLITS = ["TE18", "TL12", "JL10"]


def load_graphs(path: Path) -> dict[str, list[tuple[str, Data]]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return {
        split: [(str(graph.rna_id), graph) for graph in payload["graphs"][split]]
        for split in payload["graphs"]
    }


def select_pool(
    graphs: dict[str, list[tuple[str, Data]]],
    ids: set[str],
) -> list[tuple[str, Data]]:
    selected = [item for item in graphs["pool"] if item[0] in ids]
    if len(selected) != len(ids):
        missing = sorted(ids - {item[0] for item in selected})
        raise KeyError(f"missing GRASSP graphs: {missing[:10]}")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=["grassp_ernie", "grassp_rnafm"],
        required=True,
    )
    parser.add_argument("--stage", choices=["cv", "final"], required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--manifest", default=LARGE_MANIFEST)
    parser.add_argument("--results-root", default="results/large_protocol")
    parser.add_argument("--checkpoint-root", default="checkpoints/large")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=86)
    parser.add_argument("--dropout", type=float, default=0.6)
    parser.add_argument(
        "--sample-weight-mode",
        choices=["group", "uniform", "none"],
        default="group",
    )
    args = parser.parse_args()
    seeds = (
        args.seeds or [42]
        if args.stage == "cv"
        else args.seeds or [42, 0, 1, 2, 3, 7, 11, 13, 17, 23]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, manifest = load_large_manifest(args.manifest)
    if args.sample_weight_mode == "group":
        weights = sampling_weights(manifest)
    elif args.sample_weight_mode == "uniform":
        weights = {
            record["rna_id"]: 1.0
            for record in manifest["splits"]["pool"]
        }
    else:
        weights = None
    feature = "ernie" if args.method.endswith("ernie") else "rnafm"
    graph_path = (
        ROOT
        / "data"
        / "processed"
        / f"grassp_{feature}_large_p80_cv5.pt"
    )
    graphs = load_graphs(graph_path)
    selected = (
        load_selected(args.method, args.results_root)
        if args.stage == "final"
        else None
    )
    checkpoint_dir = ROOT / args.checkpoint_root / args.method
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    jobs = (
        [(seed, fold) for seed in seeds for fold in args.folds]
        if args.stage == "cv"
        else [(seed, None) for seed in seeds]
    )

    for seed, fold in jobs:
        path = output_path(
            "cv" if fold is not None else "final",
            args.method,
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
            train_ids = {
                record["rna_id"]
                for record in manifest["splits"]["pool"]
            }
            val_ids = set()
        splits = {
            "train": select_pool(graphs, train_ids),
            "val": select_pool(graphs, val_ids) if val_ids else [],
            **{split: graphs[split] for split in TEST_SPLITS},
        }
        fixed_epoch = None
        threshold = None
        if args.stage == "final":
            fixed_epoch = int(selected["selected_epoch"])
            threshold = float(selected["threshold"])
        print(
            f"[{args.method}] stage={args.stage} seed={seed} fold={fold} "
            f"train={len(splits['train'])} val={len(splits['val'])}",
            flush=True,
        )
        model, raw = train_one(
            splits=splits,
            seed=seed,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            hidden=args.hidden,
            dropout=args.dropout,
            batch_size=args.batch_size,
            device=device,
            rna_dim=768 if feature == "ernie" else 640,
            sample_weights=weights,
            fixed_epoch=fixed_epoch,
            threshold_override=threshold,
            evaluation_splits=TEST_SPLITS,
            patience=None,
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
                "feature": feature,
                "sample_weight": raw["sample_weight"],
            },
            checkpoint,
        )
        result = standardize_result(
            method=args.method,
            seed=seed,
            threshold=float(raw["threshold"]),
            selected_epoch=int(raw["selected_epoch"]),
            per_chain={
                split: raw["per_chain"][split] for split in TEST_SPLITS
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
            f"avg_MCC={sum(result['test'][s]['MCC'] for s in TEST_SPLITS) / 3:.4f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
