from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from duet_clean.large_protocol import LARGE_MANIFEST
from duet_clean.large_results import (
    load_selected,
    output_path,
    save_standard_result,
    standardize_result,
)


METHODS = {
    "duet_graph": {
        "graphs": "data/processed/graph_large_p80_cv5.pt",
        "arch": "rgcn",
        "model_kv": None,
        "tests": ["TE18", "TL12", "JL10", "RB9"],
    },
    "duet_tcn": {
        "graphs": "data/processed/dual_large_p80_cv5.pt",
        "arch": "tcn",
        "model_kv": "ch=160",
        "tests": ["TE18", "TL12", "JL10", "RB9"],
    },
    "duet_dense": {
        "graphs": "data/processed/dual_large_p80_cv5.pt",
        "arch": "dense",
        "model_kv": "stem=160,growth=40",
        "tests": ["TE18", "TL12", "JL10", "RB9"],
    },
    "graph_seq3d": {
        "graphs": "data/processed/graph_seq3d_large_p80_cv5.pt",
        "arch": "rgcn",
        "model_kv": None,
        "tests": ["TE18", "TL12", "JL10"],
    },
    "graph_seqss3d": {
        "graphs": "data/processed/graph_seqss3d_large_p80_cv5.pt",
        "arch": "rgcn",
        "model_kv": None,
        "tests": ["TE18", "TL12", "JL10"],
    },
    "graph_seq3d_shuffle": {
        "graphs": "data/processed/graph_seq3d_shuffle_large_p80_cv5.pt",
        "arch": "rgcn",
        "model_kv": None,
        "tests": ["TE18", "TL12", "JL10"],
    },
    "rgcn_ernie": {
        "graphs": "data/processed/graph_seq3d_ernie_large_p80_cv5.pt",
        "arch": "rgcn",
        "model_kv": None,
        "tests": ["TE18", "TL12", "JL10"],
    },
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--stage", choices=["cv", "final"], required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--manifest", default=LARGE_MANIFEST)
    parser.add_argument("--results-root", default="results/large_protocol")
    parser.add_argument("--checkpoint-root", default="checkpoints/large")
    parser.add_argument("--run-prefix", default="large")
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
    spec = METHODS[args.method]
    selected = (
        load_selected(args.method, args.results_root)
        if args.stage == "final"
        else None
    )
    log_dir = ROOT / "logs" / "large" / args.method
    log_dir.mkdir(parents=True, exist_ok=True)
    jobs = (
        [(seed, fold) for seed in seeds for fold in args.folds]
        if args.stage == "cv"
        else [(seed, None) for seed in seeds]
    )
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    for seed, fold in jobs:
        standard_path = output_path(
            "cv" if fold is not None else "final",
            args.method,
            seed,
            fold,
            results_root=args.results_root,
        )
        if standard_path.exists():
            print(f"[skip] {standard_path.relative_to(ROOT)}", flush=True)
            continue
        tag = (
            f"{args.run_prefix}_{args.method}_cv_s{seed}_f{fold}"
            if fold is not None
            else f"{args.run_prefix}_{args.method}_final_s{seed}"
        )
        command = [
            sys.executable,
            str(ROOT / "src" / "duet_model" / "train.py"),
            "--config",
            str(ROOT / "configs" / "duet_clean.yaml"),
            "--graphs",
            str(ROOT / spec["graphs"]),
            "--tag",
            tag,
            "--seed",
            str(seed),
            "--loss",
            "tcl_focal",
            "--arch",
            spec["arch"],
            "--large_manifest",
            args.manifest,
            "--sample_weight_mode",
            args.sample_weight_mode,
        ]
        if spec["model_kv"]:
            command += ["--model_kv", spec["model_kv"]]
        if fold is None:
            command += [
                "--all_pool",
                "--fixed_epochs",
                str(int(selected["selected_epoch"])),
                "--fixed_threshold",
                str(float(selected["threshold"])),
            ]
        else:
            command += [
                "--fold",
                str(fold),
                "--epochs",
                str(args.epochs),
            ]
        log_path = log_dir / f"{tag}.log"
        print(f"[run ] {' '.join(command)}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.run(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        if process.returncode:
            raise RuntimeError(f"{tag} failed; see {log_path}")
        raw_path = (
            ROOT / "results" / "training" / "duet_runs" / f"train_{tag}_seed{seed}.json"
        )
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        per_chain = {
            split: raw["per_chain"][split] for split in spec["tests"]
        }
        if fold is not None:
            per_chain["val"] = raw["per_chain"]["val"]
        result = standardize_result(
            method=args.method,
            seed=seed,
            threshold=float(raw["threshold"]),
            selected_epoch=int(raw["selected_epoch"]),
            per_chain=per_chain,
            test_splits=spec["tests"],
            fold=fold,
            fixed_epoch=(
                raw.get("fixed_epoch")
                if raw.get("fixed_epoch") is not None
                else raw["selected_epoch"] if fold is None else None
            ),
            validation_threshold_score=(
                raw["val_metrics"]["MCC"] if raw.get("val_metrics") else None
            ),
            oof_threshold=(
                float(selected["threshold"]) if fold is None else None
            ),
        )
        source_checkpoint = ROOT / "checkpoints" / "duet" / f"{tag}_seed{seed}.pt"
        target_dir = ROOT / args.checkpoint_root / args.method
        target_dir.mkdir(parents=True, exist_ok=True)
        target_checkpoint = target_dir / (
            f"seed{seed}_fold{fold}.pt"
            if fold is not None
            else f"seed{seed}.pt"
        )
        shutil.copy2(source_checkpoint, target_checkpoint)
        result["checkpoint"] = str(target_checkpoint.relative_to(ROOT))
        save_standard_result(
            "cv" if fold is not None else "final",
            result,
            results_root=args.results_root,
        )
        print(
            f"  epoch={raw['selected_epoch']} threshold={raw['threshold']:.2f} "
            f"avg_MCC="
            f"{sum(result['test'][s]['MCC'] for s in spec['tests']) / len(spec['tests']):.4f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
