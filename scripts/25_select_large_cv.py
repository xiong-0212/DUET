from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from duet_clean.large_training import select_cv_hyperparameters


METHODS = [
    "cobra",
    "duet_graph",
    "duet_tcn",
    "duet_dense",
    "graph_seq3d",
    "graph_seq3d_shuffle",
    "graph_seqss3d",
    "grassp_ernie",
    "grassp_rnafm",
    "rgcn_ernie",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-root", default="results/large_protocol")
    args = parser.parse_args()

    for method in args.methods:
        base = ROOT / args.results_root / "cv" / method
        paths = [base / f"seed{args.seed}_fold{fold}.json" for fold in range(5)]
        missing = [str(path.relative_to(ROOT)) for path in paths if not path.exists()]
        if missing:
            print(f"[wait] {method}: missing {len(missing)} folds", flush=True)
            continue
        folds = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        selected = select_cv_hyperparameters(folds)
        selected.update(
            {
                "method": method,
                "cv_seed": args.seed,
                "folds": [0, 1, 2, 3, 4],
                "fold_files": [str(path.relative_to(ROOT)) for path in paths],
            }
        )
        output = base / "selected.json"
        output.write_text(
            json.dumps(selected, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            f"[selected] {method}: epoch={selected['selected_epoch']} "
            f"threshold={selected['threshold']:.3f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
