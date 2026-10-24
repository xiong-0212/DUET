from __future__ import annotations

import argparse
import itertools
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SEEDS = [42, 0, 1, 2, 3, 7, 11, 13, 17, 23]
FEATURES = ["ernie", "rnafm"]
GRID = list(
    itertools.product(
        [1e-4, 1e-3, 1e-2],
        [1, 16],
        [86, 128],
        [0.3, 0.6],
    )
)
TOP_K = 3


def config_name(lr: float, batch: int, hidden: int, dropout: float) -> str:
    return f"lr{lr:g}_bs{batch}_h{hidden}_d{dropout:g}"


def result_root(feature: str, name: str) -> Path:
    return (
        ROOT
        / "results"
        / "followup"
        / "grasp_grid"
        / feature
        / name
    )


def checkpoint_root(feature: str, name: str) -> Path:
    return (
        ROOT
        / "checkpoints"
        / "followup"
        / "grasp_grid"
        / feature
        / name
    )


def run_selection(root: Path, method: str) -> int:
    return subprocess.run(
        [
            sys.executable,
            str(HERE / "25_select_large_cv.py"),
            "--methods",
            method,
            "--seed",
            "42",
            "--results-root",
            str(root.relative_to(ROOT)),
        ],
        cwd=ROOT,
    ).returncode


def run_tasks(
    feature: str,
    jobs: int,
    tasks: list[tuple[float, int, int, float, int]],
) -> list[str]:
    failures = []
    queue = list(tasks)
    running = []
    log_dir = ROOT / "logs" / "followup" / "grasp_grid" / feature
    log_dir.mkdir(parents=True, exist_ok=True)
    while queue or running:
        while queue and len(running) < jobs:
            lr, batch, hidden, dropout, fold = queue.pop(0)
            name = config_name(lr, batch, hidden, dropout)
            root = result_root(feature, name)
            output = (
                root
                / "cv"
                / f"grassp_{feature}"
                / f"seed42_fold{fold}.json"
            )
            if output.exists():
                continue
            command = [
                sys.executable,
                str(HERE / "23_train_large_grassp.py"),
                "--method",
                f"grassp_{feature}",
                "--stage",
                "cv",
                "--seeds",
                "42",
                "--epochs",
                "100",
                "--folds",
                str(fold),
                "--batch-size",
                str(batch),
                "--learning-rate",
                str(lr),
                "--hidden",
                str(hidden),
                "--dropout",
                str(dropout),
                "--manifest",
                "large_p80_cv5",
                "--results-root",
                str(root.relative_to(ROOT)),
                "--checkpoint-root",
                str(checkpoint_root(feature, name).relative_to(ROOT)),
            ]
            log = (
                log_dir / f"{name}_fold{fold}.log"
            ).open("w", encoding="utf-8")
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            running.append((process, name, fold, time.time(), log))
            print(f"[grid] {feature} {name} fold {fold}", flush=True)
        for index, (process, name, fold, started, log) in enumerate(list(running)):
            code = process.poll()
            if code is None:
                continue
            running.pop(index)
            log.close()
            if code:
                failures.append(f"{feature}:{name}:fold{fold}")
                print(f"[fail] {feature} {name} fold {fold}", flush=True)
            else:
                print(
                    f"[done] {feature} {name} fold {fold} "
                    f"({(time.time() - started) / 60:.1f} min)",
                    flush=True,
                )
            break
        time.sleep(2)
    return failures


def screen_fold_zero(feature: str, jobs: int) -> tuple[list[dict], list[str]]:
    failures = run_tasks(
        feature,
        jobs,
        [
            (lr, batch, hidden, dropout, 0)
            for lr, batch, hidden, dropout in GRID
        ],
    )
    rows = []
    for lr, batch, hidden, dropout in GRID:
        name = config_name(lr, batch, hidden, dropout)
        path = (
            result_root(feature, name)
            / "cv"
            / f"grassp_{feature}"
            / "seed42_fold0.json"
        )
        if not path.exists():
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "name": name,
                "learning_rate": lr,
                "batch_size": batch,
                "hidden": hidden,
                "dropout": dropout,
                "fold0_validation_threshold_score": float(
                    value["validation_threshold_score"]
                ),
            }
        )
    rows.sort(
        key=lambda row: row["fold0_validation_threshold_score"],
        reverse=True,
    )
    return rows, failures


def confirm_shortlist(
    feature: str,
    jobs: int,
    shortlist: list[dict],
) -> list[str]:
    tasks = []
    for row in shortlist[:TOP_K]:
        for fold in range(1, 5):
            tasks.append(
                (
                    float(row["learning_rate"]),
                    int(row["batch_size"]),
                    int(row["hidden"]),
                    float(row["dropout"]),
                    fold,
                )
            )
    failures = run_tasks(feature, jobs, tasks)
    for row in shortlist[:TOP_K]:
        root = result_root(feature, row["name"])
        code = run_selection(root, f"grassp_{feature}")
        if code:
            failures.append(f"{feature}:{row['name']}:selection")
    return failures


def write_grid_selection(
    feature: str,
    fold0_rows: list[dict],
    shortlist: list[dict],
) -> dict:
    candidates = []
    for row in shortlist[:TOP_K]:
        selected_path = (
            result_root(feature, row["name"])
            / "cv"
            / f"grassp_{feature}"
            / "selected.json"
        )
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
        candidates.append(
            {
                **row,
                "source_root": str(selected_path.parent.parent.parent.relative_to(ROOT)),
                **selected,
            }
        )
    candidates.sort(
        key=lambda value: float(
            value["oof_metrics_at_selected_threshold"]["MCC"]
        ),
        reverse=True,
    )
    best = candidates[0]
    output = (
        ROOT
        / "results"
        / "followup"
        / "grasp_grid"
        / feature
        / "grid_selection.json"
    )
    output.write_text(
        json.dumps(
            {
                "feature": feature,
                "design": (
                    "all 24 configurations on large-data fold 0; "
                    f"top {TOP_K} confirmed on folds 1-4"
                ),
                "fold0_screen": fold0_rows,
                "confirmed_configurations": len(candidates),
                "best": best,
                "candidates": candidates,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"[best] {feature}: {best['name']} "
        f"OOF MCC={best['oof_metrics_at_selected_threshold']['MCC']:.4f}"
    )
    return best


def run_best_final(feature: str, best: dict) -> int:
    best_root = (
        ROOT
        / "results"
        / "followup"
        / "grasp_grid"
        / "best"
        / feature
    )
    selected_dir = best_root / "cv" / f"grassp_{feature}"
    selected_dir.mkdir(parents=True, exist_ok=True)
    source_selected = (
        ROOT
        / best["source_root"]
        / "cv"
        / f"grassp_{feature}"
        / "selected.json"
    )
    shutil.copy2(source_selected, selected_dir / "selected.json")
    command = [
        sys.executable,
        str(HERE / "23_train_large_grassp.py"),
        "--method",
        f"grassp_{feature}",
        "--stage",
        "final",
        "--seeds",
        *[str(seed) for seed in SEEDS],
        "--batch-size",
        str(best["batch_size"]),
        "--learning-rate",
        str(best["learning_rate"]),
        "--hidden",
        str(best["hidden"]),
        "--dropout",
        str(best["dropout"]),
        "--manifest",
        "large_p80_cv5",
        "--results-root",
        str(best_root.relative_to(ROOT)),
        "--checkpoint-root",
        str(
            (
                ROOT
                / "checkpoints"
                / "followup"
                / "grasp_grid"
                / "best"
                / feature
            ).relative_to(ROOT)
        ),
    ]
    log_path = (
        ROOT
        / "logs"
        / "followup"
        / "grasp_grid"
        / f"best_{feature}.log"
    )
    with log_path.open("w", encoding="utf-8") as log:
        return subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
        ).returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["grid", "final", "all"], default="all")
    parser.add_argument("--features", nargs="+", choices=FEATURES, default=FEATURES)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    failures = []
    best_by_feature = {}
    if args.stage in {"grid", "all"}:
        for feature in args.features:
            fold0_rows, fold_failures = screen_fold_zero(feature, args.jobs)
            failures.extend(fold_failures)
            if failures:
                break
            if len(fold0_rows) != len(GRID):
                failures.append(f"{feature}: fold-0 screen incomplete")
                break
            failures.extend(confirm_shortlist(feature, args.jobs, fold0_rows))
            if failures:
                break
            best_by_feature[feature] = write_grid_selection(
                feature,
                fold0_rows,
                fold0_rows,
            )
    if failures:
        print(f"grid failures: {failures}", file=sys.stderr)
        return 1
    if args.stage in {"final", "all"}:
        for feature in args.features:
            best = best_by_feature.get(feature)
            if best is None:
                best = json.loads(
                    (
                        ROOT
                        / "results"
                        / "followup"
                        / "grasp_grid"
                        / feature
                        / "grid_selection.json"
                    ).read_text(encoding="utf-8")
                )["best"]
            if run_best_final(feature, best):
                failures.append(f"best-final:{feature}")
    if failures:
        print(f"failures: {failures}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
