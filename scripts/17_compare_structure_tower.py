from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from duet_clean.comparison import (
    SEEDS,
    holm_adjust,
    load_result,
    merge_member_probabilities,
    paired_test,
)
from duet_clean.io import write_json
from duet_clean.metrics import compute_metrics, find_best_threshold


SHARED = ["TE18", "TL12", "JL10"]


def run_path(tag: str, seed: int) -> Path:
    return ROOT / "results" / "training" / "duet_runs" / f"train_{tag}_seed{seed}.json"


def graph_metrics(run: dict, threshold: float | None = None) -> dict[str, float]:
    key = "at_threshold" if threshold is None else "at_0.5"
    return {
        split: float(run["test_metrics"][split][key]["MCC"])
        for split in SHARED
    }


def pooled_from_chains(chains: dict[str, dict], threshold: float) -> float:
    labels = np.concatenate(
        [np.asarray(value["label"], dtype=int) for value in chains.values()]
    )
    probabilities = np.concatenate(
        [np.asarray(value["prob"], dtype=float) for value in chains.values()]
    )
    return float(compute_metrics(labels, probabilities, threshold)["MCC"])


def ensemble(runs: list[dict], seed: int) -> tuple[dict[str, float], dict[str, float]]:
    val = merge_member_probabilities(runs, "val")
    labels = np.concatenate(
        [np.asarray(value["label"], dtype=int) for value in val.values()]
    )
    probabilities = np.concatenate(
        [np.asarray(value["prob"], dtype=float) for value in val.values()]
    )
    threshold, _ = find_best_threshold(labels, probabilities)
    selected = {}
    fixed = {}
    for split in SHARED:
        chains = merge_member_probabilities(runs, split)
        selected[split] = pooled_from_chains(chains, threshold)
        fixed[split] = pooled_from_chains(chains, 0.5)
    return selected, fixed


def average(metrics: dict[str, float]) -> float:
    return float(np.mean([metrics[split] for split in SHARED]))


def main() -> int:
    seq = {
        seed: load_result(run_path("p80_graph", seed))
        for seed in SEEDS
    }
    seq3d = {
        seed: load_result(run_path("p80_seq3d_graph", seed))
        for seed in SEEDS
    }
    seqss3d = {
        seed: load_result(run_path("p80_seqss3d_graph", seed))
        for seed in SEEDS
    }
    seq3d_shuffle = {
        seed: load_result(run_path("p80_seq3d_shuffle_graph", seed))
        for seed in SEEDS
    }
    tcn = {
        seed: load_result(run_path("p80_tcn", seed))
        for seed in SEEDS
    }
    dense = {
        seed: load_result(run_path("p80_dense", seed))
        for seed in SEEDS
    }
    grassp = {
        int(run["seed"]): run
        for run in load_result(
            ROOT / "results" / "training" / "grassp_homology_p80.json"
        )["runs"]
    }
    cobra = {
        int(run["seed"]): run
        for run in load_result(
            ROOT / "results" / "training" / "cobra_homology_p80.json"
        )["runs"]
    }

    methods = {
        "seq+ss graph": {},
        "seq+3D graph": {},
        "seq+3D shuffled graph": {},
        "seq+ss+3D graph": {},
        "seq sequence ensemble": {},
        "seq+3D ensemble": {},
        "seq+ss+3D ensemble": {},
        "GRASSP": {},
        "CoBRA-M3": {},
    }
    methods_05 = {name: {} for name in methods}
    for seed in SEEDS:
        methods["seq+ss graph"][seed] = graph_metrics(seq[seed])
        methods["seq+3D graph"][seed] = graph_metrics(seq3d[seed])
        methods["seq+3D shuffled graph"][seed] = graph_metrics(seq3d_shuffle[seed])
        methods["seq+ss+3D graph"][seed] = graph_metrics(seqss3d[seed])
        methods["GRASSP"][seed] = {
            split: grassp[seed]["test"][split]["MCC"] for split in SHARED
        }
        methods["CoBRA-M3"][seed] = {
            split: cobra[seed]["test"][split]["MCC"] for split in SHARED
        }
        methods_05["seq+ss graph"][seed] = graph_metrics(seq[seed], 0.5)
        methods_05["seq+3D graph"][seed] = graph_metrics(seq3d[seed], 0.5)
        methods_05["seq+3D shuffled graph"][seed] = graph_metrics(
            seq3d_shuffle[seed],
            0.5,
        )
        methods_05["seq+ss+3D graph"][seed] = graph_metrics(seqss3d[seed], 0.5)
        methods_05["GRASSP"][seed] = {
            split: grassp[seed]["test_at_0.5"][split]["MCC"] for split in SHARED
        }
        methods_05["CoBRA-M3"][seed] = {
            split: pooled_from_chains(cobra[seed]["per_chain"][split], 0.5)
            for split in SHARED
        }

        for name, graph_run in [
            ("seq sequence ensemble", seq[seed]),
            ("seq+3D ensemble", seq3d[seed]),
            ("seq+ss+3D ensemble", seqss3d[seed]),
        ]:
            selected, fixed = ensemble([graph_run, tcn[seed], dense[seed]], seed)
            methods[name][seed] = selected
            methods_05[name][seed] = fixed

    summaries = {}
    summaries_05 = {}
    for name, by_seed in methods.items():
        values = np.asarray([average(by_seed[seed]) for seed in SEEDS])
        summaries[name] = {
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)),
            "values": values.tolist(),
            "per_split": {
                split: float(
                    np.mean([by_seed[seed][split] for seed in SEEDS])
                )
                for split in SHARED
            },
        }
        values_05 = np.asarray([average(methods_05[name][seed]) for seed in SEEDS])
        summaries_05[name] = {
            "mean": float(values_05.mean()),
            "sd": float(values_05.std(ddof=1)),
            "values": values_05.tolist(),
        }

    comparisons = {}
    pairs = [
        ("seq3D_vs_seq", "seq+3D graph", "seq+ss graph"),
        ("seq3D_vs_shuffled", "seq+3D graph", "seq+3D shuffled graph"),
        ("shuffled3D_vs_seq", "seq+3D shuffled graph", "seq+ss graph"),
        ("shuffled3D_vs_grassp", "seq+3D shuffled graph", "GRASSP"),
        ("seqss3D_vs_seq", "seq+ss+3D graph", "seq+ss graph"),
        ("seq3D_vs_grassp", "seq+3D graph", "GRASSP"),
        ("seq3D_vs_cobra", "seq+3D graph", "CoBRA-M3"),
        ("seqss3D_vs_cobra", "seq+ss+3D graph", "CoBRA-M3"),
        ("seq3D_ensemble_vs_seq_ensemble", "seq+3D ensemble", "seq sequence ensemble"),
        ("seq3D_ensemble_vs_grassp", "seq+3D ensemble", "GRASSP"),
        ("seqss3D_ensemble_vs_seq_ensemble", "seq+ss+3D ensemble", "seq sequence ensemble"),
        ("seqss3D_ensemble_vs_grassp", "seq+ss+3D ensemble", "GRASSP"),
    ]
    for key, first, second in pairs:
        comparisons[key] = paired_test(
            np.asarray([average(methods[first][seed]) for seed in SEEDS]),
            np.asarray([average(methods[second][seed]) for seed in SEEDS]),
        )
    comparisons_05 = {}
    for key, first, second in pairs:
        comparisons_05[key] = paired_test(
            np.asarray([average(methods_05[first][seed]) for seed in SEEDS]),
            np.asarray([average(methods_05[second][seed]) for seed in SEEDS]),
        )

    per_split = {}
    for split in SHARED:
        rows = {}
        for key, first, second in pairs:
            rows[key] = paired_test(
                np.asarray([methods[first][seed][split] for seed in SEEDS]),
                np.asarray([methods[second][seed][split] for seed in SEEDS]),
            )
        per_split[split] = rows

    report = {
        "manifest": "homology_p80",
        "tests": SHARED,
        "seeds": SEEDS,
        "summaries": summaries,
        "summaries_at_0.5": summaries_05,
        "comparisons": comparisons,
        "comparisons_at_0.5": comparisons_05,
        "per_split": per_split,
    }
    write_json("results/training/structure_graph_tower.json", report)
    write_markdown(report)
    print(json.dumps({
        name: {"mean": value["mean"], "sd": value["sd"]}
        for name, value in summaries.items()
    }, indent=2, sort_keys=True))
    return 0


def write_markdown(report: dict) -> None:
    lines = [
        "# True 3D structure in the DUET graph tower",
        "",
        "Generated by `scripts/17_compare_structure_tower.py`.",
        "",
        "| Method | Mean MCC | SD | TE18 | TL12 | JL10 |",
        "|:--|--:|--:|--:|--:|--:|",
    ]
    for name, value in report["summaries"].items():
        lines.append(
            f"| {name} | {value['mean']:.3f} | {value['sd']:.3f} | "
            + " | ".join(f"{value['per_split'][split]:.3f}" for split in SHARED)
            + " |"
        )
    lines += [
        "",
        "| Comparison | Delta | 95% CI | p | Better seeds |",
        "|:--|--:|:--|--:|--:|",
    ]
    for name, value in report["comparisons"].items():
        lines.append(
            f"| {name} | {value['mean_difference']:+.3f} | "
            f"{value['ci95'][0]:+.3f} to {value['ci95'][1]:+.3f} | "
            f"{value['p']:.4g} | {value['seeds_first_better']}/{value['n']} |"
        )
    lines += [
        "",
        "At a fixed 0.5 threshold:",
        "",
    ]
    for name in [
        "seq+ss graph",
        "seq+3D graph",
        "seq+3D shuffled graph",
        "seq+ss+3D graph",
        "seq sequence ensemble",
        "seq+3D ensemble",
        "seq+ss+3D ensemble",
        "GRASSP",
        "CoBRA-M3",
    ]:
        value = report["summaries_at_0.5"][name]
        lines.append(f"- {name}: {value['mean']:.3f} ± {value['sd']:.3f}")
    lines += ["", "Fixed-threshold paired tests:"]
    for name in [
        "seq3D_vs_seq",
        "seq3D_vs_shuffled",
        "seqss3D_vs_seq",
        "seq3D_vs_cobra",
    ]:
        value = report["comparisons_at_0.5"][name]
        lines.append(
            f"- {name}: {value['mean_difference']:+.3f} MCC, p = {value['p']:.4g}."
        )
    lines += [
        "",
        "The 3D edges are the directed top-8 structural neighbours in the released "
        "GRASSP graphs. RB9 is excluded because no aligned structural graph is available.",
        "",
    ]
    path = ROOT / "results" / "training" / "STRUCTURE_GRAPH_TOWER.md"
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
