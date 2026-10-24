from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from duet_clean.comparison import (
    holm_adjust,
    load_result,
    paired_test,
)
from duet_clean.io import write_json
from duet_clean.metrics import compute_metrics


SHARED = ["TE18", "TL12", "JL10"]


def pooled_mcc(chains: dict[str, dict], threshold: float) -> float:
    labels = np.concatenate(
        [np.asarray(value["label"], dtype=int) for value in chains.values()]
    )
    probabilities = np.concatenate(
        [np.asarray(value["prob"], dtype=float) for value in chains.values()]
    )
    return compute_metrics(labels, probabilities, threshold)["MCC"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        choices=["homology_p80", "exact_sequence_unique"],
        default="homology_p80",
    )
    args = parser.parse_args()
    if args.manifest == "homology_p80":
        prefix = "p80"
        grassp_file = "grassp_homology_p80.json"
        duet_file = "duet_ensemble_p80.json"
        cobra_file = "cobra_homology_p80.json"
        output_name = "grassp_vs_sequence.json"
        markdown_name = "GRASSP_VS_SEQUENCE.md"
    else:
        prefix = "exact"
        grassp_file = "grassp_exact_sequence_unique.json"
        duet_file = "duet_ensemble_exact.json"
        cobra_file = "cobra_exact_clean.json"
        output_name = "grassp_vs_sequence_exact.json"
        markdown_name = "GRASSP_VS_SEQUENCE_EXACT.md"

    grassp = load_result(ROOT / "results" / "training" / grassp_file)
    grassp_runs = {int(run["seed"]): run for run in grassp["runs"]}
    seeds = sorted(grassp_runs)
    duet = load_result(ROOT / "results" / "training" / duet_file)
    duet_runs = {int(run["seed"]): run for run in duet["seeds"]}
    graph_runs = {
        seed: load_result(
            ROOT
            / "results"
            / "training"
            / "duet_runs"
            / f"train_{prefix}_graph_seed{seed}.json"
        )
        for seed in seeds
    }
    cobra = load_result(ROOT / "results" / "training" / cobra_file)
    cobra_runs = {int(run["seed"]): run for run in cobra["runs"]}

    values = {"GRASSP": {}, "DUET": {}, "DUET graph": {}, "CoBRA-M3": {}}
    values_05 = {"GRASSP": {}, "DUET": {}, "DUET graph": {}, "CoBRA-M3": {}}
    for seed in seeds:
        grassp_mcc = [grassp_runs[seed]["test"][split]["MCC"] for split in SHARED]
        grassp_05 = [grassp_runs[seed]["test_at_0.5"][split]["MCC"] for split in SHARED]
        duet_mcc = [duet_runs[seed]["metrics"][split]["MCC"] for split in SHARED]
        duet_05 = [duet_runs[seed]["metrics_at_0.5"][split]["MCC"] for split in SHARED]
        graph_mcc = [
            graph_runs[seed]["test_metrics"][split]["at_threshold"]["MCC"]
            for split in SHARED
        ]
        graph_05 = [
            graph_runs[seed]["test_metrics"][split]["at_0.5"]["MCC"]
            for split in SHARED
        ]
        cobra_mcc = [cobra_runs[seed]["test"][split]["MCC"] for split in SHARED]
        cobra_05 = [
            pooled_mcc(cobra_runs[seed]["per_chain"][split], 0.5)
            for split in SHARED
        ]
        values["GRASSP"][seed] = grassp_mcc
        values["DUET"][seed] = duet_mcc
        values["DUET graph"][seed] = graph_mcc
        values["CoBRA-M3"][seed] = cobra_mcc
        values_05["GRASSP"][seed] = grassp_05
        values_05["DUET"][seed] = duet_05
        values_05["DUET graph"][seed] = graph_05
        values_05["CoBRA-M3"][seed] = cobra_05

    summaries = {}
    summaries_05 = {}
    for name, by_seed in values.items():
        averages = np.asarray([np.mean(by_seed[seed]) for seed in seeds])
        per_split = {}
        for index, split in enumerate(SHARED):
            split_values = np.asarray([by_seed[seed][index] for seed in seeds])
            per_split[split] = {
                "mean": float(split_values.mean()),
                "sd": float(split_values.std(ddof=1)),
                "values": split_values.tolist(),
            }
        summaries[name] = {
            "average": {
                "mean": float(averages.mean()),
                "sd": float(averages.std(ddof=1)),
                "values": averages.tolist(),
            },
            "per_split": per_split,
        }
        averages_05 = np.asarray([np.mean(values_05[name][seed]) for seed in seeds])
        summaries_05[name] = {
            "average": {
                "mean": float(averages_05.mean()),
                "sd": float(averages_05.std(ddof=1)),
                "values": averages_05.tolist(),
            }
        }

    def seed_averages(method: str) -> np.ndarray:
        return np.asarray([np.mean(values[method][seed]) for seed in seeds])

    comparisons = {
        "GRASSP_vs_DUET": paired_test(seed_averages("GRASSP"), seed_averages("DUET")),
        "GRASSP_vs_DUET_graph": paired_test(
            seed_averages("GRASSP"),
            seed_averages("DUET graph"),
        ),
        "GRASSP_vs_CoBRA": paired_test(seed_averages("GRASSP"), seed_averages("CoBRA-M3")),
        "DUET_vs_CoBRA": paired_test(seed_averages("DUET"), seed_averages("CoBRA-M3")),
        "DUET_graph_vs_CoBRA": paired_test(
            seed_averages("DUET graph"),
            seed_averages("CoBRA-M3"),
        ),
    }
    comparisons_05 = {
        "GRASSP_vs_DUET": paired_test(
            np.asarray([np.mean(values_05["GRASSP"][seed]) for seed in seeds]),
            np.asarray([np.mean(values_05["DUET"][seed]) for seed in seeds]),
        ),
        "GRASSP_vs_DUET_graph": paired_test(
            np.asarray([np.mean(values_05["GRASSP"][seed]) for seed in seeds]),
            np.asarray([np.mean(values_05["DUET graph"][seed]) for seed in seeds]),
        ),
        "GRASSP_vs_CoBRA": paired_test(
            np.asarray([np.mean(values_05["GRASSP"][seed]) for seed in seeds]),
            np.asarray([np.mean(values_05["CoBRA-M3"][seed]) for seed in seeds]),
        ),
    }
    per_split = {}
    for split_index, split in enumerate(SHARED):
        first = np.asarray([values["GRASSP"][seed][split_index] for seed in seeds])
        second = np.asarray([values["DUET"][seed][split_index] for seed in seeds])
        third = np.asarray([values["CoBRA-M3"][seed][split_index] for seed in seeds])
        per_split[split] = {
            "GRASSP_vs_DUET": paired_test(first, second),
            "GRASSP_vs_CoBRA": paired_test(first, third),
        }
    for key in ("GRASSP_vs_DUET", "GRASSP_vs_CoBRA"):
        raw = [per_split[split][key]["p"] for split in SHARED]
        adjusted = holm_adjust(raw)
        for split, p_value in zip(SHARED, adjusted):
            per_split[split][key]["p_holm"] = p_value

    report = {
        "manifest": args.manifest,
        "shared_tests": SHARED,
        "seeds": seeds,
        "summaries": summaries,
        "summaries_at_0.5": summaries_05,
        "comparisons": comparisons,
        "comparisons_at_0.5": comparisons_05,
        "per_split": per_split,
        "protocol_note": (
            "GRASSP uses its released structural graph features and model with its "
            "authors' defaults (hidden 86, dropout 0.6, lr 1e-4, batch 1, 100 epochs). "
            "All methods use the same train/validation manifest and the same "
            "test chains in the three shared benchmarks."
        ),
    }
    write_json(f"results/training/{output_name}", report)
    write_markdown(report, output_path=markdown_name)
    print(
        json.dumps(
            {
                name: {
                    "mean": summary["average"]["mean"],
                    "sd": summary["average"]["sd"],
                }
                for name, summary in summaries.items()
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def write_markdown(report: dict, output_path: str) -> None:
    lines = [
        f"# Structural-data comparison on the `{report['manifest']}` split",
        "",
        "Generated by `scripts/15_compare_grassp.py`.",
        "",
        "| Method | Mean | SD | TE18 | TL12 | JL10 |",
        "|:--|--:|--:|--:|--:|--:|",
    ]
    for name, summary in report["summaries"].items():
        lines.append(
            f"| {name} | {summary['average']['mean']:.3f} | "
            f"{summary['average']['sd']:.3f} | "
            + " | ".join(
                f"{summary['per_split'][split]['mean']:.3f}"
                for split in SHARED
            )
            + " |"
        )
    lines += [
        "",
        "| Comparison | Delta | 95% CI | p | Better seeds |",
        "|:--|--:|:--|--:|--:|",
    ]
    for name, result in report["comparisons"].items():
        lines.append(
            f"| {name} | {result['mean_difference']:+.3f} | "
            f"{result['ci95'][0]:+.3f} to {result['ci95'][1]:+.3f} | "
            f"{result['p']:.4g} | {result['seeds_first_better']}/{result['n']} |"
        )
    lines += ["", "At a fixed 0.5 threshold:"]
    for name, result in report["comparisons_at_0.5"].items():
        lines.append(
            f"- {name}: {result['mean_difference']:+.3f} MCC, "
            f"95% CI {result['ci95'][0]:+.3f} to {result['ci95'][1]:+.3f}, "
            f"p = {result['p']:.4g}, {result['seeds_first_better']}/{result['n']} seeds."
        )
    lines += [
        "",
        "## Per benchmark",
        "",
        "| Benchmark | GRASSP | DUET | Delta (G-D) | p | p Holm | GRASSP vs CoBRA Delta | p | p Holm |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for split in SHARED:
        g = report["summaries"]["GRASSP"]["per_split"][split]
        d = report["summaries"]["DUET"]["per_split"][split]
        gd = report["per_split"][split]["GRASSP_vs_DUET"]
        gc = report["per_split"][split]["GRASSP_vs_CoBRA"]
        lines.append(
            f"| {split} | {g['mean']:.3f} | {d['mean']:.3f} | "
            f"{gd['mean_difference']:+.3f} | {gd['p']:.4g} | {gd['p_holm']:.4g} | "
            f"{gc['mean_difference']:+.3f} | {gc['p']:.4g} | {gc['p_holm']:.4g} |"
        )
    lines += [
        "",
        "RB9 is excluded because the released GRASSP processed graphs do not contain "
        "an aligned RB9 test set with the same labels.",
        "",
    ]
    path = ROOT / "results" / "training" / output_path
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
