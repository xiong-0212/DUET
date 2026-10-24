from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    exact = load(ROOT / "results" / "training" / "duet_vs_cobra_exact.json")
    p80 = load(ROOT / "results" / "training" / "duet_vs_cobra_p80.json")
    grassp = load(ROOT / "results" / "training" / "grassp_vs_sequence.json")
    grassp_exact = load(
        ROOT / "results" / "training" / "grassp_vs_sequence_exact.json"
    )
    structure_tower = load(ROOT / "results" / "training" / "structure_graph_tower.json")
    reproduction = exact["cobra_reproduction"]
    lines = [
        "# Training summary",
        "",
        "All comparisons use ten paired seeds. GRASSP was trained on the same p80 "
        "split using its released structural graph features.",
        "",
        "## CoBRA paper reproduction",
        "",
        "| Source | TE18 | TL12 | JL10 | RB9 | Average |",
        "|:--|--:|--:|--:|--:|--:|",
        "| CoBRA paper Table 3 | "
        + " | ".join(f"{reproduction['paper_external_MCC'][split]:.3f}" for split in ["TE18", "TL12", "JL10", "RB9"])
        + f" | {reproduction['paper_average_MCC']:.3f} |",
        "| Released checkpoint, unmodified tests | "
        + " | ".join(
            f"{reproduction['released_checkpoint']['paper_released']['per_split_MCC'][split]:.3f}"
            for split in ["TE18", "TL12", "JL10", "RB9"]
        )
        + f" | {reproduction['released_checkpoint']['paper_released']['average_external_MCC']:.3f} |",
        "| Retrained CoBRA-M3, seed 42 | "
        + " | ".join(
            f"{reproduction['retrained_seed42_external_MCC'][split]:.3f}"
            for split in ["TE18", "TL12", "JL10", "RB9"]
        )
        + f" | {reproduction['retrained_seed42_external_average']:.3f} |",
        "",
        f"Retrained seed-42 internal-test MCC: "
        f"{reproduction['retrained_seed42_internal_MCC']:.3f} "
        "(CoBRA paper Table 2: 0.629).",
        "",
        "## Controlled comparisons",
        "",
        "| Split | CoBRA MCC | DUET MCC | Delta | Paired t p | DUET better |",
        "|:--|--:|--:|--:|--:|--:|",
    ]
    for name, report in [("Exact-sequence unique", exact), ("Homology p80", p80)]:
        primary = report["primary_duet_vs_cobra"]
        lines.append(
            f"| {name} | "
            f"{report['summaries']['CoBRA-M3']['average']['mean']:.3f} | "
            f"{report['summaries']['DUET ensemble']['average']['mean']:.3f} | "
            f"{primary['mean_difference']:+.3f} | {primary['p']:.4g} | "
            f"{primary['seeds_first_better']}/{primary['n']} |"
        )
    lines += [
        "",
        "For the strict p80 split, the paired chain bootstrap (seeds fixed) gives "
        f"{p80['chain_bootstrap_fixed_seeds']['ci95'][0]:+.3f} to "
        f"{p80['chain_bootstrap_fixed_seeds']['ci95'][1]:+.3f} MCC "
        f"(p = {p80['chain_bootstrap_fixed_seeds']['p_two_sided']:.4g}).",
        "",
        "At a fixed 0.5 threshold, the p80 ensemble gain is "
        f"{p80['fixed_threshold_0.5']['average']['mean_difference']:+.3f} MCC "
        f"(p = {p80['fixed_threshold_0.5']['average']['p']:.4g}), so validation "
        "threshold selection contributes to the primary result.",
        "",
        "The gain is concentrated in JL10 and RB9; TE18 and TL12 do not show a "
        "significant DUET advantage. See `DUET_VS_COBRA_P80.md` for the full table.",
        "",
        "On p80, the graph tower alone averages "
        f"{p80['summaries']['p80_graph']['average']['mean']:.3f} MCC, versus "
        f"{p80['summaries']['DUET ensemble']['average']['mean']:.3f} for the ensemble "
        f"(paired difference {p80['component_tests']['ensemble_vs_p80_graph']['mean_difference']:+.3f}, "
        f"p = {p80['component_tests']['ensemble_vs_p80_graph']['p']:.4g}). "
        "The ensemble is not better than the graph tower on this split, although "
        "the two are statistically indistinguishable.",
        "",
        "## Structural data on the three shared benchmarks",
        "",
        "| Method | Mean MCC | SD |",
        "|:--|--:|--:|",
    ]
    for name in ["GRASSP", "DUET", "CoBRA-M3"]:
        summary = grassp["summaries"][name]["average"]
        lines.append(f"| {name} | {summary['mean']:.3f} | {summary['sd']:.3f} |")
    lines += [
        "",
        f"GRASSP minus DUET: "
        f"{grassp['comparisons']['GRASSP_vs_DUET']['mean_difference']:+.3f} MCC "
        f"(p = {grassp['comparisons']['GRASSP_vs_DUET']['p']:.4g}, "
        f"{grassp['comparisons']['GRASSP_vs_DUET']['seeds_first_better']}/10 seeds).",
        f"GRASSP minus the DUET graph tower: "
        f"{grassp['comparisons']['GRASSP_vs_DUET_graph']['mean_difference']:+.3f} MCC "
        f"(p = {grassp['comparisons']['GRASSP_vs_DUET_graph']['p']:.4g}, "
        f"{grassp['comparisons']['GRASSP_vs_DUET_graph']['seeds_first_better']}/10 seeds).",
        f"GRASSP minus CoBRA: "
        f"{grassp['comparisons']['GRASSP_vs_CoBRA']['mean_difference']:+.3f} MCC "
        f"(p = {grassp['comparisons']['GRASSP_vs_CoBRA']['p']:.4g}, "
        f"{grassp['comparisons']['GRASSP_vs_CoBRA']['seeds_first_better']}/10 seeds).",
        "",
        "This comparison uses TE18, TL12 and JL10 only. The released GRASSP graphs "
        "do not include an RB9 test set with aligned labels. The structural advantage "
        "remains at a fixed 0.5 threshold.",
        "",
        "## True 3D structure inside the DUET graph tower",
        "",
        "| Graph tower | Mean MCC | SD |",
        "|:--|--:|--:|",
    ]
    for name in [
        "seq+ss graph",
        "seq+3D graph",
        "seq+3D shuffled graph",
        "seq+ss+3D graph",
        "GRASSP",
    ]:
        summary = structure_tower["summaries"][name]
        lines.append(f"| {name} | {summary['mean']:.3f} | {summary['sd']:.3f} |")
    lines += [
        "",
        f"Adding 3D edges to the graph tower changes MCC by "
        f"{structure_tower['comparisons']['seq3D_vs_seq']['mean_difference']:+.3f} "
        f"(p = {structure_tower['comparisons']['seq3D_vs_seq']['p']:.4g}). "
        f"Adding MFE and 3D relations changes it by "
        f"{structure_tower['comparisons']['seqss3D_vs_seq']['mean_difference']:+.3f} "
        f"(p = {structure_tower['comparisons']['seqss3D_vs_seq']['p']:.4g}).",
        "",
        "At a fixed 0.5 threshold the 3D-only gain is "
        f"{structure_tower['comparisons_at_0.5']['seq3D_vs_seq']['mean_difference']:+.3f} "
        f"(p = {structure_tower['comparisons_at_0.5']['seq3D_vs_seq']['p']:.4g}).",
        "",
        "The true 3D topology beats degree-matched shuffled 3D edges by "
        f"{structure_tower['comparisons']['seq3D_vs_shuffled']['mean_difference']:+.3f} MCC "
        f"(p = {structure_tower['comparisons']['seq3D_vs_shuffled']['p']:.4g}; "
        f"{structure_tower['comparisons']['seq3D_vs_shuffled']['seeds_first_better']}/10 seeds). "
        "At a fixed 0.5 threshold the difference is "
        f"{structure_tower['comparisons_at_0.5']['seq3D_vs_shuffled']['mean_difference']:+.3f} "
        f"(p = {structure_tower['comparisons_at_0.5']['seq3D_vs_shuffled']['p']:.4g}).",
        "",
        "Even with true 3D edges, the graph tower remains below GRASSP: "
        f"{structure_tower['comparisons']['seq3D_vs_grassp']['mean_difference']:+.3f} MCC "
        f"(p = {structure_tower['comparisons']['seq3D_vs_grassp']['p']:.4g}).",
        "",
        "## Training-set size control",
        "",
        "| Split | Train chains | GRASSP | DUET | Delta (G-D) | p |",
        "|:--|--:|--:|--:|--:|--:|",
        f"| exact-sequence unique | 160 | "
        f"{grassp_exact['summaries']['GRASSP']['average']['mean']:.3f} | "
        f"{grassp_exact['summaries']['DUET']['average']['mean']:.3f} | "
        f"{grassp_exact['comparisons']['GRASSP_vs_DUET']['mean_difference']:+.3f} | "
        f"{grassp_exact['comparisons']['GRASSP_vs_DUET']['p']:.4g} |",
        f"| homology p80 | 79 | "
        f"{grassp['summaries']['GRASSP']['average']['mean']:.3f} | "
        f"{grassp['summaries']['DUET']['average']['mean']:.3f} | "
        f"{grassp['comparisons']['GRASSP_vs_DUET']['mean_difference']:+.3f} | "
        f"{grassp['comparisons']['GRASSP_vs_DUET']['p']:.4g} |",
        "",
        "Both rows use ten seeds. The structural gap falls from +0.107 at 79 training "
        "chains to +0.035 at 160 training chains. The exact-split difference remains "
        "significant under validation-selected thresholds, but not at a fixed 0.5 "
        "threshold (p=0.085). This supports a substantial training-set-size effect.",
        "",
    ]
    path = ROOT / "results" / "training" / "SUMMARY.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
