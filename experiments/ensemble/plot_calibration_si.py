from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "src")) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)

sys.path.insert(0, os.path.join(ROOT, "scripts"))
from audit_panel_alignment import require_matplotlib_panel_alignment

FIG = os.path.join(ROOT, "figures")
BLUE, ORANGE, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#4D4D4D"

plt.rcParams.update({
    "font.size": 8,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def panel_label(ax, letter, title):
    ax.text(-0.16, 1.02, letter, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9, fontweight="bold")
    ax.set_title(title, loc="left", pad=7)


def save(fig, name):
    require_matplotlib_panel_alignment(
        fig,
        json_out=os.path.join(FIG, f"{name}.alignment.json"),
        overlay_svg=os.path.join(FIG, f"{name}.alignment.svg"),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )
    fig.savefig(os.path.join(FIG, f"{name}.pdf"))
    fig.savefig(os.path.join(FIG, f"{name}.svg"))
    fig.savefig(os.path.join(FIG, f"{name}.png"), dpi=600)
    fig.savefig(os.path.join(FIG, f"{name}.tiff"), dpi=600)
    plt.close(fig)


def main():
    with open(os.path.join(ROOT, "results", "calibration.json"), encoding="utf-8") as handle:
        data = json.load(handle)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.20, top=0.88, wspace=0.46)

    ax = axes[0]
    reliability = data["reliability"]
    for key, label, color in [
        ("single_uncal", "Single model", GREY),
        ("ensemble", "DUET ensemble", GREEN),
    ]:
        bins = [b for b in reliability[key]["bins"] if b["n"] > 0]
        x = np.array([b["conf"] for b in bins])
        y = np.array([b["acc"] for b in bins])
        n = np.array([b["n"] for b in bins], dtype=float)
        ax.plot(x, y, "-", lw=1.1, color=color, label=label)
        ax.scatter(x, y, s=np.clip(n / n.max() * 34, 7, 34), color=color, zorder=3)
    ax.plot([0, 1], [0, 1], "--", color=GREY, lw=0.9, label="Perfect calibration")
    ax.set_xlim(0.15, 0.85)
    ax.set_ylim(0.05, 0.85)
    ax.set_yticks(np.arange(0.2, 0.9, 0.1))
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Empirical positive rate")
    ax.legend(loc="lower right", fontsize=6.8)
    panel_label(ax, "a", "Reliability")

    ax = axes[1]
    rows = sorted(data["result"]["selective"]["pooled"], key=lambda row: row["coverage"])
    coverage = np.array([row["coverage"] for row in rows])
    for key, label, marker, color in [
        ("MCC", "MCC", "o", BLUE),
        ("Precision", "Precision", "s", ORANGE),
        ("Recall", "Sensitivity (recall)", "^", GREEN),
    ]:
        values = np.array([row[key] for row in rows])
        ax.plot(coverage, values, marker=marker, ms=3.2, lw=1.1, color=color, label=label)
    ax.set_xlim(0.15, 1.05)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Retained fraction of residues")
    ax.set_ylabel("Metric value")
    ax.legend(loc="lower left", fontsize=6.8)
    panel_label(ax, "b", "Selective prediction")

    save(fig, "FigS1_calibration")
    print("[written] figures/FigS1_calibration.pdf / .svg / .png / .tiff")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
