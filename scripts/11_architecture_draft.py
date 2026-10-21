from __future__ import annotations

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE


while not os.path.isdir(os.path.join(ROOT, 'src')) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)

C_MAIN = '#1D9E8F'
C_ALT = '#4c72b0'
C_GREY = '#8c8c8c'
C_BG = '#F4F7F6'


def box(ax, x, y, w, h, text, fc='white', ec=C_GREY, fs=7.2, bold=False, ls='-'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.006,rounding_size=0.012',
                                fc=fc, ec=ec, lw=0.9, linestyle=ls, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs,
            zorder=3, fontweight='bold' if bold else 'normal', linespacing=1.35)


def arrow(ax, p1, p2, color=C_GREY, ls='-', lw=1.0, style='-|>'):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=8,
                                 color=color, lw=lw, linestyle=ls, zorder=1,
                                 shrinkA=1, shrinkB=1))


def main():
    fig, ax = plt.subplots(figsize=(7.6, 5.0), dpi=300)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    ax.add_patch(FancyBboxPatch((0.008, 0.40), 0.984, 0.58, boxstyle='round,pad=0.01',
                                fc=C_BG, ec='#c9d6d3', lw=0.8, zorder=0))
    ax.text(0.016, 0.965, 'A · single model: PRISM-Fusion (representation-level fusion)',
            fontsize=8, fontweight='bold', color='#2b5f58', va='top')
    ax.add_patch(FancyBboxPatch((0.008, 0.03), 0.984, 0.31, boxstyle='round,pad=0.01',
                                fc='#fbfbfb', ec='#d8d8d8', lw=0.8, zorder=0))
    ax.text(0.016, 0.325, 'B · final model: PRISM-Ens (decision-level ensemble)',
            fontsize=8, fontweight='bold', color='#333333', va='top')


    box(ax, 0.025, 0.62, 0.10, 0.16, 'RNA sequence\n(A/C/G/U)', fc='white', ec=C_MAIN, fs=7.4,
        bold=True)


    box(ax, 0.165, 0.76, 0.155, 0.135,
        'RNA-FM  (frozen)\n640-d per nucleotide', fc='white', ec=C_MAIN, fs=7.0)
    box(ax, 0.165, 0.50, 0.155, 0.135,
        'RiNALMo-giga  (frozen)\n1280-d per nucleotide', fc='white', ec=C_ALT, fs=7.0)
    arrow(ax, (0.132, 0.735), (0.163, 0.815), color=C_MAIN)
    arrow(ax, (0.132, 0.66), (0.163, 0.575), color=C_ALT)


    box(ax, 0.355, 0.815, 0.115, 0.095, 'RNAfold\n(sequence only)', fc='#ffffff', ec=C_GREY, fs=6.8)
    box(ax, 0.355, 0.60, 0.135, 0.185,
        'Graph construction\n• seq adjacency\n• MFE base pairs', fc='white', ec=C_MAIN, fs=6.9)
    arrow(ax, (0.132, 0.70), (0.352, 0.862), color=C_GREY, ls=':')
    arrow(ax, (0.322, 0.83), (0.352, 0.80), color=C_GREY)
    arrow(ax, (0.300, 0.79), (0.352, 0.72), color=C_MAIN)


    box(ax, 0.548, 0.70, 0.165, 0.16,
        'Graph tower\nR-GCN ×2 (GraphNorm,\nresidual) — 386k', fc='white', ec=C_MAIN, fs=6.9)
    box(ax, 0.548, 0.47, 0.165, 0.15,
        'Sequence tower\nper-residue MLP\n1024–256–128 — 1.65M', fc='white', ec=C_ALT, fs=6.9)
    arrow(ax, (0.494, 0.72), (0.545, 0.78), color=C_MAIN)
    arrow(ax, (0.322, 0.70), (0.545, 0.545), color=C_ALT)


    box(ax, 0.748, 0.575, 0.115, 0.155,
        'Fusion\nequal weight or\ngated  $h=g h_1+(1{-}g)h_2$', fc='white', ec='#c44e52', fs=6.8)
    arrow(ax, (0.717, 0.78), (0.815, 0.74), color=C_MAIN)
    arrow(ax, (0.717, 0.545), (0.815, 0.575), color=C_ALT)

    box(ax, 0.888, 0.60, 0.10, 0.11, 'Classifier\nhead', fc='white', ec=C_GREY, fs=6.9)
    arrow(ax, (0.865, 0.655), (0.886, 0.655))


    box(ax, 0.20, 0.10, 0.20, 0.14,
        'Graph-tower branch\n→ per-residue logits', fc='white', ec=C_MAIN, fs=7.0)
    box(ax, 0.46, 0.10, 0.20, 0.14,
        'MLP-tower branch\n→ per-residue logits', fc='white', ec=C_ALT, fs=7.0)
    arrow(ax, (0.30, 0.42), (0.30, 0.245), color=C_MAIN, ls='--')
    arrow(ax, (0.56, 0.42), (0.56, 0.245), color=C_ALT, ls='--')
    box(ax, 0.70, 0.10, 0.12, 0.14, 'Probability\naverage', fc='white', ec='#c44e52', fs=6.9)
    arrow(ax, (0.402, 0.17), (0.697, 0.17), color=C_GREY)
    arrow(ax, (0.662, 0.17), (0.697, 0.17), color=C_GREY)

    box(ax, 0.672, 0.155, 0.20, 0.155,
        'Output\nper-nucleotide\nbinding probability\n+ confidence', fc='white', ec=C_GREY, fs=6.8)
    arrow(ax, (0.607, 0.23), (0.670, 0.232))
    ax.text(0.672, 0.128, 'green = binding   red = non-binding', fontsize=6.5, color=C_GREY)

    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(ROOT, 'figures', f'Fig1_architecture_draft.{ext}'),
                    bbox_inches='tight')
    print('[written] figures/Fig1_architecture_draft.png/pdf')


if __name__ == '__main__':
    main()
