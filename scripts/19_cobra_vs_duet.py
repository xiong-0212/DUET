from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, 'src')) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
FIG = os.path.join(ROOT, 'figures')

SHARED = '#D6E4F0'
THEIRS = '#D55E00'
OURS = '#0072B2'
GREY = '#4D4D4D'


def box(ax, x, y, w, h, text, fc, ec=GREY, fs=7, lw=0.8, weight='normal'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.012,rounding_size=0.02',
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, zorder=3,
            weight=weight, linespacing=1.35)


def arrow(ax, p, q, color=GREY, style='-|>', lw=1.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=8, lw=lw, color=color,
                                 shrinkA=1, shrinkB=1, zorder=1))


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    plt.rcParams.update({'font.family': 'sans-serif',
                         'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
                         'savefig.bbox': 'tight'})
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')


    ax = axes[0]
    ax.set_title('(a) CoBRA (M3, released weights)', fontsize=9, loc='left')
    box(ax, 0.30, 0.88, 0.40, 0.08, 'RNA sequence', '#F2F2F2')
    box(ax, 0.20, 0.70, 0.60, 0.13,
        'RiNALMo-giga  (frozen)\n1280-d per nucleotide', SHARED)
    box(ax, 0.10, 0.34, 0.80, 0.30,
        'per-residue MLP   (1.62 M parameters)\n'
        '1280 → 1024 → 256 → 128 → 64\n'
        'each: Linear + LayerNorm + ReLU + Dropout 0.1', SHARED)
    box(ax, 0.22, 0.20, 0.56, 0.10, 'Linear(64 → 2)', SHARED)
    box(ax, 0.30, 0.05, 0.40, 0.09, 'softmax → threshold 0.5', THEIRS, ec=THEIRS)
    arrow(ax, (0.5, 0.88), (0.5, 0.83))
    arrow(ax, (0.5, 0.70), (0.5, 0.64))
    arrow(ax, (0.5, 0.34), (0.5, 0.30))
    arrow(ax, (0.5, 0.20), (0.5, 0.14))
    ax.text(0.5, -0.015,
            'trained with cross-entropy (BCE equivalent)\n'
            'on 345 chains, train/test overlap not removed',
            ha='center', va='top', fontsize=6.5, color=THEIRS)


    ax = axes[1]
    ax.set_title('(b) DUET (this study)', fontsize=9, loc='left')
    box(ax, 0.32, 0.88, 0.36, 0.08, 'RNA sequence', '#F2F2F2')
    box(ax, 0.04, 0.68, 0.42, 0.14, 'RNAfold\nMFE base pairs\n(from sequence only)', OURS)
    box(ax, 0.52, 0.68, 0.44, 0.14, 'RNA-FM  (frozen)\n640-d per nucleotide', OURS)
    box(ax, 0.04, 0.42, 0.42, 0.20,
        'graph tower\nR-GCN × 2 on seq + ss edges\n→ head(128→64→1)', OURS)
    box(ax, 0.52, 0.42, 0.44, 0.20,
        'sequence tower\nRiNALMo (frozen) 1280-d\nMLP 1280→1024→256→128→head', SHARED)
    box(ax, 0.20, 0.25, 0.60, 0.09, 'equal-weight average of probabilities', OURS, ec=OURS)
    box(ax, 0.30, 0.10, 0.40, 0.09, 'threshold selected on validation', OURS, ec=OURS)
    arrow(ax, (0.5, 0.88), (0.25, 0.82))
    arrow(ax, (0.5, 0.88), (0.74, 0.82))
    arrow(ax, (0.25, 0.68), (0.25, 0.62))
    arrow(ax, (0.74, 0.68), (0.74, 0.62))
    arrow(ax, (0.25, 0.42), (0.38, 0.34))
    arrow(ax, (0.74, 0.42), (0.62, 0.34))
    arrow(ax, (0.5, 0.25), (0.5, 0.19))
    ax.text(0.5, 0.045,
            'TCL-focal loss (class-balanced focal + triplet-centre)\n'
            'trained on 314 de-leaked chains',
            ha='center', va='top', fontsize=6.5, color=OURS)

    fig.suptitle('CoBRA (M3) versus DUET: the sequence tower is the same network, '
                 'the loss and the training data differ', fontsize=9.5, y=1.04)
    os.makedirs(FIG, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG, f'FigS_cobra_vs_duet.{ext}'), dpi=400)
    plt.close(fig)
    print('[written] figures/FigS_cobra_vs_duet.pdf / .png')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
