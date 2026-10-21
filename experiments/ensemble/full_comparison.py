from __future__ import annotations

import csv
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE


while not os.path.isdir(os.path.join(ROOT, 'src')) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
REPO = os.path.dirname(ROOT)
T = ['TE18', 'TL12', 'JL10', 'RB9']
METRICS = [('MCC', 'MCC'), ('AUC', 'AUC'), ('AUPRC', 'AUPRC'), ('F1', 'F1'),
           ('Precision', 'Precision'), ('Recall', 'Recall'), ('GMean', 'GMean'),
           ('Accuracy', 'Accuracy')]

PAPER_KEYS = {'MCC': 'MCC', 'AUC': 'AUROC', 'Precision': 'Precision', 'Recall': 'Recall'}


def from_results(ds, tag):
    
    rows = []
    for f in glob.glob(os.path.join(ROOT, 'results', 'controlled',
                                    f'train_{ds}_{tag}_seed*.json')):
        rows.append(json.load(open(f, encoding='utf-8'))['test_metrics'])
    if not rows:
        return None
    out = {}
    for mkey, _ in METRICS:
        out[mkey] = {}
        for split in T + ['_avg']:
            vals = []
            for r in rows:
                if split == '_avg':
                    vals.append(float(np.mean([r[s]['at_threshold'][mkey] for s in T])))
                elif split in r:
                    vals.append(float(r[split]['at_threshold'][mkey]))
            if vals:
                out[mkey][split] = (float(np.mean(vals)),
                                    float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
    return out


def from_m3(ds):
    p = os.path.join(ROOT, 'results', 'cobra_m3_equiv.json')
    if not os.path.exists(p):
        return None
    rows = json.load(open(p, encoding='utf-8'))[ds]['bce']
    out = {}
    for mkey, _ in METRICS:
        out[mkey] = {}
        for split in T + ['_avg']:
            vals = []
            for r in rows:
                if split == '_avg':
                    vals.append(float(np.mean([r['test'][s]['at_threshold'][mkey]
                                               for s in T])))
                elif split in r['test']:
                    vals.append(float(r['test'][split]['at_threshold'][mkey]))
            if vals:
                out[mkey][split] = (float(np.mean(vals)),
                                    float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
    return out


def from_paper():
    p = os.path.join(REPO, 'cobra_repro', 'results.json')
    d = json.load(open(p, encoding='utf-8'))['paper_M3']
    out = {}
    for mkey, pkey in PAPER_KEYS.items():
        out[mkey] = {}
        vals = []
        for split in T:
            v = d.get(split, {}).get(pkey)
            if v is not None:
                out[mkey][split] = (float(v), None)
                vals.append(float(v))
        if vals:
            out[mkey]['_avg'] = (float(np.mean(vals)), None)
    return out


def from_repro():
    p = os.path.join(REPO, 'cobra_repro', 'results.json')
    d = json.load(open(p, encoding='utf-8'))['reproduced']
    keymap = {'MCC': 'MCC', 'AUC': 'AUROC', 'AUPRC': 'AUPRC', 'F1': 'F1',
              'Precision': 'Precision', 'Recall': 'Recall'}
    out = {}
    for mkey, rkey in keymap.items():
        out[mkey] = {}
        vals = []
        for split in T:
            v = d.get(split, {}).get(rkey)
            if v is not None:
                out[mkey][split] = (float(v), None)
                vals.append(float(v))
        if vals:
            out[mkey]['_avg'] = (float(np.mean(vals)), None)
    return out


def fmt(v, std):
    if std is None:
        return f'{v:.3f}'
    return f'{v:.3f}±{std:.3f}' if std > 0 else f'{v:.3f}'


def method_table(methods, metrics_note=''):
    
    lines = []
    for mkey, mlab in METRICS:

        if not any(d and mkey in d and d[mkey] for _, d in methods):
            continue
        lines.append(f'\n### {mlab}\n')
        lines.append('| test set | ' + ' | '.join(lab for lab, _ in methods) + ' |')
        lines.append('|:---|' + '---:|' * len(methods))
        for split, sname in list(zip(T, T)) + [('_avg', '**mean (4 sets)**')]:
            cells = []
            for _, d in methods:
                if d and mkey in d and split in d[mkey]:
                    v, s = d[mkey][split]
                    cells.append(fmt(v, s))
                else:
                    cells.append('—')
            lines.append(f'| {sname} | ' + ' | '.join(cells) + ' |')
    return lines


def main():

    clean = [
        ('CoBRA-M3 equivalent\n(RiNALMo+MLP+BCE)', from_m3('clean')),
        ('**Ours seq+ss**\n(RNA-FM+R-GCN)', from_results('clean', 'seq_ss')),
        ('Ours main model\n(seq+ss+ss_alt)', from_results('clean', 'seq_ss_alt')),
        ('MLP (RNA-FM)', from_results('clean', 'mlp')),
        ('Ours seq+ss\n(RiNALMo backbone)', from_results('clean', 'rinalmo_ss')),
    ]

    orig = [
        ('CoBRA-M3\n(paper values)', from_paper()),
        ('CoBRA-M3\n(reproduced with public weights)', from_repro()),
        ('CoBRA-M3 equivalent\n(retrained)', from_m3('orig')),
        ('**Ours seq+ss**', from_results('orig', 'seq_ss')),
        ('Ours main model\n(seq+ss+ss_alt)', from_results('orig', 'seq_ss_alt')),
        ('MLP (RNA-FM)', from_results('orig', 'mlp')),
    ]

    md = ['# All-metric comparison: CoBRA vs ours (2026-10-11)', '',
          '> All numbers are generated from result files: `results/controlled/` (5 seeds, mean±std, validation-selected threshold),',
          '> `results/cobra_m3_equiv.json`, `cobra_repro/results.json` (paper values + reproduction with public weights, fixed threshold 0.5).',
          '> **Test sets are deduplicated**: RB9 goes from 9 to 6 chains (1nem_A/1q8n_A/2tob_A duplicate TE18 and are kept in TE18).',
          '> The paper-value and reproduction rows report only the 4 metrics of the original paper protocol; all other metrics are shown as —.', '',
          '## 1. Cleaned (leakage removed, train=314 / val=40) — main results', '']
    md += method_table(clean)
    md += ['', '---', '', '## 2. Pre-cleaning (original benchmark, train=345 / val=40)', '']
    md += method_table(orig)

    out_md = os.path.join(ROOT, 'docs', 'analysis', 'A+_all_metrics_comparison.md')
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md) + '\n')


    csv_path = os.path.join(ROOT, 'results', 'full_comparison.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['setting', 'method', 'split', 'metric', 'mean', 'std'])
        for setting, methods in [('clean', clean), ('orig', orig)]:
            for lab, d in methods:
                if not d:
                    continue
                lab1 = lab.replace('\n', ' ')
                for mkey, _ in METRICS:
                    for split in T + ['_avg']:
                        if mkey in d and split in d[mkey]:
                            v, s = d[mkey][split]
                            w.writerow([setting, lab1, split, mkey, f'{v:.4f}',
                                        '' if s is None else f'{s:.4f}'])
    print(f'wrote {out_md}')
    print(f'wrote {csv_path}')


    for setting, methods in [('cleaned', clean), ('pre-cleaning', orig)]:
        print(f'\n===== {setting} =====')
        for mkey in ['MCC', 'AUC']:
            print(f'-- {mkey} (per set / mean)')
            for lab, d in methods:
                if not d or mkey not in d:
                    continue
                cells = '  '.join(
                    f'{s}={fmt(*d[mkey][s])}' if s in d[mkey] else f'{s}=—'
                    for s in T + ['_avg'])
                print(f'   {lab.replace(chr(10)," "):<34}{cells}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
