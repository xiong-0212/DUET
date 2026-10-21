from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE


while not os.path.isdir(os.path.join(ROOT, 'src')) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, HERE)

from data_loader import load_all

SS_INPUT = os.path.join(ROOT, 'data', 'raw', 'ss_input.fa')
SS_OUTPUT = os.path.join(ROOT, 'data', 'raw', 'ss_output.tsv')


def prepare_input(out_path: str = SS_INPUT) -> str:
    
    data = load_all()
    seen = set()
    n = 0
    with open(out_path, 'w', newline='\n') as f:
        for split, samples in data.items():
            for s in samples:
                rid = s['rna_id']
                if rid in seen:
                    continue
                seen.add(rid)
                f.write(f'>{rid}\n{s["sequence"]}\n')
                n += 1
    print(f'wrote {n} sequences -> {out_path}')
    return out_path


def load_folds(path: str = SS_OUTPUT, expected_len: dict = None,
               verbose: bool = False) -> dict:
    
    if not os.path.exists(path):
        return {}
    folds = {}
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) < 2:
                parts = line.split()
            if len(parts) < 2:
                continue
            rid, db = parts[0], parts[1]
            folds[rid] = db

    if expected_len:
        bad = {r: (len(d), expected_len[r]) for r, d in folds.items()
               if r in expected_len and len(d) != expected_len[r]}
        missing = [r for r in expected_len if r not in folds]
        if verbose or bad or missing:
            print(f'[ss] {path}: {len(folds)} entries; {len(bad)} length mismatches; {len(missing)} missing')
            for r, v in list(bad.items())[:5]:
                print(f'     [length mismatch] {r}: dot-bracket {v[0]} vs sequence {v[1]}')
            for r in missing[:5]:
                print(f'     [missing] {r}')
    return folds


def pairs_from_dotbracket(db: str) -> list:
    
    stack, out = [], []
    for i, c in enumerate(db):
        if c == '(':
            stack.append(i)
        elif c == ')':
            if stack:
                out.append((stack.pop(), i))
    return out



def load_bp_prob(path: str, prob_min: float = 0.1) -> dict:
    
    bp = {}
    with open(path, encoding='utf-8') as f:
        f.readline()
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 4:
                continue
            rid, i, j, p = parts[0], int(parts[1]) - 1, int(parts[2]) - 1, float(parts[3])
            if p < prob_min:
                continue
            bp.setdefault(rid, {})[(min(i, j), max(i, j))] = p
    return bp
