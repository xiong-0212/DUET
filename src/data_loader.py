from __future__ import annotations

import csv
import json
import os
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE


while not os.path.isdir(os.path.join(ROOT, 'src')) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)

COBRA_DIR = os.path.join(ROOT, 'data', 'raw', 'cobra')

MAX_LEN = 161


TEST_SETS = OrderedDict([
    ('TE18', 'TE18/TE18-bindingsite.txt'),
    ('TL12', 'TL12/TL12-bindingsite.txt'),
    ('JL10', 'JL10/JL10-bindingsite.txt'),
    ('RB9',  'RB9/RB9-bindingsite.txt'),
])

TRAIN_SPLIT = 'Hariboss_TR60/Hariboss_TR60-bindingsite.txt'








DROP_FROM_SPLIT = {'RB9': ['1nem_A', '1q8n_A', '2tob_A']}


def parse_bindingsite(path: str) -> dict:
    

    lines = [l.strip() for l in open(path, encoding='utf-8-sig',
                                     errors='replace').read().split('\n')]
    lines = [l for l in lines if l]
    if len(lines) % 3 != 0:
        raise ValueError(f'{path}: number of non-empty lines {len(lines)} is not a multiple of 3')
    out = {}
    for i in range(0, len(lines), 3):
        hid, seq, lab = lines[i], lines[i + 1], lines[i + 2]
        if not hid.startswith('>'):
            raise ValueError(f'{path}: line {i} is not a header: {hid!r}')
        if len(seq) != len(lab):
            raise ValueError(f'{path}: {hid} sequence length {len(seq)} != label length {len(lab)}')
        out[hid[1:]] = {'seq': seq.upper(), 'labels': lab}
    return out


def _split_ids(split: str) -> list:
    
    path = os.path.join(COBRA_DIR, 'Hariboss_TR60', f'{split}.csv')
    rows = list(csv.DictReader(open(path, encoding='utf-8-sig')))
    return [f"{r['PDB_ID']}_{r['Chain']}" for r in rows]


def sample_from(rid: str, rec: dict, split: str) -> dict:
    pdb, _, chain = rid.partition('_')
    return {
        'rna_id': rid,
        'chain': chain,
        'sequence': rec['seq'],
        'labels': [int(c) for c in rec['labels']],
        'split': split,
    }


def load_all(max_len: int = MAX_LEN, drop_over_length: bool = True) -> dict:
    
    bs = parse_bindingsite(os.path.join(COBRA_DIR, TRAIN_SPLIT))

    out = OrderedDict()
    for split in ('train', 'val'):
        recs = []
        missing = []
        for rid in _split_ids(split):
            if rid not in bs:
                missing.append(rid)
                continue
            s = sample_from(rid, bs[rid], split)
            if drop_over_length and len(s['sequence']) > max_len:
                continue
            recs.append(s)
        out[split] = recs
        if missing:
            print(f'[warn] {split}: {len(missing)} IDs missing from bindingsite')

    recs = []
    for rid in _split_ids('test'):
        if rid in bs:
            s = sample_from(rid, bs[rid], 'hariofficial_test')
            if not (drop_over_length and len(s['sequence']) > max_len):
                recs.append(s)
    out['hariofficial_test'] = recs

    for name, rel in TEST_SETS.items():
        d = parse_bindingsite(os.path.join(COBRA_DIR, rel))
        drop = set(DROP_FROM_SPLIT.get(name, []))
        recs = []
        for rid, rec in d.items():
            if rid in drop:
                continue
            s = sample_from(rid, rec, name)
            if drop_over_length and len(s['sequence']) > max_len:
                continue
            recs.append(s)
        out[name] = recs

    return out


def stats(samples: list) -> dict:
    n_res = sum(len(s['sequence']) for s in samples)
    n_pos = sum(sum(s['labels']) for s in samples)
    return dict(chains=len(samples), residues=n_res, pos=n_pos, neg=n_res - n_pos,
                pos_ratio=(n_pos / n_res if n_res else 0.0))


def availability_report() -> dict:
    data = load_all()
    rep = OrderedDict()
    for k, v in data.items():
        rep[k] = stats(v)
    return rep
