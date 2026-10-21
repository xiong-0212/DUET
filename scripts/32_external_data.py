from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

import io

import numpy as np
from Bio import Align
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, 'src')) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

import data_loader as dl

SPLITS = {'hariboss_test': 'HARIBOSS_test.txt', 'external_jiang': 'External_Jiang.txt',
          'time_dependent': 'Time_Dependent.txt'}
THRESHOLDS = [1.0, 0.9, 0.8, 0.7]
CACHE = os.path.join(ROOT, 'data', 'external', 'fasta_cache.json')
CIF_CACHE = os.path.join(ROOT, 'data', 'external', 'cif_cache.json')
OUT_JSON = os.path.join(ROOT, 'data', 'external', 'external_sets.json')
OUT_AUDIT = os.path.join(ROOT, 'results', 'external_data_audit.json')
OUT_MD = os.path.join(ROOT, 'docs', 'tables', 'external_dataset_audit_2026-10-16.md')

aligner = Align.PairwiseAligner()
aligner.mode = 'global'
aligner.match_score = 1.0
aligner.mismatch_score = -1.0
aligner.open_gap_score = -4.0
aligner.extend_gap_score = -1.0


def identity(a, b):
    la, lb = len(a), len(b)
    if min(la, lb) / max(la, lb) < 0.8:
        return float('nan')
    try:
        aln = aligner.align(a, b)[0]
    except Exception:
        return float('nan')
    m = 0
    for (sa, ea), (sb, eb) in zip(aln.aligned[0], aln.aligned[1]):
        m += sum(1 for x, y in zip(a[sa:ea], b[sb:eb]) if x == y)
    return m / min(la, lb)


def fetch_fasta(pdb_id, cache):
    if pdb_id in cache:
        return cache[pdb_id]
    url = f'https://www.rcsb.org/fasta/entry/{pdb_id}'
    try:
        txt = urllib.request.urlopen(url, timeout=45).read().decode()
    except Exception as e:
        print(f'  !! {pdb_id}: {e}')
        txt = ''
    cache[pdb_id] = txt
    json.dump(cache, open(CACHE, 'w', encoding='utf-8'))
    time.sleep(0.25)
    return txt


def parse_fasta(txt):
    
    out = []
    names, seq = None, []
    for line in txt.splitlines():
        if line.startswith('>'):
            if names:
                out.append((names, ''.join(seq)))
            head = line[1:].split('|')
            chain_field = next((x for x in head if x.lower().startswith('chain')), '')
            names = [c.strip() for c in chain_field.replace('Chains', '').replace('Chain', '')
                     .replace(',', ' ').split() if c.strip()]
            seq = []
        elif line.strip():
            seq.append(line.strip())
    if names:
        out.append((names, ''.join(seq)))
    return out


def fetch_cif(pdb_id, cache):
    
    if pdb_id in cache:
        return cache[pdb_id]
    try:
        txt = urllib.request.urlopen(f'https://files.rcsb.org/download/{pdb_id}.cif',
                                     timeout=90).read().decode('utf-8', 'replace')
    except Exception as e:
        print(f'  !! cif {pdb_id}: {e}')
        txt = ''
    cache[pdb_id] = txt
    json.dump(cache, open(CIF_CACHE, 'w', encoding='utf-8'))
    time.sleep(0.3)
    return txt


def cif_chains(txt):
    
    if not txt:
        return []
    d = MMCIF2Dict(io.StringIO(txt))
    try:
        asym = d['_atom_site.auth_asym_id']
        seqid = d['_atom_site.label_seq_id']
        comp = d['_atom_site.label_comp_id']
    except KeyError:
        return []
    per, order = {}, []
    for c, s_, comp_id in zip(asym, seqid, comp):
        key = (c, s_)
        if key in per:
            continue
        per[key] = comp_id
        if c not in order:
            order.append(c)
    out = []
    for c in order:
        res = [per[(c, s_)] for (cc, s_) in per if cc == c]
        std = [x for x in res if x in ('A', 'C', 'G', 'U')]
        if std:
            out.append((c, ''.join(std)))
    return out


def audit_against(records, ref, label):
    
    out = {}
    for cid, seq in records.items():
        best, best_to = 0.0, None
        for rid, rseq in ref.items():
            v = identity(seq, rseq)
            if not np.isnan(v) and v > best:
                best, best_to = v, rid
        out[cid] = (best, best_to)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--root',
        default=os.path.join(ROOT, 'third_party', 'SMARTPocket-main'),
        help='path to the SMARTPocket-main checkout',
    )
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    cache = json.load(open(CACHE, encoding='utf-8')) if os.path.exists(CACHE) else {}
    cif_cache = (json.load(open(CIF_CACHE, encoding='utf-8'))
                 if os.path.exists(CIF_CACHE) else {})


    ours = dl.load_all()
    ref_train = {s['rna_id']: s['sequence'] for s in ours['train']}
    ref_train.update({s['rna_id']: s['sequence'] for s in ours['val']})
    ref_bench = {s['rna_id']: s['sequence']
                 for sp in ['TE18', 'TL12', 'JL10', 'RB9'] for s in ours[sp]}
    print(f'reference: {len(ref_train)} train+val chains, {len(ref_bench)} benchmark chains')

    audit = {'reference': {'train_val': len(ref_train), 'benchmarks': len(ref_bench)},
             'splits': {}}
    payload = {}
    for name, fname in SPLITS.items():
        path = os.path.join(a.root, 'data', 'splits', fname)
        chains = [l.strip() for l in open(path, encoding='utf-8') if l.strip()]
        contacts = json.load(open(os.path.join(a.root, 'data', 'contacts',
                                               f'contacts_{"hariboss" if name == "hariboss_test" else name}.json'),
                                  encoding='utf-8'))
        contacts = {k.lower(): v for k, v in contacts.items()}
        print(f'\n[{name}] {len(chains)} chains in the split list, '
              f'{len(contacts)} labelled entries')
        recs = {}
        for c in chains:
            pdb, _, chain = c.partition('_')
            lab = contacts.get(pdb.lower())
            if lab is None:
                print(f'  skip {c}: no labels')
                continue
            seq, lab_chain = None, None

            cif = cif_chains(fetch_cif(pdb, cif_cache))
            total = sum(len(x[1]) for x in cif)
            for ch_, s_ in cif:
                if len(lab) == len(s_) and ch_ == chain:
                    seq, lab_chain = s_, lab
                    break
                if len(lab) == total and ch_ == chain:
                    off = sum(len(x[1]) for x in cif[:cif.index((ch_, s_))])
                    seq, lab_chain = s_, lab[off:off + len(s_)]
                    break
            if seq is not None:
                recs[c] = {'sequence': seq, 'labels': [int(x) for x in lab_chain]}
                continue

            txt = fetch_fasta(pdb, cache)
            records = parse_fasta(txt)
            if not records:
                print(f'  skip {c}: no sequence')
                continue
            for chains, s_ in records:
                if len(lab) == len(s_):
                    if chain in chains or len(chains) == 1 or len(records) == 1:
                        seq, lab_chain = s_, lab
                        break
                elif len(lab) == len(s_) * len(chains):
                    if chain in chains:
                        i = chains.index(chain)
                        seq = s_
                        lab_chain = lab[i * len(s_):(i + 1) * len(s_)]
                        break
            if seq is None:
                print(f'  skip {c}: no length-consistent chain '
                      f'(1 seq x {len(records)} records, labels {len(lab)})')
                continue
            recs[c] = {'sequence': seq, 'labels': [int(x) for x in lab_chain]}


        keep, dropped = [], {}
        for cid in recs:
            hit = None
            for k in keep:
                v = identity(recs[cid]['sequence'], recs[k]['sequence'])
                if not np.isnan(v) and v >= 0.8:
                    hit = (k, round(float(v), 3))
                    break
            if hit:
                dropped[cid] = hit
            else:
                keep.append(cid)

        vs_train = audit_against({c: recs[c]['sequence'] for c in keep}, ref_train, 'train')
        vs_bench = audit_against({c: recs[c]['sequence'] for c in keep}, ref_bench, 'bench')
        clean = [c for c in keep if vs_train[c][0] < 0.8]
        removed = {c: {'pid': round(vs_train[c][0], 3), 'to': vs_train[c][1]}
                   for c in keep if vs_train[c][0] >= 0.8}

        hits = {f'p{int(t * 100)}': sum(1 for c in keep if vs_train[c][0] >= t) for t in THRESHOLDS}
        hits_b = {f'p{int(t * 100)}': sum(1 for c in keep if vs_bench[c][0] >= t) for t in THRESHOLDS}
        n_pos = sum(sum(recs[c]['labels']) for c in clean)
        n_res = sum(len(recs[c]['labels']) for c in clean)
        audit['splits'][name] = {
            'n_listed': len(chains), 'n_built': len(recs), 'n_after_internal_dedup': len(keep),
            'internal_duplicates': dropped, 'vs_train_val': {'hits': hits, 'removed': removed},
            'vs_benchmarks': {'hits': hits_b}, 'n_final': len(clean),
            'n_residues': n_res, 'positive_ratio': round(n_pos / n_res, 3) if n_res else 0.0}
        payload[name] = {c: recs[c] for c in clean}
        print(f'  built {len(recs)} | internal dups {len(dropped)} | vs train/val >=80 %: '
              f'{len(removed)} | final {len(clean)} chains, {n_res} residues, '
              f'positive ratio {audit["splits"][name]["positive_ratio"]}')
        print(f'  identity to train/val: ' + ', '.join(f'>= {k[1:]}%: {v}' for k, v in hits.items()))
        print(f'  identity to our benchmarks: ' + ', '.join(f'>= {k[1:]}%: {v}' for k, v in hits_b.items()))

    json.dump(payload, open(OUT_JSON, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    json.dump(audit, open(OUT_AUDIT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    lines = ['# External dataset audit (SMARTPocket release, 2026-10-16)', '',
             '> Sequences come from RCSB; labels come from the SMARTPocket contacts JSON; the identity criterion matches the main text',
             '> (global alignment, length ratio ≥0.8, identity = number of matches / length of the shorter sequence).', '',
             '| Dataset | Listed | Built | After internal dedup | ≥80% vs train/val | **Final** | Residues | Positive ratio |',
             '|:--|--:|--:|--:|--:|--:|--:|--:|']
    for name, d in audit['splits'].items():
        lines.append(f"| {name} | {d['n_listed']} | {d['n_built']} | {d['n_after_internal_dedup']} | "
                     f"{len(d['vs_train_val']['removed'])} | **{d['n_final']}** | {d['n_residues']} | "
                     f"{d['positive_ratio']:.3f} |")
    lines += ['', '> Identity distribution against the train/validation sets (after internal deduplication):', '']
    for name, d in audit['splits'].items():
        lines.append(f"- {name}: " + ', '.join(f"≥{k[1:]}% {v}" for k, v in
                                               d['vs_train_val']['hits'].items()))
    lines += ['', '> Identity distribution against our four benchmarks (to judge whether the two test suites are independent):', '']
    for name, d in audit['splits'].items():
        lines.append(f"- {name}: " + ', '.join(f"≥{k[1:]}% {v}" for k, v in
                                               d['vs_benchmarks']['hits'].items()))
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    open(OUT_MD, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    print(f'\n[written] {OUT_JSON}\n[written] {OUT_AUDIT}\n[written] {OUT_MD}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
