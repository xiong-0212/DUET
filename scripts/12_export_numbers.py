from __future__ import annotations

import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE


while not os.path.isdir(os.path.join(ROOT, 'src')) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)

VALUES = os.path.join(ROOT, 'results', 'table_values.json')
BOOT = os.path.join(ROOT, 'results', 'chain_bootstrap.json')
THREE = os.path.join(ROOT, 'results', 'three_tower_tables.json')
TOWERS = os.path.join(ROOT, 'results', 'tower_analysis.json')
GRASSP = os.path.join(ROOT, 'results', 'grassp_vs_duet.json')
OUT = os.path.join(ROOT, 'paper', 'numbers.tex')

DUET = 'DUET (final model)'
BASE = 'CoBRA-M3 equiv. (retrained baseline)'


def r3(x):
    return f'{x:.3f}'


def s3(x):
    
    return f'{x:+.3f}'

def unique_newcommands(lines):
    
    out = []
    seen = {}
    pattern = re.compile(r'\\newcommand\{\\([A-Za-z0-9@]+)\}')
    for line in lines:
        match = pattern.match(line)
        if not match:
            out.append(line)
            continue
        name = match.group(1)
        if name in seen:
            previous = out[seen[name]]
            if previous != line:
                raise ValueError(
                    f'conflicting definitions for \\{name}:\n{previous}\n{line}'
                )
            continue
        seen[name] = len(out)
        out.append(line)
    return out


def main():
    v = json.load(open(VALUES, encoding='utf-8'))
    lines = ['% ---------------------------------------------------------------------------',
             '% numbers.tex — every reported value, as a LaTeX macro.',
             '%',
             '% GENERATED FILE — do not edit by hand.',
             '%   python scripts/12_export_numbers.py',
             '%',
             '% Source of truth:',
             '%   results/table_values.json        (scripts/07_make_tables.py)      Table 1, S1-S3',
             '%   results/three_tower_tables.json  (scripts/22_three_tower_tables.py)  Table 2, 3',
             '%   results/per_set_metrics.json      (scripts/17_per_set_metrics.py)   Fig. 3',
             '%   results/tower_analysis.json       (scripts/14_tower_analysis.py)    leakage macros',
             '%   results/grassp_vs_duet.json       (baselines/grassp/...)            Table 4',
             '%   results/chain_bootstrap.json      (scripts/13_chain_bootstrap.py)   internal only',
             '%',
             '% All macros are wrapped in \\ensuremath, so they can be used in text or math.',
             '% ---------------------------------------------------------------------------',
             '']





    if os.path.exists(THREE):
        t3 = json.load(open(THREE, encoding='utf-8'))
        lines += ['% -- main results, towers, combination and capacity controls (Table 2, 3) -----']
        for name, val in t3['macros'].items():
            lines.append(f'\\newcommand{{\\{name}}}{{\\ensuremath{{{val}}}}}')
        lines += ['']

    lines += ['% -- deployment protocol (Table S3): five-seed probability averaging ----------']
    lines += [f'\\newcommand{{\\DeployMCC}}{{\\ensuremath{{{r3(v["tableS3"][DUET]["average_mcc"])}}}}}',
              f'\\newcommand{{\\BaseDeployMCC}}{{\\ensuremath{{{r3(v["tableS3"][BASE]["average_mcc"])}}}}}',
              f'\\newcommand{{\\DeployThr}}{{\\ensuremath{{{v["tableS3"][DUET]["threshold"]:.2f}}}}}',
              '']

    if os.path.exists(BOOT):
        b = json.load(open(BOOT, encoding='utf-8'))
        avg = b['average']
        lines += ['% -- chain-level paired bootstrap of the deployment predictions ----------------',
                  '%    (NOT reported in the manuscript; kept for the internal record)']
        lines += [f'\\newcommand{{\\BootAvgDelta}}{{\\ensuremath{{{s3(avg["delta"])}}}}}',
                  f'\\newcommand{{\\BootAvgCILo}}{{\\ensuremath{{{s3(avg["ci95"][0])}}}}}',
                  f'\\newcommand{{\\BootAvgCIHi}}{{\\ensuremath{{{s3(avg["ci95"][1])}}}}}',
                  f'\\newcommand{{\\BootAvgP}}{{\\ensuremath{{{avg["p_delta_gt_0"]:.3f}}}}}']
        if 'RB9' in b:
            rb = b['RB9']
            lines += [f'\\newcommand{{\\BootRBDelta}}{{\\ensuremath{{{s3(rb["delta"])}}}}}',
                      f'\\newcommand{{\\BootRBCILo}}{{\\ensuremath{{{s3(rb["ci95"][0])}}}}}',
                      f'\\newcommand{{\\BootRBCIHi}}{{\\ensuremath{{{s3(rb["ci95"][1])}}}}}',
                      f'\\newcommand{{\\BootRBP}}{{\\ensuremath{{{rb["p_delta_gt_0"]:.3f}}}}}']
        lines += ['']


    if v.get('table1'):
        lines += ['% -- Table 1 cells: dataset sizes and class imbalance ---------------------------']
        key2tag = {'train': 'Train', 'val': 'Val', 'TE18': 'TE', 'TL12': 'TL',
                   'JL10': 'JL', 'RB9': 'RB'}
        for row in v['table1']:
            tag = key2tag[row['key']]
            num = lambda n: f'{n:,}'.replace(',', '{,}')

            chains = row['chains'].replace('→', r'\,\textrightarrow\,')
            lines += [f'\\newcommand{{\\TbOne{tag}Chains}}{{{chains}}}',
                      f'\\newcommand{{\\TbOne{tag}Res}}{{\\ensuremath{{{num(row["nucleotides"])}}}}}',
                      f'\\newcommand{{\\TbOne{tag}Bind}}{{\\ensuremath{{{num(row["binding"])}}}}}',
                      f'\\newcommand{{\\TbOne{tag}Non}}{{\\ensuremath{{{num(row["non_binding"])}}}}}',
                      f'\\newcommand{{\\TbOne{tag}Pos}}{{\\ensuremath{{{row["positive_ratio"]:.3f}}}}}',
                      f'\\newcommand{{\\TbOne{tag}Imb}}{{\\ensuremath{{{row["imbalance"]:.2f}}}}}']
        lines += ['']




    if os.path.exists(TOWERS):
        tow = json.load(open(TOWERS, encoding='utf-8'))
        lines += ['% -- leakage control (scripts/14_tower_analysis.py, five development seeds) ---']
        for name, val in tow.get('macros', {}).items():
            if name.startswith('Leak'):
                lines.append(f'\\newcommand{{\\{name}}}{{\\ensuremath{{{val}}}}}')
        lines += ['']


    if os.path.exists(GRASSP):
        gc = json.load(open(GRASSP, encoding='utf-8'))
        lines += ['% -- Table 4: DUET vs GRASSP on the three shared benchmarks ---------------------']
        tag = {'TE18': 'TE', 'TL12': 'TL', 'JL10': 'JL'}
        short = {'DUET (sequence only)': 'GcDuet',
                 'GRASSP (sequence + 3D structure)': 'GcGrassp'}
        for label, pre in short.items():
            for sp, sg in tag.items():
                m = gc['per_benchmark'][sp][label]['MCC']
                lines.append(f'\\newcommand{{\{pre}{sg}}}{{\\ensuremath{{'
                             f'{m["mean"]:.3f} \\pm {m["sd"]:.3f}}}}}')
            m = gc['average'][label]['MCC']
            lines.append(f'\\newcommand{{\{pre}Avg}}{{\\ensuremath{{'
                         f'{m["mean"]:.3f} \\pm {m["sd"]:.3f}}}}}')

        def pfmt(x):
            return '{<}0.001' if x < 0.001 else f'{x:.3f}'

        for sp, sg in tag.items():
            d = gc['significance']['MCC'][sp]
            lines.append(f'\\newcommand{{\\GcDelta{sg}}}{{\\ensuremath{{{d["delta"]:+.3f}}}}}')
            lines.append(f'\\newcommand{{\\GcP{sg}}}{{\\ensuremath{{{pfmt(d["p"])}}}}}')
        d = gc['significance']['MCC']['average']
        lines.append(f'\\newcommand{{\\GcDeltaAvg}}{{\\ensuremath{{{d["delta"]:+.3f}}}}}')
        lines.append(f'\\newcommand{{\\GcPAvg}}{{\\ensuremath{{{pfmt(d["p"])}}}}}')
        lines.append(f'\\newcommand{{\\GcSeeds}}{{\\ensuremath{{{len(gc["seeds"])}}}}}')
        for full, ab in [('AUC', 'AUC'), ('AUPRC', 'AUPRC'), ('F1', 'FOne'),
                         ('Precision', 'Precision'), ('Sensitivity', 'Sensitivity'),
                         ('Specificity', 'Specificity'), ('GMean', 'GMean')]:
            if full not in gc['significance']:
                continue
            c = gc['significance'][full]['average']
            lines.append(f'\\newcommand{{\\GcDelta{ab}}}{{\\ensuremath{{{c["delta"]:+.3f}}}}}')
            lines.append(f'\\newcommand{{\\GcP{ab}}}{{\\ensuremath{{{pfmt(c["p"])}}}}}')
        lines += ['']




    rig = os.path.join(ROOT, 'results', 'rigour_analyses.json')
    extra = os.path.join(ROOT, 'results', 'reporting_extras.json')
    if os.path.exists(rig):
        r = json.load(open(rig, encoding='utf-8'))
        d = r['chain_level']['duet']
        b = r['chain_level']['baseline']
        h = r['hierarchical_delta']['MCC']
        lines += ['% -- chain-level inference (scripts/27_rigour_analyses.py) ----------------------']
        lines += [f'\\newcommand{{\\ChainDuet}}{{\\ensuremath{{{d["mean"]:.3f}}}}}',
                  f'\\newcommand{{\\ChainDuetLo}}{{\\ensuremath{{{d["ci95"][0]:.3f}}}}}',
                  f'\\newcommand{{\\ChainDuetHi}}{{\\ensuremath{{{d["ci95"][1]:.3f}}}}}',
                  f'\\newcommand{{\\ChainBase}}{{\\ensuremath{{{b["mean"]:.3f}}}}}',
                  f'\\newcommand{{\\ChainDelta}}{{\\ensuremath{{{h["delta"]:+.3f}}}}}',
                  f'\\newcommand{{\\ChainDeltaLo}}{{\\ensuremath{{{h["ci95"][0]:+.3f}}}}}',
                  f'\\newcommand{{\\ChainDeltaHi}}{{\\ensuremath{{{h["ci95"][1]:+.3f}}}}}',
                  f'\\newcommand{{\\ChainP}}{{\\ensuremath{{{h["p_boot"]:.3f}}}}}']
        mm, mb = r['macro_micro']['duet'], r['macro_micro']['baseline']
        lines += [f'\\newcommand{{\\MicroDuet}}{{\\ensuremath{{{mm["micro"]:.3f}}}}}',
                  f'\\newcommand{{\\MicroBase}}{{\\ensuremath{{{mb["micro"]:.3f}}}}}',
                  f'\\newcommand{{\\MacroDuet}}{{\\ensuremath{{{mm["macro"]:.3f}}}}}',
                  f'\\newcommand{{\\MacroBase}}{{\\ensuremath{{{mb["macro"]:.3f}}}}}']
        loo = {k: r['leave_one_out']['duet']['loo_benchmark'][k]
               - r['leave_one_out']['baseline']['loo_benchmark'][k]
               for k in r['leave_one_out']['duet']['loo_benchmark']}
        lines += [f'\\newcommand{{\\LooMin}}{{\\ensuremath{{{min(loo.values()):+.3f}}}}}',
                  f'\\newcommand{{\\LooMax}}{{\\ensuremath{{{max(loo.values()):+.3f}}}}}']
        for sp, sg in [('TE18', 'TE'), ('TL12', 'TL'), ('JL10', 'JL'), ('RB9', 'RB')]:
            lines.append(f'\\newcommand{{\\Loo{sg}}}{{\\ensuremath{{{loo[sp]:+.3f}}}}}')
        td, tb = r['threshold']['duet'], r['threshold']['baseline']
        lines += [f'\\newcommand{{\\FixDuet}}{{\\ensuremath{{{td["fixed_0.5"]:.3f}}}}}',
                  f'\\newcommand{{\\FixBase}}{{\\ensuremath{{{tb["fixed_0.5"]:.3f}}}}}',
                  f'\\newcommand{{\\OracleDuet}}{{\\ensuremath{{{td["oracle_threshold"]:.3f}}}}}',
                  f'\\newcommand{{\\OracleBase}}{{\\ensuremath{{{tb["oracle_threshold"]:.3f}}}}}']
        c = r['calibration']
        lines += [f'\\newcommand{{\\CalDuetBrier}}{{\\ensuremath{{{c["duet"]["brier"]:.3f}}}}}',
                  f'\\newcommand{{\\CalBaseBrier}}{{\\ensuremath{{{c["baseline"]["brier"]:.3f}}}}}',
                  f'\\newcommand{{\\CalDuetECE}}{{\\ensuremath{{{c["duet"]["ece_10bin"]:.3f}}}}}',
                  f'\\newcommand{{\\CalBaseECE}}{{\\ensuremath{{{c["baseline"]["ece_10bin"]:.3f}}}}}',
                  f'\\newcommand{{\\CalTowerECE}}{{\\ensuremath{{{c["dense"]["ece_10bin"]:.3f}}}}}']
        dv = r['diversity']
        for key, tag in [('graph|tcn', 'GraphTcn'), ('graph|dense', 'GraphDense'),
                         ('tcn|dense', 'TcnDense')]:
            lines.append(f'\\newcommand{{\\DivR{tag}}}{{\\ensuremath{{{dv[key]["prob_correlation"]:.2f}}}}}')
            lines.append(f'\\newcommand{{\\DivQ{tag}}}{{\\ensuremath{{{dv[key]["q_statistic"]:.2f}}}}}')
            lines.append(f'\\newcommand{{\\DivDis{tag}}}{{\\ensuremath{{{dv[key]["disagreement"]:.3f}}}}}')
        lines.append(f'\\newcommand{{\\OracleBound}}{{\\ensuremath{{{dv["oracle_any_member_correct"]:.3f}}}}}')
        fu = r['fusion']
        for key, tag in [('plain', 'Plain'), ('rank', 'Rank'), ('weighted_val', 'Weighted'),
                         ('logit', 'Logit'), ('majority', 'Majority')]:
            lines.append(f'\\newcommand{{\\Fuse{tag}}}{{\\ensuremath{{{fu[key]:.3f}}}}}')
        lines += ['']
    if os.path.exists(extra):
        e = json.load(open(extra, encoding='utf-8'))
        fam = e['multiplicity']['all']
        avgfam = e['multiplicity']['average_only']
        lines += ['% -- multiplicity (Benjamini-Hochberg) and runtime cost (scripts/29) -----------']
        lines += [f'\\newcommand{{\\BHSurvive}}{{\\ensuremath{{{sum(1 for v in fam.values() if v["survives"])}}}}}',
                  f'\\newcommand{{\\BHTotal}}{{\\ensuremath{{{len(fam)}}}}}',
                  f'\\newcommand{{\\BHSurviveAvg}}{{\\ensuremath{{{sum(1 for v in avgfam.values() if v["survives"])}}}}}',
                  f'\\newcommand{{\\BHTotalAvg}}{{\\ensuremath{{{len(avgfam)}}}}}',
                  f'\\newcommand{{\\GPUName}}{{{e["gpu"].split(",")[0]}}}']
        for k, tag in [('graph', 'Graph'), ('tcn', 'Tcn'), ('dense', 'Dense')]:
            ms = e['inference_cost_cpu'][k]['seconds_per_chain'] * 1000
            lines.append(f'\\newcommand{{\\Infer{tag}}}{{\\ensuremath{{{ms:.0f}}}}}')
        lines += ['']

    lines += ['% -- data and experimental scale --------------------------------------------']


    import re as _re
    n_runs = len([f for f in os.listdir(os.path.join(ROOT, 'results', 'controlled'))
                  if f.endswith('.json') and not _re.match(r'train_p(80|70|50)_', f)])


    lines += ['\\newcommand{\\NTrainChains}{\\ensuremath{314}}',
              '\\newcommand{\\NValChains}{\\ensuremath{40}}',
              '\\newcommand{\\NRBChains}{\\ensuremath{6}}',
              '\\newcommand{\\NTestChains}{\\ensuremath{49}}',
              '\\newcommand{\\NTestChainsDeleaked}{\\ensuremath{46}}',
              '\\newcommand{\\LeakIdentical}{\\ensuremath{16}}',
              '\\newcommand{\\LeakHomologous}{\\ensuremath{22}}',
              f'\\newcommand{{\\NControlledRuns}}{{\\ensuremath{{{n_runs}}}}}',
              '']

    ext = os.path.join(ROOT, 'results', 'external_split.json')
    if os.path.exists(ext):
        x = json.load(open(ext, encoding='utf-8'))
        lines += ['% -- release hold-out split (scripts/28_external_split.py) -----------------------']
        lines += [f'\\newcommand{{\\ExtChains}}{{\\ensuremath{{{x["homology"]["n_chains"]}}}}}',
                  f'\\newcommand{{\\ExtIdentical}}{{\\ensuremath{{{x["homology"]["vs_train_val"]["n_exact"]}}}}}',
                  f'\\newcommand{{\\ExtHom}}{{\\ensuremath{{{x["homology"]["vs_train_val"]["hits"]["p80"]}}}}}',
                  f'\\newcommand{{\\ExtMCC}}{{\\ensuremath{{{x["evaluation"]["duet"]["mean"]:.3f}}}}}']
        lines += ['']

    strict = os.path.join(ROOT, 'results', 'strict_summary.json')
    if os.path.exists(strict):
        st = json.load(open(strict, encoding='utf-8'))['thresholds']
        lines += ['% -- stricter de-leaking sweep (scripts/30_strict_summary.py) -------------------']
        for thr in ['p80', 'p70', 'p50']:
            if thr not in st:
                continue
            e = st[thr]
            sg = {'p80': 'Eighty', 'p70': 'Seventy', 'p50': 'Fifty'}[thr]
            lines += [f'\\newcommand{{\\Strict{sg}Train}}{{\\ensuremath{{{e["n_train"]}}}}}',
                      f'\\newcommand{{\\Strict{sg}Val}}{{\\ensuremath{{{e["n_val"]}}}}}',
                      f'\\newcommand{{\\Strict{sg}Duet}}{{\\ensuremath{{{e["duet_mean"]:.3f}}}}}',
                      f'\\newcommand{{\\Strict{sg}Base}}{{\\ensuremath{{{e["baseline_mean"]:.3f}}}}}',
                      f'\\newcommand{{\\Strict{sg}Delta}}{{\\ensuremath{{{e["delta"]:+.3f}}}}}',
                      f'\\newcommand{{\\Strict{sg}P}}{{\\ensuremath{{{"< 0.001" if e["p_seed_level"] < 0.001 else format(e["p_seed_level"], ".3f")}}}}}',
                      f'\\newcommand{{\\Strict{sg}Lo}}{{\\ensuremath{{{e["chain_ci"]["ci95"][0]:+.3f}}}}}',
                      f'\\newcommand{{\\Strict{sg}Hi}}{{\\ensuremath{{{e["chain_ci"]["ci95"][1]:+.3f}}}}}',
                      f'\\newcommand{{\\Strict{sg}Seeds}}{{\\ensuremath{{{len(e["seeds"])}}}}}']
            if 'grassp' in e:
                g = e['grassp']
                lines += [f'\\newcommand{{\\Strict{sg}Grassp}}{{\\ensuremath{{{g["mean"]:.3f}}}}}',
                          f'\\newcommand{{\\Strict{sg}GrasspDelta}}{{\\ensuremath{{{g["delta"]:+.3f}}}}}',
                          f'\\newcommand{{\\Strict{sg}GrasspP}}{{\\ensuremath{{{"< 0.001" if g["p"] < 0.001 else format(g["p"], ".3f")}}}}}',
                          f'\\newcommand{{\\Strict{sg}ThreeSet}}{{\\ensuremath{{{g["duet_3set"]:.3f}}}}}',
                          f'\\newcommand{{\\Strict{sg}GrasspGap}}{{\\ensuremath{{{-g["delta"]:+.3f}}}}}']
        lines += ['']

    abl = os.path.join(ROOT, 'docs', 'tables', 'ablation_Table3.md')
    if os.path.exists(abl):
        import re as _re
        txt = open(abl, encoding='utf-8').read()
        tag = {'w/o secondary-structure edges': 'Ss', 'GraphNorm → LayerNorm': 'Norm',
               'w/o DropEdge': 'Drop', 'TCL-focal → BCE loss': 'Bce',
               'w/o residual connections': 'Res'}
        lines += ['% -- graph-tower ablation, ten seeds, per-seed protocol (summarize_ablation.py) --']
        for label, tg in tag.items():
            row = next((l for l in txt.splitlines() if l.startswith(f'| {label} ')), None)
            if not row:
                continue
            cells = [c.strip() for c in row.split('|')]
            lines += [f'\\newcommand{{\\Abl{tg}Delta}}{{\\ensuremath{{{cells[5]}}}}}',
                      f'\\newcommand{{\\Abl{tg}P}}{{\\ensuremath{{{"< 0.001" if cells[6] == "0.000" else cells[6]}}}}}']
        lines += ['']

    ext = os.path.join(ROOT, 'results', 'external_eval.json')
    extb = os.path.join(ROOT, 'results', 'external_eval_baseline.json')
    r2 = os.path.join(ROOT, 'results', 'review2_analyses.json')
    if os.path.exists(ext):
        e = json.load(open(ext, encoding='utf-8'))['per_set']
        eb = json.load(open(extb, encoding='utf-8'))['per_set'] if os.path.exists(extb) else {}
        lines += ['% -- external validation on the SMARTPocket splits (scripts/33, 34) ----------']
        tag = {'time_dependent': 'Time', 'external_jiang': 'Jiang', 'hariboss_test': 'HariTest'}
        for name, tg in tag.items():
            if name not in e:
                continue
            m = e[name]
            lines += [f'\\newcommand{{\\Ext{tg}Chains}}{{\\ensuremath{{{m["n_chains"]}}}}}',
                      f'\\newcommand{{\\Ext{tg}Res}}{{\\ensuremath{{{m["n_residues"]}}}}}',
                      f'\\newcommand{{\\Ext{tg}MCC}}{{\\ensuremath{{{m["MCC"]["mean"]:.3f}}}}}',
                      f'\\newcommand{{\\Ext{tg}MCCSD}}{{\\ensuremath{{{m["MCC"]["sd"]:.3f}}}}}',
                      f'\\newcommand{{\\Ext{tg}AUC}}{{\\ensuremath{{{m["AUC"]["mean"]:.3f}}}}}']
            if name in eb:
                b = eb[name]
                lines += [f'\\newcommand{{\\Ext{tg}Base}}{{\\ensuremath{{{b["MCC"]["mean"]:.3f}}}}}',
                          f'\\newcommand{{\\Ext{tg}BaseSD}}{{\\ensuremath{{{b["MCC"]["sd"]:.3f}}}}}']
                p_ = b.get('paired_vs_duet')
                if p_:
                    lines += [f'\\newcommand{{\\Ext{tg}Delta}}{{\\ensuremath{{{p_["delta"]:+.3f}}}}}',
                              f'\\newcommand{{\\Ext{tg}P}}{{\\ensuremath{{{("< 0.001" if p_["p"] < 0.001 else format(p_["p"], ".3f"))}}}}}']
        lines += ['']
    if os.path.exists(r2):
        d = json.load(open(r2, encoding='utf-8'))
        lines += ['% -- second-review analyses (scripts/31_review2_extras.py) ---------------------']
        for sp, sg in [('TE18', 'TE'), ('TL12', 'TL'), ('JL10', 'JL'), ('RB9', 'RB')]:
            c = d['A_per_benchmark_ci'][sp]
            lines += [f'\\newcommand{{\\Ci{sg}Lo}}{{\\ensuremath{{{c["duet"]["ci95"][0]:.3f}}}}}',
                      f'\\newcommand{{\\Ci{sg}Hi}}{{\\ensuremath{{{c["duet"]["ci95"][1]:.3f}}}}}']
        sf, sp_ = d['B_tests']['chain_signflip'], d['B_tests']['seed_signflip']
        lines += [f'\\newcommand{{\\PermChain}}{{\\ensuremath{{{sf["observed"]:+.3f}}}}}',
                  f'\\newcommand{{\\PermChainP}}{{\\ensuremath{{{sf["p"]:.3f}}}}}',
                  f'\\newcommand{{\\PermSeedP}}{{\\ensuremath{{{sp_["p_signflip"]:.3f}}}}}']
        c2 = d['C_shared_chains']
        lines += [f'\\newcommand{{\\SharedCur}}{{\\ensuremath{{{c2["current_te18_18_rb9_6"]:.3f}}}}}',
                  f'\\newcommand{{\\SharedAlt}}{{\\ensuremath{{{c2["alternative_te18_15_rb9_9"]:.3f}}}}}']
        th = d['E_thresholds']
        lines += [f'\\newcommand{{\\ThrMean}}{{\\ensuremath{{{th["mean"]:.2f}}}}}',
                  f'\\newcommand{{\\ThrSD}}{{\\ensuremath{{{th["sd"]:.3f}}}}}',
                  f'\\newcommand{{\\ThrJLo}}{{\\ensuremath{{{th["per_benchmark_optimum"]["JL10"]:.2f}}}}}',
                  f'\\newcommand{{\\ThrRBo}}{{\\ensuremath{{{th["per_benchmark_optimum"]["RB9"]:.2f}}}}}']
        st = d['D_strata']
        for k, tg in [('short (<= median)', 'Short'), ('long (> median)', 'Long')]:
            if k in st:
                lines += [f'\\newcommand{{\\Strat{tg}Duet}}{{\\ensuremath{{{st[k]["duet"]:.3f}}}}}',
                          f'\\newcommand{{\\Strat{tg}Base}}{{\\ensuremath{{{st[k]["baseline"]:.3f}}}}}']
        lines += ['']

    off = os.path.join(ROOT, 'results', 'cobra_official_release.json')
    if os.path.exists(off):
        o = json.load(open(off, encoding='utf-8'))
        lines += ['% -- the officially released CoBRA checkpoint on the same test chains ----------']
        lines += [f'\\newcommand{{\\OfficialAvg}}{{\\ensuremath{{{o["avg_MCC"]:.3f}}}}}']
        for sp, sg in [('TE18', 'TE'), ('TL12', 'TL'), ('JL10', 'JL'), ('RB9', 'RB')]:
            lines.append(f'\\newcommand{{\\Official{sg}}}{{\\ensuremath{{{o["results"][sp]["MCC"]:.3f}}}}}')
        lines += ['']

    lines = unique_newcommands(lines)
    lines = [
        '\\providecommand' + line[len('\\newcommand'):]
        if line.startswith('\\newcommand') else line
        for line in lines
    ]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    print(f'[written] {OUT}  '
          f'({len([l for l in lines if l.startswith(chr(92)+"providecommand")])} macros)')
    print('note: NTestChains / leak counts / chain counts are constants of the dataset.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
