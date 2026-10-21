# Results

Every file here is produced by the scripts in `scripts/` and `experiments/`; nothing is
hand-edited. Paths are relative to the repository root.

| File | Produced by | Content |
|:--|:--|:--|
| `table_values.json` | `scripts/07_make_tables.py` | machine-readable version of every number in the paper tables (per-seed and deployment protocols) |
| `cobra_m3_equiv.json` | `scripts/06_train_baselines.py` | per-seed, per-chain probabilities of the retrained CoBRA-M3-equivalent baseline (used for all paired tests) |
| `controlled/train_<tag>_seed<S>.json` | `scripts/05_train_duet.py`, `experiments/**` | per-run training logs: validation curve, selected threshold, test metrics per benchmark |
| `homology_audit.json`, `homology_identity.json` | `scripts/09_homology_audit.py` | train↔test sequence identity census |
| `stats_rigor.json` | `scripts/08_statistics.py` | chain-level bootstrap CIs, paired bootstrap, permutation tests |
| `ensemble_probe.json`, `ensemble_member_probs.json` | `experiments/ensemble/` | member probabilities and the ensemble feasibility probe |
| `gate_analysis.json`, `gate_counterfactual.json` | `experiments/gating/` | gate statistics and gate counterfactuals |
| `calibration.json` | `experiments/ensemble/calibration.py` | temperature scaling, ECE, selective prediction |
| `cobra_equiv_sensitivity.json` | `experiments/leakage/` | leakage sensitivity of the reproduced baseline |
| `full_comparison.csv` | `experiments/ensemble/full_comparison.py` | all eight metrics per benchmark and model |

`controlled/` is excluded from version control (311 files); regenerate it by re-running the
training scripts, or use the released copy. Paths recorded inside these JSONs are
repository-relative (`graphs_file`), so they stay valid on any machine.

## Reproducing the headline numbers

```bash
python scripts/07_make_tables.py
```

prints Tables 1, 2, S1–S3 and writes `table_values.json`. The per-seed vectors that back every
paired *t*-test are stored under `table2/<method>/per_seed_average` in that file, so any
reported *p*-value can be recomputed in one line.
