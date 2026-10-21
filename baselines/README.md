# Baselines

## `scripts/06_train_baselines.py` — CoBRA-M3 equivalent (main comparison)

The main-table baseline is **retrained by us on the same de-leaked split, with the same
protocol and the same five seeds**, because the values reported in the original CoBRA paper
cannot be reproduced from their public release (see below). The equivalent model is the
CoBRA-M3 configuration: RiNALMo-giga per-residue embeddings + MLP + BCE loss.

```bash
python scripts/06_train_baselines.py     # -> results/cobra_m3_equiv.json
```

The JSON stores, for every seed, the validation-selected threshold and the per-chain
probabilities of all four benchmarks, which is what makes the paired *t*-tests in
Table 2 same-seed exact.

## `cobra_official/` — reproduction of the published CoBRA release

`run_official_pipeline.py` runs the authors' code end to end (their preprocessing, their
checkpoint) on the four benchmarks; `run_cobra_eval.py` is our own evaluator over the same
predictions, used to cross-check the numbers. Findings, documented in
[`reproduction_audit_report.md`](cobra_official/reproduction_audit_report.md):

* the released checkpoint **reproduces exactly** with our evaluator (average MCC 0.4451 on the
  6-chain RB9 evaluation set we use) — our earlier hand-written reproduction agrees to the
  last digit;
* the per-benchmark MCC values printed in the CoBRA paper's Table 3 **cannot be reproduced**
  from the released code and weights; the discrepancy is not explained by thresholds alone
  (`threshold_analysis.py`, `metrics_variants.py`, `subset_analysis.py` probe the usual
  suspects: threshold handling, pooled vs per-chain averaging, metric variants, chain subsets).

Consequently the paper compares against a **retrained** baseline and lists the original
reported values separately, with an explicit note that they are not reproducible.

## `grassp/` — GRASSP on RB9

GRASSP does not evaluate on RB9, so its published numbers cannot be compared on that set.
`grassp_rb9_inference.py` runs the GRASSP model on RB9 (which has experimental structures, so
all of GRASSP's inputs are available) using the authors' code and checkpoints. It expects the
GRASSP repository next to this one:

```bash
GRASSP_ROOT=/path/to/GRASSP-main python baselines/grassp/grassp_rb9_inference.py
```

## Original reported values

Original-paper values and retrained values under the identical de-leaked protocol are stored
in `results/per_set_metrics.json` and `results/table_values.json`.
