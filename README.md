# DUET: sequence-only prediction of RNA-small molecule binding sites

This repository contains the code and processed resources for the DUET manuscript by Kaifeng
Xiong and Jianhua Jia.

Author ORCID: Kaifeng Xiong, <https://orcid.org/0009-0009-7176-9654>

## Large-data protocol

The current experiment matrix is the 290-chain test-clean pool with 80%
identity connected-component clustering, five homology-aware folds and
sequence-group-balanced sampling. It includes CoBRA, the three DUET towers,
3D graph ablations, GRASSP, crossed encoder/architecture controls and
SMARTPocket external validation.

Run the complete pipeline with:

```powershell
.\run_large_protocol.ps1
```

Protocol details are in `PROTOCOL_LARGE.md`. Generated outputs are under
`results/large_protocol/`, and the current training summary is
`results/training/SUMMARY.md`.

## Follow-up robustness experiments

The reviewer-facing follow-up matrix is implemented separately:

```powershell
.\run_followup.ps1
```

It runs all-metric significance analysis, the large-data shuffled-3D control,
the nested 80/160/290 training-size curve, three repeated cluster-stratified
CV assignments, the complete GRASSP hyperparameter grid, and the exact-duplicate
weighting ablation. Outputs are under `results/followup/`; acceptance is
recorded in `results/followup/acceptance.json`.

This directory is an isolated rebuild of the data-audit and sequence
de-duplication stage for the DUET manuscript. It does not modify any existing
directory under `E:\paper\r_little`.

The first milestone is deliberately limited to data curation. Model training
will start only after the generated split manifests pass structural and
sequence-level validation.

## What this pipeline fixes

- Exact duplicate sequences are removed across and within splits.
- Exact train/validation overlap is removed from training.
- Same-sequence families with conflicting residue labels are excluded from
  training instead of being assigned an arbitrary target.
- Homology is measured as global sequence identity with the conventional
  denominator, `matches / alignment columns`.
- Near-homologous records are removed from training and validation relative to
  the test sets, then clustered within training and validation.
- Every removed record is written to an auditable ledger with a reason.

## Layout

```text
configs/protocol.json                 declared policy
data/raw/cobra/                       untouched CoBRA input data
data/clean/manifests/                 generated split manifests
src/duet_clean/                       parsing, identity, de-duplication, validation
scripts/01_build_dedup.py             build manifests and audit reports
scripts/02_validate.py                verify all invariants
tests/                                unit tests for identity and de-duplication
results/                              inventories, ledgers and validation reports
```

## Run

```powershell
$env:PYTHONIOENCODING='utf-8'
& E:\conda_envs\xxx\python.exe .\scripts\run_all.py
```

The convenience wrapper runs the build, validation and unit tests:

```powershell
.\run_all.ps1
```

Pass `-Python <path>` if the interpreter is installed elsewhere.

To run the unit tests:

```powershell
$env:PYTHONIOENCODING='utf-8'
& E:\conda_envs\xxx\python.exe -m unittest discover -s .\tests -v
```

The current environment is `E:\conda_envs\xxx\python.exe` from the Miniconda
installation at `E:\miniconda3`. Any Python 3.9+ environment with PyTorch,
PyTorch Geometric, scikit-learn, SciPy and Biopython installed is sufficient.

To reproduce the complete released-checkpoint evaluation, CoBRA training and
DUET training:

```powershell
.\run_experiments.ps1
```

## Primary outputs

- `results/raw_inventory.json`: original counts and input checks.
- `results/exact_duplicate_groups.json`: all exact-sequence groups.
- `results/dedup_summary.json`: retained counts, removal counts and policies.
- `results/validation_report.json`: machine-checked invariants.
- `data/clean/manifests/exact_sequence_unique.json`: exact-clean split.
- `data/clean/manifests/homology_p{90,80,70,50}.json`: stricter splits.

The manifests contain complete sequence and label records, content hashes and
the removal ledger. They are the only inputs that should be used for later
training.

## Training results

The primary strict comparison is `homology_p80`; `exact_sequence_unique` is the
exact-only reference. Training outputs and the generated comparison reports are
under `results/training/`.
