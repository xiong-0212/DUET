# DUET

A sequence-only RNA language-model ensemble for RNA-small-molecule binding-site prediction.

This directory contains the source code, model weights and raw benchmark data
needed to inspect and retrain DUET. It intentionally excludes the manuscript,
processed tensors, logs and the full controlled-experiment checkpoint collection.

## Included

- `src/`: graph tower, TCN tower, DenseNet tower, losses and training loop
- `scripts/`: data preparation, training, evaluation and reporting pipeline
- `configs/duet.yaml`: canonical training configuration
- `checkpoints/controlled/`: 30 main-model weights, 10 seeds for each tower
- `data/raw/`: benchmark sequences, labels and RNA secondary-structure inputs
- `data/external/`: external-validation metadata needed to rebuild processed files
- `results/`: compact JSON results used by the manuscript tables

The manuscript source is deliberately not included.

## Repository structure

```text
duet/
├── src/                  core models, graph builder, losses and training loop
├── scripts/              data preparation, training, evaluation and reporting
├── configs/              canonical training and experiment settings
├── experiments/          controlled experiments behind the paper
├── baselines/            CoBRA and GRASSP comparison code
├── checkpoints/controlled/   30 released main-model checkpoints
├── data/raw/             benchmark sequences and labels
├── data/external/        external-validation metadata
├── results/              compact metrics and per-chain predictions
└── metadata/             checksums for released files
```

## Environment

```powershell
conda create -n duet python=3.9 -y
conda activate duet
pip install -r requirements.txt
```

## Weights

`checkpoints/controlled/` contains the main three-tower weights:

- development seeds `42, 0, 1, 2, 3`
- confirmation seeds `7, 11, 13, 17, 23`
- graph tags `clean_seq_ss` / `ns_graph`
- TCN tags `slim_tcn` / `ns_tcn`
- DenseNet tags `slim_dense` / `ns_dense`

These files are marked for Git LFS in `.gitattributes`. If LFS is unavailable,
publish the same directory as a GitHub Release asset instead.

Before adding the repository to Git, enable LFS and verify that exactly the 30
checkpoints are matched:

```powershell
git lfs install
git add .gitattributes checkpoints/controlled
git lfs ls-files
```

The external pretrained language models RNA-FM and RiNALMo are not redistributed
here. Set `RNAFM_REPO`, `RNAFM_CKPT`, `RINALMO_REPO` and `RINALMO_CKPT` as
documented by their upstream projects.

## Rebuild processed data

The repository includes raw sequences and labels. Generate frozen embeddings
and graph tensors with:

```powershell
python scripts/01_extract_rnafm_embeddings.py
python scripts/02_extract_rinalmo_embeddings.py
python scripts/03_build_graphs.py
python scripts/04_build_graphs_dual.py
```

The processed `.pt` graphs are intentionally excluded from ordinary Git history
because they are large. The two files needed to reproduce the reported
predictions without rerunning the language models are published as GitHub
Release assets:

- `graphs_seq_ss_clean.pt` (about 59 MiB)
- `graphs_dual_clean.pt` (about 173 MiB)

Download both files into `data/processed/`. Their expected sizes and SHA-256
checksums are recorded in `metadata/release_assets.sha256` and
`data/processed/README.md`. They can also be rebuilt from the included raw data.

## Strict de-leaking

The five identity-threshold removal lists are included. Regenerate them and
the corresponding filtered graph inputs with:

```powershell
python scripts/25_strict_deleak.py
python scripts/26_build_strict_graphs.py
```

## Train DUET

```powershell
python scripts/05_train_duet.py --batch development
python scripts/05_train_duet.py --batch confirmation
```

This launches the graph, TCN and DenseNet towers. Existing checkpoints are
skipped, so interrupted runs can resume.

## Evaluate

After processed graphs exist, reproduce metrics and tables with:

```powershell
python scripts/22_three_tower_tables.py
python scripts/17_per_set_metrics.py
python scripts/29_reporting_extras.py
```

See `metadata/repository.sha256` for SHA-256 checksums covering the included
benchmark data and model checkpoints.
