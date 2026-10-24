# Large-data homology-aware protocol

## 1. Training pool

The pool is the union of the raw HARIBOSS+TR60 training and validation
records. It is filtered only against the canonical 44-chain test set:
records at or above 80% global sequence identity to any TE18, TL12, JL10 or
RB9 chain are removed. Exact duplicates inside the remaining pool are kept.

Generated manifest:

```text
data/clean/manifests/large_p80_cv5.json
```

Current counts:

- pool: 290 chains;
- retained exact duplicates: 110 records;
- 80% identity connected components: 115;
- largest component: 27 chains;
- five validation folds: 58 chains each.

Homology components cannot be split, so equal chain counts do not imply equal
positive-residue counts. The five folds contain 558, 1253, 1063, 744 and 853
positive residues. This imbalance is reported rather than hidden; every model
uses the identical fold assignment, so method contrasts are paired on the same
evaluation partitions.

Identity is computed by global alignment with match `+1`, mismatch `-1`, gap
opening `-4`, gap extension `-1`, and length ratio gate `0.8`. Selection uses
`matches / alignment columns`.

## 2. Clustering and folds

The 80% identity graph is built over all pool records and clustered into
connected components. Connected components, rather than star-shaped greedy
clusters, are used so that two non-representative records in different folds
cannot remain homologous above the threshold.

Whole components are assigned by longest-processing-time scheduling: the
largest component goes to the currently smallest fold, with positive-residue
and residue counts used as deterministic tie breakers. Audit file:

```text
results/large_protocol/large_p80_cv5_leakage.json
```

The audit checks every validation/training pair in every fold and reports zero
violations at 80% identity.

## 3. Group-balanced sampling

Each record in a homology component of size `s` receives weight `1 / s`.
Sampling is with replacement for exactly as many draws as there are training
records in the current fold or final pool. Therefore each component has total
sampling weight one, irrespective of how many redundant records it contains.

## 4. CV and final fitting

Five-fold CV uses seed 42. For each model:

1. train on four folds and early-stop on the held-out fold at threshold 0.5;
2. retain the fold-best epoch;
3. pool out-of-fold probabilities across the five folds;
4. choose one threshold by pooled OOF MCC;
5. use the median fold-best epoch as the final epoch.

The final model is then retrained from scratch on all 290 pool records for the
selected epoch, using the selected threshold. Final comparisons use the ten
paired seeds `42, 0, 1, 2, 3, 7, 11, 13, 17, 23`.

The graph and sequence-tower CV searches run up to 40 epochs. GRASSP is given a
100-epoch search without early stopping because its released training recipe
selects late epochs (88-100 in the prior strict-split runs); applying the
short 40-epoch budget would bias the crossed encoder comparison.

Selected hyperparameters:

```text
results/large_protocol/cv/<method>/selected.json
```

Final runs:

```text
results/large_protocol/final/<method>/seed<seed>.json
```

## 5. Models

The matrix contains:

- CoBRA-M3 with RiNALMo features;
- DUET R-GCN, TCN and DenseNet towers;
- the equal-weight DUET ensemble;
- R-GCN on sequence + 3D structural edges;
- R-GCN on sequence + secondary structure + 3D structural edges;
- GRASSP with ERNIE-RNA;
- crossed control: GRASSP architecture with RNA-FM;
- crossed control: R-GCN with sequence + 3D edges and ERNIE-RNA.

GRASSP and the ERNIE-RNA R-GCN control are evaluated on TE18, TL12 and JL10
because the released GRASSP structural assets do not contain an aligned RB9
set.

## 6. SMARTPocket external validation

The SMARTPocket release is rebuilt under the same global-identity definition
and checked against the large training pool:

- `external_jiang`: 2 chains, 97 residues;
- `time_dependent`: 11 chains, 885 residues;
- `hariboss_test`: 6 chains, 1010 residues, source-matched control only.

Jiang chain `7REX_A` is an exact sequence duplicate of canonical test chain
`TL12/7rex_A` and is removed. One time-dependent chain (`8FB3_A`) is an exact
sequence duplicate of Jiang chain `7REX_A` and is also removed. No retained
chain reaches 80% identity to the large training pool or any canonical test.

External 3D edges are directed top-8 residue-neighbour graphs from RCSB CIF
coordinates, using `C4'`, then `P`, then `C1'` as the representative atom.
Four chains have one unmodelled standard residue in their CIF; their
coordinates are filled from the nearest aligned residue and recorded in:

```text
results/large_protocol/smartpocket_graphs_audit.json
```

The external secondary-structure tensor uses four channels derived from the
released dot-bracket: paired, opening, closing and unpaired. This external
encoding is documented separately from the released GRASSP training tensors.

## 7. Statistical analysis

All main comparisons use the same ten paired seeds. The independent
resampling unit is the benchmark chain; chain bootstrap keeps the ten seeds
paired and resamples chains within each benchmark.

Holm correction is applied within four prespecified families:

- primary sequence-model comparisons over TE18/TL12/JL10/RB9;
- primary comparisons at fixed threshold 0.5;
- structural comparisons over TE18/TL12/JL10;
- crossed encoder/architecture controls over TE18/TL12/JL10.

SMARTPocket comparisons use the independent external sets only. The
source-matched HARIBOSS hold-out is reported as a control and is excluded from
the independent-external average.

Machine-readable result:

```text
results/large_protocol/analysis.json
```

Human-readable result:

```text
results/large_protocol/LARGE_PROTOCOL_RESULTS.md
```
