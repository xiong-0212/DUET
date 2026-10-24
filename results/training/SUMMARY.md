# Training summary

The current split is the large test-clean pool with 80% identity clustering and five homology-aware folds. All final comparisons use the same ten paired seeds.

## Large-data protocol

- Pool: **290** chains.
- Homology components: **115**.
- Retained exact duplicates: **110** records.
- Fold sizes: **[58, 58, 58, 58, 58]**.
- Cross-fold homology violations at 80%: **0**.
- Sampling weight per record: **1 / homology-cluster size**.
- CV selected epoch as the median fold-best epoch and threshold from pooled out-of-fold MCC.

## Internal test MCC

| Method | Mean | SD | TE18 | TL12 | JL10 | RB9 |
|:--|--:|--:|--:|--:|--:|--:|
| cobra | 0.192 | 0.031 | 0.167 | 0.190 | 0.180 | 0.231 |
| duet_dense | 0.213 | 0.012 | 0.189 | 0.177 | 0.206 | 0.280 |
| duet_ensemble | 0.240 | 0.013 | 0.225 | 0.245 | 0.213 | 0.276 |
| duet_graph | 0.193 | 0.017 | 0.123 | 0.213 | 0.176 | 0.260 |
| duet_tcn | 0.208 | 0.024 | 0.188 | 0.201 | 0.201 | 0.242 |
| graph_seq3d | 0.212 | 0.022 | 0.153 | 0.278 | 0.206 | - |
| graph_seqss3d | 0.180 | 0.025 | 0.152 | 0.252 | 0.137 | - |
| grassp_ernie | 0.254 | 0.018 | 0.181 | 0.340 | 0.243 | - |
| grassp_rnafm | 0.261 | 0.014 | 0.258 | 0.323 | 0.201 | - |
| rgcn_ernie | 0.161 | 0.031 | 0.081 | 0.172 | 0.230 | - |

## Primary comparisons

| Comparison | Delta MCC | 95% CI | raw p | Holm p | Seeds better |
|:--|--:|:--|--:|--:|--:|
| duet_dense_vs_cobra | +0.021 | -0.005 to +0.047 | 0.1036 | 0.3107 | 8/10 |
| duet_ensemble_vs_cobra | +0.047 | +0.023 to +0.072 | 0.001833 | 0.007333 | 10/10 |
| duet_ensemble_vs_duet_graph | +0.047 | +0.031 to +0.063 | 9.46e-05 | 0.000473 | 10/10 |
| duet_graph_vs_cobra | +0.001 | -0.023 to +0.025 | 0.9491 | 0.9491 | 5/10 |
| duet_tcn_vs_cobra | +0.016 | -0.014 to +0.046 | 0.2695 | 0.5391 | 6/10 |

At fixed threshold 0.5:

| Comparison | Delta MCC | 95% CI | raw p | Holm p | Seeds better |
|:--|--:|:--|--:|--:|--:|
| duet_dense_vs_cobra | +0.032 | +0.003 to +0.061 | 0.03232 | 0.09696 | 9/10 |
| duet_ensemble_vs_cobra | +0.052 | +0.024 to +0.081 | 0.002557 | 0.01023 | 10/10 |
| duet_ensemble_vs_duet_graph | +0.043 | +0.029 to +0.057 | 5.837e-05 | 0.0002918 | 10/10 |
| duet_graph_vs_cobra | +0.009 | -0.022 to +0.040 | 0.5214 | 0.5214 | 5/10 |
| duet_tcn_vs_cobra | +0.024 | -0.007 to +0.055 | 0.114 | 0.228 | 6/10 |

## Structure

| Comparison | Delta MCC | 95% CI | raw p | Holm p | Seeds better |
|:--|--:|:--|--:|--:|--:|
| graph_seq3d_vs_duet_graph | +0.042 | +0.028 to +0.056 | 8.954e-05 | 0.0004477 | 10/10 |
| graph_seqss3d_vs_graph_seq3d | -0.032 | -0.051 to -0.013 | 0.003928 | 0.007856 | 1/10 |
| grassp_ernie_vs_cobra | +0.075 | +0.046 to +0.105 | 0.0002636 | 0.0007908 | 10/10 |
| grassp_ernie_vs_duet_ensemble | +0.027 | +0.007 to +0.047 | 0.01274 | 0.01274 | 8/10 |
| grassp_ernie_vs_graph_seqss3d | +0.074 | +0.047 to +0.101 | 0.0001603 | 0.0006411 | 10/10 |
| grassp_rnafm_vs_duet_ensemble | +0.033 | +0.025 to +0.042 | 1.117e-05 | 6.699e-05 | 10/10 |

## Crossed controls

| Comparison | Delta MCC | 95% CI | raw p | Holm p | Seeds better |
|:--|--:|:--|--:|--:|--:|
| grassp_ernie_vs_grassp_rnafm | -0.006 | -0.021 to +0.008 | 0.3562 | 0.3562 | 3/10 |
| grassp_ernie_vs_rgcn_ernie | +0.094 | +0.067 to +0.120 | 2.597e-05 | 0.0001039 | 10/10 |
| grassp_rnafm_vs_graph_seq3d | +0.048 | +0.029 to +0.068 | 0.0003181 | 0.0009543 | 9/10 |
| rgcn_ernie_vs_graph_seq3d | -0.051 | -0.081 to -0.021 | 0.003732 | 0.007463 | 1/10 |

## Chain bootstrap

| Comparison | Delta | 95% CI | p |
|:--|--:|:--|--:|
| crossed_controls_3split/grassp_ernie_vs_grassp_rnafm | -0.006 | -0.057 to +0.041 | 0.8026 |
| crossed_controls_3split/grassp_ernie_vs_rgcn_ernie | +0.094 | +0.050 to +0.142 | 0.0009995 |
| crossed_controls_3split/grassp_rnafm_vs_graph_seq3d | +0.048 | +0.002 to +0.099 | 0.04298 |
| crossed_controls_3split/rgcn_ernie_vs_graph_seq3d | -0.051 | -0.109 to +0.012 | 0.1119 |
| primary_4split/duet_dense_vs_cobra | +0.021 | -0.008 to +0.049 | 0.1709 |
| primary_4split/duet_ensemble_vs_cobra | +0.047 | +0.019 to +0.076 | 0.0009995 |
| primary_4split/duet_ensemble_vs_duet_graph | +0.047 | -0.009 to +0.109 | 0.1149 |
| primary_4split/duet_graph_vs_cobra | +0.001 | -0.070 to +0.062 | 0.9835 |
| primary_4split/duet_tcn_vs_cobra | +0.016 | -0.011 to +0.042 | 0.2889 |
| primary_4split_fixed_0.5/duet_dense_vs_cobra | +0.032 | +0.007 to +0.057 | 0.008996 |
| primary_4split_fixed_0.5/duet_ensemble_vs_cobra | +0.052 | +0.023 to +0.083 | 0.0009995 |
| primary_4split_fixed_0.5/duet_ensemble_vs_duet_graph | +0.043 | -0.012 to +0.108 | 0.1369 |
| primary_4split_fixed_0.5/duet_graph_vs_cobra | +0.009 | -0.055 to +0.065 | 0.8036 |
| primary_4split_fixed_0.5/duet_tcn_vs_cobra | +0.024 | -0.002 to +0.048 | 0.06797 |
| structure_3split/graph_seq3d_vs_duet_graph | +0.042 | +0.005 to +0.074 | 0.02499 |
| structure_3split/graph_seqss3d_vs_graph_seq3d | -0.032 | -0.064 to +0.006 | 0.09295 |
| structure_3split/grassp_ernie_vs_cobra | +0.075 | +0.015 to +0.141 | 0.01599 |
| structure_3split/grassp_ernie_vs_duet_ensemble | +0.027 | -0.033 to +0.094 | 0.4298 |
| structure_3split/grassp_ernie_vs_graph_seqss3d | +0.074 | +0.008 to +0.137 | 0.03698 |
| structure_3split/grassp_rnafm_vs_duet_ensemble | +0.033 | -0.031 to +0.103 | 0.3078 |
| structure_3split_fixed_0.5/graph_seq3d_vs_duet_graph | +0.040 | +0.005 to +0.072 | 0.02799 |
| structure_3split_fixed_0.5/graph_seqss3d_vs_graph_seq3d | -0.033 | -0.064 to +0.004 | 0.07396 |
| structure_3split_fixed_0.5/grassp_ernie_vs_cobra | +0.092 | +0.030 to +0.155 | 0.003998 |
| structure_3split_fixed_0.5/grassp_ernie_vs_duet_ensemble | +0.038 | -0.022 to +0.103 | 0.2399 |
| structure_3split_fixed_0.5/grassp_ernie_vs_graph_seqss3d | +0.077 | +0.012 to +0.138 | 0.03298 |
| structure_3split_fixed_0.5/grassp_rnafm_vs_duet_ensemble | +0.044 | -0.020 to +0.113 | 0.1779 |

## SMARTPocket external validation

Independent sets are `external_jiang` and `time_dependent`; `hariboss_test` is a source-matched control and is excluded from the independent average.

| Method | Jiang MCC | Time-dependent MCC | Independent mean |
|:--|--:|--:|--:|
| cobra | 0.183 | 0.091 | 0.137 |
| duet_dense | 0.283 | 0.116 | 0.200 |
| duet_ensemble | 0.324 | 0.168 | 0.246 |
| duet_graph | 0.324 | 0.102 | 0.213 |
| duet_tcn | 0.304 | 0.127 | 0.216 |
| graph_seq3d | 0.302 | 0.118 | 0.210 |
| graph_seqss3d | 0.246 | 0.149 | 0.197 |
| grassp_ernie | 0.187 | 0.066 | 0.126 |
| grassp_rnafm | 0.328 | 0.159 | 0.243 |
| rgcn_ernie | -0.120 | -0.125 | -0.123 |

| Comparison | Delta MCC | 95% CI | raw p | Holm p | Seeds better |
|:--|--:|:--|--:|--:|--:|
| duet_ensemble_vs_cobra | +0.109 | +0.043 to +0.175 | 0.00466 | 0.01398 | 8/10 |
| graph_seqss3d_vs_graph_seq3d | -0.012 | -0.050 to +0.025 | 0.4761 | 0.9523 | 4/10 |
| grassp_ernie_vs_duet_ensemble | -0.120 | -0.172 to -0.067 | 0.0005997 | 0.002999 | 0/10 |
| grassp_rnafm_vs_cobra | +0.107 | +0.055 to +0.158 | 0.00115 | 0.004599 | 9/10 |
| grassp_rnafm_vs_duet_ensemble | -0.003 | -0.038 to +0.032 | 0.8644 | 0.9523 | 5/10 |

## Reproduction

Run `run_large_protocol.ps1`. Machine-readable outputs are under `results/large_protocol/`; the cluster and leakage audits are `large_p80_cv5_clusters.csv` and `large_p80_cv5_leakage.json`.

The previous strict-split results are preserved in `SUMMARY_LEGACY.md`.

## Follow-up experiments

All seven metrics, Holm-adjusted paired tests and chain bootstrap are reported under `results/followup/`.

- Large-data shuffled-3D control completed.
- Fixed-composition 80/160/290 training-size curve completed.
- Three additional cluster-stratified five-fold assignments completed with five final seeds per partition (reduced from ten for compute).
- Full 24-point GRASSP grid completed for ERNIE-RNA and RNA-FM.
- Duplicate-weighting ablation completed for DUET graph and GRASSP RNA-FM.

See `results/followup/FOLLOWUP_RESULTS.md` for key values and each experiment directory for full tables.
