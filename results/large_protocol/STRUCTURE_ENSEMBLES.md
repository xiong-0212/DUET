# Large-data structure ensembles

| Method | Mean MCC | SD |
|:--|--:|--:|
| cobra | 0.179 | 0.045 |
| duet_graph | 0.170 | 0.020 |
| duet_ensemble | 0.227 | 0.016 |
| graph_seq3d | 0.212 | 0.022 |
| graph_seqss3d | 0.180 | 0.025 |
| graph_seq3d_ensemble | 0.239 | 0.020 |
| graph_seqss3d_ensemble | 0.236 | 0.022 |
| grassp_rnafm | 0.261 | 0.014 |

| Comparison | Delta | 95% CI | p | Seeds better |
|:--|--:|:--|--:|--:|
| graph_seq3d_ensemble_vs_cobra | +0.060 | +0.020 to +0.100 | 0.008219 | 10/10 |
| graph_seq3d_ensemble_vs_duet_ensemble | +0.012 | +0.005 to +0.018 | 0.00247 | 10/10 |
| graph_seq3d_ensemble_vs_graph_seq3d | +0.027 | +0.007 to +0.047 | 0.0153 | 9/10 |
| graph_seq3d_ensemble_vs_grassp_rnafm | -0.021 | -0.034 to -0.009 | 0.002981 | 1/10 |
| graph_seqss3d_ensemble_vs_cobra | +0.057 | +0.013 to +0.101 | 0.01677 | 9/10 |
| graph_seqss3d_ensemble_vs_duet_ensemble | +0.009 | +0.001 to +0.016 | 0.03273 | 7/10 |
| graph_seq3d_vs_cobra | +0.033 | -0.011 to +0.077 | 0.1231 | 6/10 |
| graph_seqss3d_vs_cobra | +0.001 | -0.046 to +0.048 | 0.9619 | 4/10 |

## External validation

| Method | Jiang | Time-dependent | Independent mean |
|:--|--:|--:|--:|
| cobra | 0.183 | 0.091 | 0.137 |
| duet_graph | 0.324 | 0.102 | 0.213 |
| duet_ensemble | 0.324 | 0.168 | 0.246 |
| graph_seq3d | 0.302 | 0.118 | 0.210 |
| graph_seqss3d | 0.246 | 0.149 | 0.197 |
| graph_seq3d_ensemble | 0.306 | 0.174 | 0.240 |
| graph_seqss3d_ensemble | 0.322 | 0.179 | 0.251 |
| grassp_rnafm | 0.328 | 0.159 | 0.243 |

| External comparison | Delta | 95% CI | p | Seeds better |
|:--|--:|:--|--:|--:|
| graph_seq3d_ensemble_vs_cobra | +0.103 | +0.040 to +0.166 | 0.004829 | 8/10 |
| graph_seq3d_ensemble_vs_duet_ensemble | -0.006 | -0.023 to +0.011 | 0.4399 | 4/10 |
| graph_seq3d_ensemble_vs_grassp_rnafm | -0.003 | -0.038 to +0.031 | 0.8314 | 6/10 |

## All-metric internal summaries

| Method | Metric | Mean | SD |
|:--|:--|--:|--:|
| cobra | MCC | 0.1793 | 0.0451 |
| cobra | AUC | 0.6490 | 0.0286 |
| cobra | AUPRC | 0.4660 | 0.0243 |
| cobra | F1 | 0.4161 | 0.0744 |
| cobra | Precision | 0.4793 | 0.0356 |
| cobra | Recall | 0.3939 | 0.1161 |
| cobra | GMean | 0.5371 | 0.0628 |
| duet_graph | MCC | 0.1704 | 0.0200 |
| duet_graph | AUC | 0.6257 | 0.0134 |
| duet_graph | AUPRC | 0.4583 | 0.0160 |
| duet_graph | F1 | 0.4088 | 0.0358 |
| duet_graph | Precision | 0.4773 | 0.0120 |
| duet_graph | Recall | 0.3615 | 0.0537 |
| duet_graph | GMean | 0.5336 | 0.0308 |
| duet_ensemble | MCC | 0.2274 | 0.0157 |
| duet_ensemble | AUC | 0.6722 | 0.0085 |
| duet_ensemble | AUPRC | 0.4939 | 0.0131 |
| duet_ensemble | F1 | 0.4734 | 0.0316 |
| duet_ensemble | Precision | 0.5028 | 0.0181 |
| duet_ensemble | Recall | 0.4581 | 0.0669 |
| duet_ensemble | GMean | 0.5867 | 0.0259 |
| graph_seq3d | MCC | 0.2123 | 0.0220 |
| graph_seq3d | AUC | 0.6455 | 0.0115 |
| graph_seq3d | AUPRC | 0.4719 | 0.0178 |
| graph_seq3d | F1 | 0.4624 | 0.0339 |
| graph_seq3d | Precision | 0.4924 | 0.0279 |
| graph_seq3d | Recall | 0.4443 | 0.0737 |
| graph_seq3d | GMean | 0.5770 | 0.0275 |
| graph_seqss3d | MCC | 0.1803 | 0.0251 |
| graph_seqss3d | AUC | 0.6217 | 0.0169 |
| graph_seqss3d | AUPRC | 0.4501 | 0.0199 |
| graph_seqss3d | F1 | 0.4535 | 0.0416 |
| graph_seqss3d | Precision | 0.4601 | 0.0138 |
| graph_seqss3d | Recall | 0.4557 | 0.0823 |
| graph_seqss3d | GMean | 0.5693 | 0.0323 |
| graph_seq3d_ensemble | MCC | 0.2392 | 0.0198 |
| graph_seq3d_ensemble | AUC | 0.6782 | 0.0088 |
| graph_seq3d_ensemble | AUPRC | 0.5008 | 0.0150 |
| graph_seq3d_ensemble | F1 | 0.5001 | 0.0312 |
| graph_seq3d_ensemble | Precision | 0.4933 | 0.0162 |
| graph_seq3d_ensemble | Recall | 0.5177 | 0.0743 |
| graph_seq3d_ensemble | GMean | 0.6074 | 0.0234 |
| graph_seqss3d_ensemble | MCC | 0.2361 | 0.0224 |
| graph_seqss3d_ensemble | AUC | 0.6693 | 0.0116 |
| graph_seqss3d_ensemble | AUPRC | 0.4940 | 0.0144 |
| graph_seqss3d_ensemble | F1 | 0.4691 | 0.0346 |
| graph_seqss3d_ensemble | Precision | 0.5169 | 0.0161 |
| graph_seqss3d_ensemble | Recall | 0.4378 | 0.0634 |
| graph_seqss3d_ensemble | GMean | 0.5834 | 0.0288 |
| grassp_rnafm | MCC | 0.2607 | 0.0135 |
| grassp_rnafm | AUC | 0.6860 | 0.0076 |
| grassp_rnafm | AUPRC | 0.5093 | 0.0114 |
| grassp_rnafm | F1 | 0.5480 | 0.0070 |
| grassp_rnafm | Precision | 0.4675 | 0.0108 |
| grassp_rnafm | Recall | 0.6635 | 0.0220 |
| grassp_rnafm | GMean | 0.6359 | 0.0070 |
