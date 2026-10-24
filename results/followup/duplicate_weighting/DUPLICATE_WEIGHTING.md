# Exact-duplicate weighting ablation

All three strategies use the same folds, ten seeds, selected epochs and thresholds. The deduplicated manifest removes exact duplicate records; label-conflicting exact families are dropped entirely.

## Metric summaries

| Model | Strategy | Metric | Mean | SD |
|:--|:--|:--|--:|--:|
| duet_graph | group | MCC | 0.1928 | 0.0169 |
| duet_graph | group | AUC | 0.6410 | 0.0136 |
| duet_graph | group | AUPRC | 0.4498 | 0.0141 |
| duet_graph | group | F1 | 0.4217 | 0.0285 |
| duet_graph | group | Precision | 0.4566 | 0.0113 |
| duet_graph | group | Recall | 0.4106 | 0.0502 |
| duet_graph | group | GMean | 0.5593 | 0.0256 |
| duet_graph | uniform | MCC | 0.1817 | 0.0198 |
| duet_graph | uniform | AUC | 0.6393 | 0.0106 |
| duet_graph | uniform | AUPRC | 0.4411 | 0.0088 |
| duet_graph | uniform | F1 | 0.4005 | 0.0477 |
| duet_graph | uniform | Precision | 0.4589 | 0.0196 |
| duet_graph | uniform | Recall | 0.3766 | 0.0800 |
| duet_graph | uniform | GMean | 0.5387 | 0.0442 |
| duet_graph | dedup | MCC | 0.1427 | 0.0161 |
| duet_graph | dedup | AUC | 0.6470 | 0.0043 |
| duet_graph | dedup | AUPRC | 0.4482 | 0.0052 |
| duet_graph | dedup | F1 | 0.2765 | 0.0508 |
| duet_graph | dedup | Precision | 0.4875 | 0.0232 |
| duet_graph | dedup | Recall | 0.2022 | 0.0507 |
| duet_graph | dedup | GMean | 0.4190 | 0.0508 |
| grassp_rnafm | group | MCC | 0.2607 | 0.0135 |
| grassp_rnafm | group | AUC | 0.6860 | 0.0076 |
| grassp_rnafm | group | AUPRC | 0.5093 | 0.0114 |
| grassp_rnafm | group | F1 | 0.5480 | 0.0070 |
| grassp_rnafm | group | Precision | 0.4675 | 0.0108 |
| grassp_rnafm | group | Recall | 0.6635 | 0.0220 |
| grassp_rnafm | group | GMean | 0.6359 | 0.0070 |
| grassp_rnafm | uniform | MCC | 0.2546 | 0.0173 |
| grassp_rnafm | uniform | AUC | 0.6875 | 0.0073 |
| grassp_rnafm | uniform | AUPRC | 0.5253 | 0.0132 |
| grassp_rnafm | uniform | F1 | 0.5432 | 0.0100 |
| grassp_rnafm | uniform | Precision | 0.4645 | 0.0134 |
| grassp_rnafm | uniform | Recall | 0.6555 | 0.0304 |
| grassp_rnafm | uniform | GMean | 0.6319 | 0.0092 |
| grassp_rnafm | dedup | MCC | 0.1184 | 0.1155 |
| grassp_rnafm | dedup | AUC | 0.6685 | 0.0069 |
| grassp_rnafm | dedup | AUPRC | 0.5015 | 0.0187 |
| grassp_rnafm | dedup | F1 | 0.5262 | 0.0211 |
| grassp_rnafm | dedup | Precision | 0.3826 | 0.0459 |
| grassp_rnafm | dedup | Recall | 0.8835 | 0.1337 |
| grassp_rnafm | dedup | GMean | 0.3039 | 0.3036 |

## Paired comparisons

| Model | Metric | Comparison | Delta | 95% CI | raw p | Holm p |
|:--|:--|:--|--:|:--|--:|--:|
| duet_graph | MCC | uniform_vs_group | -0.0112 | -0.0294 to +0.0070 | 0.1979 | 0.1979 |
| duet_graph | MCC | dedup_vs_group | -0.0502 | -0.0690 to -0.0313 | 0.0001973 | 0.000592 |
| duet_graph | MCC | dedup_vs_uniform | -0.0390 | -0.0591 to -0.0188 | 0.001782 | 0.003564 |
| duet_graph | AUC | uniform_vs_group | -0.0016 | -0.0094 to +0.0062 | 0.6492 | 0.6492 |
| duet_graph | AUC | dedup_vs_group | +0.0060 | -0.0032 to +0.0153 | 0.1747 | 0.3494 |
| duet_graph | AUC | dedup_vs_uniform | +0.0076 | +0.0012 to +0.0141 | 0.02589 | 0.07768 |
| duet_graph | AUPRC | uniform_vs_group | -0.0087 | -0.0174 to -0.0001 | 0.04845 | 0.0969 |
| duet_graph | AUPRC | dedup_vs_group | -0.0016 | -0.0089 to +0.0056 | 0.6205 | 0.6205 |
| duet_graph | AUPRC | dedup_vs_uniform | +0.0071 | +0.0014 to +0.0128 | 0.02065 | 0.06196 |
| duet_graph | F1 | uniform_vs_group | -0.0212 | -0.0599 to +0.0176 | 0.2477 | 0.2477 |
| duet_graph | F1 | dedup_vs_group | -0.1452 | -0.1922 to -0.0981 | 6.489e-05 | 0.0001947 |
| duet_graph | F1 | dedup_vs_uniform | -0.1240 | -0.1747 to -0.0733 | 0.0003645 | 0.000729 |
| duet_graph | Precision | uniform_vs_group | +0.0024 | -0.0107 to +0.0154 | 0.6894 | 0.6894 |
| duet_graph | Precision | dedup_vs_group | +0.0310 | +0.0112 to +0.0507 | 0.006264 | 0.01369 |
| duet_graph | Precision | dedup_vs_uniform | +0.0286 | +0.0113 to +0.0458 | 0.004562 | 0.01369 |
| duet_graph | Recall | uniform_vs_group | -0.0340 | -0.0981 to +0.0302 | 0.2616 | 0.2616 |
| duet_graph | Recall | dedup_vs_group | -0.2084 | -0.2667 to -0.1500 | 2.046e-05 | 6.137e-05 |
| duet_graph | Recall | dedup_vs_uniform | -0.1744 | -0.2400 to -0.1088 | 0.0001993 | 0.0003986 |
| duet_graph | GMean | uniform_vs_group | -0.0206 | -0.0566 to +0.0154 | 0.228 | 0.228 |
| duet_graph | GMean | dedup_vs_group | -0.1404 | -0.1859 to -0.0948 | 6.521e-05 | 0.0001956 |
| duet_graph | GMean | dedup_vs_uniform | -0.1198 | -0.1679 to -0.0716 | 0.0003231 | 0.0006462 |
| grassp_rnafm | MCC | uniform_vs_group | -0.0060 | -0.0209 to +0.0089 | 0.3868 | 0.3868 |
| grassp_rnafm | MCC | dedup_vs_group | -0.1422 | -0.2243 to -0.0602 | 0.003502 | 0.01051 |
| grassp_rnafm | MCC | dedup_vs_uniform | -0.1362 | -0.2187 to -0.0538 | 0.004636 | 0.01051 |
| grassp_rnafm | AUC | uniform_vs_group | +0.0016 | -0.0043 to +0.0074 | 0.5636 | 0.5636 |
| grassp_rnafm | AUC | dedup_vs_group | -0.0175 | -0.0240 to -0.0110 | 0.0001836 | 0.0003672 |
| grassp_rnafm | AUC | dedup_vs_uniform | -0.0190 | -0.0256 to -0.0124 | 0.0001065 | 0.0003195 |
| grassp_rnafm | AUPRC | uniform_vs_group | +0.0160 | +0.0055 to +0.0265 | 0.007334 | 0.022 |
| grassp_rnafm | AUPRC | dedup_vs_group | -0.0078 | -0.0213 to +0.0058 | 0.2256 | 0.2256 |
| grassp_rnafm | AUPRC | dedup_vs_uniform | -0.0238 | -0.0400 to -0.0075 | 0.009037 | 0.022 |
| grassp_rnafm | F1 | uniform_vs_group | -0.0048 | -0.0146 to +0.0050 | 0.2978 | 0.2978 |
| grassp_rnafm | F1 | dedup_vs_group | -0.0218 | -0.0385 to -0.0052 | 0.01582 | 0.04746 |
| grassp_rnafm | F1 | dedup_vs_uniform | -0.0170 | -0.0343 to +0.0002 | 0.0522 | 0.1044 |
| grassp_rnafm | Precision | uniform_vs_group | -0.0029 | -0.0121 to +0.0062 | 0.4902 | 0.4902 |
| grassp_rnafm | Precision | dedup_vs_group | -0.0849 | -0.1152 to -0.0546 | 0.0001355 | 0.0004065 |
| grassp_rnafm | Precision | dedup_vs_uniform | -0.0820 | -0.1129 to -0.0511 | 0.0002026 | 0.0004065 |
| grassp_rnafm | Recall | uniform_vs_group | -0.0080 | -0.0345 to +0.0186 | 0.5152 | 0.5152 |
| grassp_rnafm | Recall | dedup_vs_group | +0.2200 | +0.1282 to +0.3118 | 0.0004204 | 0.0008407 |
| grassp_rnafm | Recall | dedup_vs_uniform | +0.2279 | +0.1453 to +0.3106 | 0.0001522 | 0.0004567 |
| grassp_rnafm | GMean | uniform_vs_group | -0.0039 | -0.0116 to +0.0038 | 0.2794 | 0.2794 |
| grassp_rnafm | GMean | dedup_vs_group | -0.3320 | -0.5484 to -0.1156 | 0.007046 | 0.02114 |
| grassp_rnafm | GMean | dedup_vs_uniform | -0.3280 | -0.5447 to -0.1114 | 0.00756 | 0.02114 |
