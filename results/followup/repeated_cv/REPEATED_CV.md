# Repeated cluster-stratified CV

Three independent five-fold cluster assignments were generated. Each assignment keeps whole 80% identity components together and balances chain and positive-residue counts. Final models use the same five paired seeds for every partition. The reduced seed count is explicit because each partition contains five CV fits and five final fits for each method.

## Partition balance

| Assignment | Fold sizes | Positive residues |
|--:|:--|:--|
| 101 | [58, 58, 58, 58, 58] | [895, 894, 894, 894, 894] |
| 102 | [58, 58, 58, 58, 58] | [892, 893, 902, 892, 892] |
| 103 | [58, 58, 58, 58, 58] | [894, 894, 896, 894, 893] |

## Selected hyperparameters

| Assignment | Method | Epoch | Threshold |
|--:|:--|--:|--:|
| 101 | cobra | 15 | 0.4700 |
| 101 | duet_graph | 25 | 0.5000 |
| 101 | duet_tcn | 19 | 0.4900 |
| 101 | duet_dense | 30 | 0.5400 |
| 101 | duet_ensemble | 25 | 0.4700 |
| 101 | graph_seq3d | 20 | 0.5000 |
| 101 | graph_seq3d_ensemble | 19 | 0.4700 |
| 101 | grassp_rnafm | 76 | 0.5000 |
| 102 | cobra | 11 | 0.4900 |
| 102 | duet_graph | 13 | 0.5000 |
| 102 | duet_tcn | 17 | 0.4600 |
| 102 | duet_dense | 16 | 0.5000 |
| 102 | duet_ensemble | 16 | 0.4400 |
| 102 | graph_seq3d | 15 | 0.5000 |
| 102 | graph_seq3d_ensemble | 16 | 0.4300 |
| 102 | grassp_rnafm | 84 | 0.6200 |
| 103 | cobra | 23 | 0.4100 |
| 103 | duet_graph | 18 | 0.5000 |
| 103 | duet_tcn | 18 | 0.5200 |
| 103 | duet_dense | 17 | 0.4900 |
| 103 | duet_ensemble | 17 | 0.5100 |
| 103 | graph_seq3d | 16 | 0.4800 |
| 103 | graph_seq3d_ensemble | 17 | 0.4400 |
| 103 | grassp_rnafm | 91 | 0.5000 |

## Final-test comparisons

| Assignment | Metric | Comparison | Delta | 95% CI | raw p | Holm p |
|--:|:--|:--|--:|:--|--:|--:|
| 101 | MCC | duet_ensemble_vs_cobra | +0.0633 | -0.0048 to +0.1313 | 0.06134 | 0.1227 |
| 101 | MCC | grassp_rnafm_vs_duet_ensemble | +0.0398 | +0.0140 to +0.0655 | 0.01274 | 0.03823 |
| 101 | MCC | graph_seq3d_vs_duet_graph | +0.0243 | -0.0085 to +0.0571 | 0.1085 | 0.1227 |
| 101 | AUC | duet_ensemble_vs_cobra | +0.0075 | -0.0160 to +0.0310 | 0.425 | 0.425 |
| 101 | AUC | grassp_rnafm_vs_duet_ensemble | +0.0318 | +0.0239 to +0.0396 | 0.0003565 | 0.001069 |
| 101 | AUC | graph_seq3d_vs_duet_graph | +0.0185 | +0.0060 to +0.0310 | 0.01455 | 0.0291 |
| 101 | AUPRC | duet_ensemble_vs_cobra | +0.0103 | -0.0104 to +0.0311 | 0.2388 | 0.2388 |
| 101 | AUPRC | grassp_rnafm_vs_duet_ensemble | +0.0339 | +0.0228 to +0.0450 | 0.001058 | 0.003174 |
| 101 | AUPRC | graph_seq3d_vs_duet_graph | +0.0304 | +0.0121 to +0.0487 | 0.009928 | 0.01986 |
| 101 | F1 | duet_ensemble_vs_cobra | +0.1016 | -0.0309 to +0.2341 | 0.1004 | 0.2008 |
| 101 | F1 | grassp_rnafm_vs_duet_ensemble | +0.0828 | +0.0765 to +0.0890 | 3.198e-06 | 9.595e-06 |
| 101 | F1 | graph_seq3d_vs_duet_graph | +0.0139 | -0.0264 to +0.0542 | 0.393 | 0.393 |
| 101 | Precision | duet_ensemble_vs_cobra | +0.0141 | -0.0192 to +0.0474 | 0.3056 | 0.3708 |
| 101 | Precision | grassp_rnafm_vs_duet_ensemble | -0.0347 | -0.0709 to +0.0015 | 0.05626 | 0.1688 |
| 101 | Precision | graph_seq3d_vs_duet_graph | +0.0184 | -0.0136 to +0.0505 | 0.1854 | 0.3708 |
| 101 | Recall | duet_ensemble_vs_cobra | +0.1407 | -0.0582 to +0.3396 | 0.1211 | 0.2421 |
| 101 | Recall | grassp_rnafm_vs_duet_ensemble | +0.2295 | +0.1831 to +0.2759 | 0.0001626 | 0.0004877 |
| 101 | Recall | graph_seq3d_vs_duet_graph | +0.0117 | -0.0609 to +0.0844 | 0.6775 | 0.6775 |
| 101 | GMean | duet_ensemble_vs_cobra | +0.0919 | -0.0304 to +0.2141 | 0.1052 | 0.2104 |
| 101 | GMean | grassp_rnafm_vs_duet_ensemble | +0.0523 | +0.0472 to +0.0574 | 9.051e-06 | 2.715e-05 |
| 101 | GMean | graph_seq3d_vs_duet_graph | +0.0112 | -0.0216 to +0.0440 | 0.3963 | 0.3963 |
| 102 | MCC | duet_ensemble_vs_cobra | +0.0772 | +0.0337 to +0.1206 | 0.007869 | 0.02361 |
| 102 | MCC | grassp_rnafm_vs_duet_ensemble | +0.0368 | -0.0057 to +0.0793 | 0.07395 | 0.1479 |
| 102 | MCC | graph_seq3d_vs_duet_graph | +0.0038 | -0.0246 to +0.0323 | 0.7278 | 0.7278 |
| 102 | AUC | duet_ensemble_vs_cobra | +0.0324 | +0.0079 to +0.0568 | 0.02144 | 0.06433 |
| 102 | AUC | grassp_rnafm_vs_duet_ensemble | +0.0076 | -0.0060 to +0.0213 | 0.1963 | 0.1963 |
| 102 | AUC | graph_seq3d_vs_duet_graph | -0.0182 | -0.0355 to -0.0008 | 0.04401 | 0.08802 |
| 102 | AUPRC | duet_ensemble_vs_cobra | +0.0393 | +0.0110 to +0.0677 | 0.01817 | 0.05451 |
| 102 | AUPRC | grassp_rnafm_vs_duet_ensemble | +0.0078 | -0.0123 to +0.0280 | 0.3421 | 0.3421 |
| 102 | AUPRC | graph_seq3d_vs_duet_graph | -0.0181 | -0.0445 to +0.0082 | 0.1284 | 0.2568 |
| 102 | F1 | duet_ensemble_vs_cobra | +0.1085 | -0.0265 to +0.2436 | 0.08941 | 0.2682 |
| 102 | F1 | grassp_rnafm_vs_duet_ensemble | +0.0460 | -0.0153 to +0.1072 | 0.1054 | 0.2682 |
| 102 | F1 | graph_seq3d_vs_duet_graph | +0.0476 | -0.0311 to +0.1264 | 0.1685 | 0.2682 |
| 102 | Precision | duet_ensemble_vs_cobra | +0.0200 | -0.0259 to +0.0660 | 0.2931 | 0.5862 |
| 102 | Precision | grassp_rnafm_vs_duet_ensemble | -0.0001 | -0.0051 to +0.0049 | 0.9507 | 0.9507 |
| 102 | Precision | graph_seq3d_vs_duet_graph | -0.0236 | -0.0552 to +0.0079 | 0.1062 | 0.3185 |
| 102 | Recall | duet_ensemble_vs_cobra | +0.1397 | -0.1525 to +0.4320 | 0.2551 | 0.2918 |
| 102 | Recall | grassp_rnafm_vs_duet_ensemble | +0.0990 | -0.0226 to +0.2206 | 0.08666 | 0.26 |
| 102 | Recall | graph_seq3d_vs_duet_graph | +0.0839 | -0.0453 to +0.2131 | 0.1459 | 0.2918 |
| 102 | GMean | duet_ensemble_vs_cobra | +0.1078 | +0.0051 to +0.2104 | 0.04353 | 0.1306 |
| 102 | GMean | grassp_rnafm_vs_duet_ensemble | +0.0337 | -0.0141 to +0.0815 | 0.1217 | 0.2433 |
| 102 | GMean | graph_seq3d_vs_duet_graph | +0.0413 | -0.0271 to +0.1097 | 0.1687 | 0.2433 |
| 103 | MCC | duet_ensemble_vs_cobra | -0.0078 | -0.0320 to +0.0164 | 0.4199 | 0.4199 |
| 103 | MCC | grassp_rnafm_vs_duet_ensemble | +0.0416 | +0.0044 to +0.0788 | 0.03613 | 0.1084 |
| 103 | MCC | graph_seq3d_vs_duet_graph | +0.0221 | -0.0166 to +0.0608 | 0.1884 | 0.3768 |
| 103 | AUC | duet_ensemble_vs_cobra | -0.0007 | -0.0115 to +0.0102 | 0.8745 | 1 |
| 103 | AUC | grassp_rnafm_vs_duet_ensemble | +0.0176 | +0.0051 to +0.0300 | 0.01718 | 0.05154 |
| 103 | AUC | graph_seq3d_vs_duet_graph | +0.0056 | -0.0292 to +0.0403 | 0.6789 | 1 |
| 103 | AUPRC | duet_ensemble_vs_cobra | +0.0028 | -0.0083 to +0.0138 | 0.5263 | 0.8833 |
| 103 | AUPRC | grassp_rnafm_vs_duet_ensemble | +0.0124 | -0.0036 to +0.0284 | 0.09854 | 0.2956 |
| 103 | AUPRC | graph_seq3d_vs_duet_graph | +0.0127 | -0.0287 to +0.0542 | 0.4416 | 0.8833 |
| 103 | F1 | duet_ensemble_vs_cobra | -0.0270 | -0.0686 to +0.0146 | 0.1455 | 0.291 |
| 103 | F1 | grassp_rnafm_vs_duet_ensemble | +0.1057 | +0.0522 to +0.1592 | 0.00538 | 0.01614 |
| 103 | F1 | graph_seq3d_vs_duet_graph | +0.0239 | -0.0424 to +0.0903 | 0.3735 | 0.3735 |
| 103 | Precision | duet_ensemble_vs_cobra | +0.0151 | +0.0082 to +0.0219 | 0.003662 | 0.01099 |
| 103 | Precision | grassp_rnafm_vs_duet_ensemble | -0.0445 | -0.0729 to -0.0161 | 0.0121 | 0.02421 |
| 103 | Precision | graph_seq3d_vs_duet_graph | +0.0073 | -0.0146 to +0.0291 | 0.4083 | 0.4083 |
| 103 | Recall | duet_ensemble_vs_cobra | -0.0799 | -0.1556 to -0.0043 | 0.04257 | 0.08514 |
| 103 | Recall | grassp_rnafm_vs_duet_ensemble | +0.2620 | +0.1669 to +0.3570 | 0.001568 | 0.004705 |
| 103 | Recall | graph_seq3d_vs_duet_graph | +0.0441 | -0.0871 to +0.1752 | 0.4037 | 0.4037 |
| 103 | GMean | duet_ensemble_vs_cobra | -0.0190 | -0.0542 to +0.0162 | 0.2085 | 0.4171 |
| 103 | GMean | grassp_rnafm_vs_duet_ensemble | +0.0751 | +0.0299 to +0.1202 | 0.009901 | 0.0297 |
| 103 | GMean | graph_seq3d_vs_duet_graph | +0.0178 | -0.0334 to +0.0691 | 0.3891 | 0.4171 |

## All-metric method summaries

| Assignment | Method | Metric | Mean | SD |
|--:|:--|:--|--:|--:|
| 101 | cobra | MCC | 0.1603 | 0.0479 |
| 101 | cobra | AUC | 0.6601 | 0.0131 |
| 101 | cobra | AUPRC | 0.4501 | 0.0183 |
| 101 | cobra | F1 | 0.3629 | 0.1021 |
| 101 | cobra | Precision | 0.4432 | 0.0155 |
| 101 | cobra | Recall | 0.3769 | 0.1494 |
| 101 | cobra | GMean | 0.5024 | 0.0931 |
| 101 | duet_graph | MCC | 0.1538 | 0.0194 |
| 101 | duet_graph | AUC | 0.6108 | 0.0111 |
| 101 | duet_graph | AUPRC | 0.4134 | 0.0080 |
| 101 | duet_graph | F1 | 0.4155 | 0.0123 |
| 101 | duet_graph | Precision | 0.4159 | 0.0158 |
| 101 | duet_graph | Recall | 0.4387 | 0.0208 |
| 101 | duet_graph | GMean | 0.5555 | 0.0104 |
| 101 | duet_tcn | MCC | 0.1988 | 0.0094 |
| 101 | duet_tcn | AUC | 0.6513 | 0.0190 |
| 101 | duet_tcn | AUPRC | 0.4409 | 0.0137 |
| 101 | duet_tcn | F1 | 0.4288 | 0.0095 |
| 101 | duet_tcn | Precision | 0.4570 | 0.0078 |
| 101 | duet_tcn | Recall | 0.4426 | 0.0114 |
| 101 | duet_tcn | GMean | 0.5661 | 0.0090 |
| 101 | duet_dense | MCC | 0.2027 | 0.0183 |
| 101 | duet_dense | AUC | 0.6673 | 0.0084 |
| 101 | duet_dense | AUPRC | 0.4499 | 0.0071 |
| 101 | duet_dense | F1 | 0.4272 | 0.0088 |
| 101 | duet_dense | Precision | 0.4586 | 0.0206 |
| 101 | duet_dense | Recall | 0.4440 | 0.0179 |
| 101 | duet_dense | GMean | 0.5654 | 0.0076 |
| 101 | duet_ensemble | MCC | 0.2236 | 0.0151 |
| 101 | duet_ensemble | AUC | 0.6676 | 0.0074 |
| 101 | duet_ensemble | AUPRC | 0.4604 | 0.0082 |
| 101 | duet_ensemble | F1 | 0.4645 | 0.0063 |
| 101 | duet_ensemble | Precision | 0.4573 | 0.0161 |
| 101 | duet_ensemble | Recall | 0.5176 | 0.0182 |
| 101 | duet_ensemble | GMean | 0.5942 | 0.0068 |
| 101 | graph_seq3d | MCC | 0.1646 | 0.0184 |
| 101 | graph_seq3d | AUC | 0.6144 | 0.0075 |
| 101 | graph_seq3d | AUPRC | 0.4549 | 0.0084 |
| 101 | graph_seq3d | F1 | 0.4247 | 0.0253 |
| 101 | graph_seq3d | Precision | 0.4614 | 0.0151 |
| 101 | graph_seq3d | Recall | 0.3966 | 0.0433 |
| 101 | graph_seq3d | GMean | 0.5478 | 0.0208 |
| 101 | graph_seq3d_ensemble | MCC | 0.2293 | 0.0130 |
| 101 | graph_seq3d_ensemble | AUC | 0.6645 | 0.0053 |
| 101 | graph_seq3d_ensemble | AUPRC | 0.4889 | 0.0097 |
| 101 | graph_seq3d_ensemble | F1 | 0.4706 | 0.0103 |
| 101 | graph_seq3d_ensemble | Precision | 0.5077 | 0.0139 |
| 101 | graph_seq3d_ensemble | Recall | 0.4452 | 0.0189 |
| 101 | graph_seq3d_ensemble | GMean | 0.5853 | 0.0085 |
| 101 | grassp_rnafm | MCC | 0.2560 | 0.0105 |
| 101 | grassp_rnafm | AUC | 0.6874 | 0.0041 |
| 101 | grassp_rnafm | AUPRC | 0.5100 | 0.0040 |
| 101 | grassp_rnafm | F1 | 0.5472 | 0.0060 |
| 101 | grassp_rnafm | Precision | 0.4621 | 0.0120 |
| 101 | grassp_rnafm | Recall | 0.6727 | 0.0314 |
| 101 | grassp_rnafm | GMean | 0.6325 | 0.0055 |
| 102 | cobra | MCC | 0.1569 | 0.0327 |
| 102 | cobra | AUC | 0.6522 | 0.0170 |
| 102 | cobra | AUPRC | 0.4411 | 0.0183 |
| 102 | cobra | F1 | 0.3802 | 0.1050 |
| 102 | cobra | Precision | 0.4263 | 0.0296 |
| 102 | cobra | Recall | 0.4398 | 0.2246 |
| 102 | cobra | GMean | 0.5040 | 0.0796 |
| 102 | duet_graph | MCC | 0.1933 | 0.0218 |
| 102 | duet_graph | AUC | 0.6494 | 0.0102 |
| 102 | duet_graph | AUPRC | 0.4570 | 0.0102 |
| 102 | duet_graph | F1 | 0.4013 | 0.0502 |
| 102 | duet_graph | Precision | 0.4708 | 0.0152 |
| 102 | duet_graph | Recall | 0.3714 | 0.0850 |
| 102 | duet_graph | GMean | 0.5395 | 0.0463 |
| 102 | duet_tcn | MCC | 0.2182 | 0.0225 |
| 102 | duet_tcn | AUC | 0.6624 | 0.0133 |
| 102 | duet_tcn | AUPRC | 0.4489 | 0.0179 |
| 102 | duet_tcn | F1 | 0.4615 | 0.0290 |
| 102 | duet_tcn | Precision | 0.4524 | 0.0076 |
| 102 | duet_tcn | Recall | 0.5052 | 0.0550 |
| 102 | duet_tcn | GMean | 0.5934 | 0.0237 |
| 102 | duet_dense | MCC | 0.1839 | 0.0482 |
| 102 | duet_dense | AUC | 0.6579 | 0.0130 |
| 102 | duet_dense | AUPRC | 0.4513 | 0.0183 |
| 102 | duet_dense | F1 | 0.3782 | 0.0704 |
| 102 | duet_dense | Precision | 0.4662 | 0.0317 |
| 102 | duet_dense | Recall | 0.3610 | 0.0871 |
| 102 | duet_dense | GMean | 0.5202 | 0.0654 |
| 102 | duet_ensemble | MCC | 0.2340 | 0.0274 |
| 102 | duet_ensemble | AUC | 0.6845 | 0.0112 |
| 102 | duet_ensemble | AUPRC | 0.4805 | 0.0177 |
| 102 | duet_ensemble | F1 | 0.4887 | 0.0346 |
| 102 | duet_ensemble | Precision | 0.4463 | 0.0129 |
| 102 | duet_ensemble | Recall | 0.5795 | 0.0759 |
| 102 | duet_ensemble | GMean | 0.6118 | 0.0267 |
| 102 | graph_seq3d | MCC | 0.1726 | 0.0143 |
| 102 | graph_seq3d | AUC | 0.6165 | 0.0086 |
| 102 | graph_seq3d | AUPRC | 0.4490 | 0.0128 |
| 102 | graph_seq3d | F1 | 0.4314 | 0.0349 |
| 102 | graph_seq3d | Precision | 0.4666 | 0.0140 |
| 102 | graph_seq3d | Recall | 0.4072 | 0.0652 |
| 102 | graph_seq3d | GMean | 0.5524 | 0.0291 |
| 102 | graph_seq3d_ensemble | MCC | 0.2267 | 0.0254 |
| 102 | graph_seq3d_ensemble | AUC | 0.6686 | 0.0089 |
| 102 | graph_seq3d_ensemble | AUPRC | 0.4863 | 0.0123 |
| 102 | graph_seq3d_ensemble | F1 | 0.5160 | 0.0343 |
| 102 | graph_seq3d_ensemble | Precision | 0.4616 | 0.0127 |
| 102 | graph_seq3d_ensemble | Recall | 0.5957 | 0.0902 |
| 102 | graph_seq3d_ensemble | GMean | 0.6127 | 0.0220 |
| 102 | grassp_rnafm | MCC | 0.2675 | 0.0206 |
| 102 | grassp_rnafm | AUC | 0.6829 | 0.0065 |
| 102 | grassp_rnafm | AUPRC | 0.4995 | 0.0054 |
| 102 | grassp_rnafm | F1 | 0.5432 | 0.0155 |
| 102 | grassp_rnafm | Precision | 0.4830 | 0.0112 |
| 102 | grassp_rnafm | Recall | 0.6219 | 0.0326 |
| 102 | grassp_rnafm | GMean | 0.6387 | 0.0121 |
| 103 | cobra | MCC | 0.2284 | 0.0108 |
| 103 | cobra | AUC | 0.6719 | 0.0120 |
| 103 | cobra | AUPRC | 0.4623 | 0.0105 |
| 103 | cobra | F1 | 0.4705 | 0.0268 |
| 103 | cobra | Precision | 0.4559 | 0.0164 |
| 103 | cobra | Recall | 0.5414 | 0.0723 |
| 103 | cobra | GMean | 0.5968 | 0.0175 |
| 103 | duet_graph | MCC | 0.1546 | 0.0101 |
| 103 | duet_graph | AUC | 0.6150 | 0.0176 |
| 103 | duet_graph | AUPRC | 0.4209 | 0.0151 |
| 103 | duet_graph | F1 | 0.4171 | 0.0130 |
| 103 | duet_graph | Precision | 0.4171 | 0.0106 |
| 103 | duet_graph | Recall | 0.4386 | 0.0279 |
| 103 | duet_graph | GMean | 0.5565 | 0.0105 |
| 103 | duet_tcn | MCC | 0.2150 | 0.0079 |
| 103 | duet_tcn | AUC | 0.6608 | 0.0057 |
| 103 | duet_tcn | AUPRC | 0.4484 | 0.0062 |
| 103 | duet_tcn | F1 | 0.4614 | 0.0173 |
| 103 | duet_tcn | Precision | 0.4506 | 0.0150 |
| 103 | duet_tcn | Recall | 0.5114 | 0.0510 |
| 103 | duet_tcn | GMean | 0.5923 | 0.0125 |
| 103 | duet_dense | MCC | 0.1778 | 0.0218 |
| 103 | duet_dense | AUC | 0.6499 | 0.0066 |
| 103 | duet_dense | AUPRC | 0.4446 | 0.0102 |
| 103 | duet_dense | F1 | 0.4073 | 0.0530 |
| 103 | duet_dense | Precision | 0.4425 | 0.0331 |
| 103 | duet_dense | Recall | 0.4280 | 0.1231 |
| 103 | duet_dense | GMean | 0.5432 | 0.0407 |
| 103 | duet_ensemble | MCC | 0.2206 | 0.0208 |
| 103 | duet_ensemble | AUC | 0.6713 | 0.0068 |
| 103 | duet_ensemble | AUPRC | 0.4651 | 0.0060 |
| 103 | duet_ensemble | F1 | 0.4435 | 0.0348 |
| 103 | duet_ensemble | Precision | 0.4710 | 0.0146 |
| 103 | duet_ensemble | Recall | 0.4615 | 0.0645 |
| 103 | duet_ensemble | GMean | 0.5778 | 0.0294 |
| 103 | graph_seq3d | MCC | 0.1685 | 0.0338 |
| 103 | graph_seq3d | AUC | 0.6102 | 0.0225 |
| 103 | graph_seq3d | AUPRC | 0.4492 | 0.0231 |
| 103 | graph_seq3d | F1 | 0.4406 | 0.0516 |
| 103 | graph_seq3d | Precision | 0.4537 | 0.0133 |
| 103 | graph_seq3d | Recall | 0.4374 | 0.0985 |
| 103 | graph_seq3d | GMean | 0.5589 | 0.0396 |
| 103 | graph_seq3d_ensemble | MCC | 0.2180 | 0.0251 |
| 103 | graph_seq3d_ensemble | AUC | 0.6657 | 0.0145 |
| 103 | graph_seq3d_ensemble | AUPRC | 0.4885 | 0.0165 |
| 103 | graph_seq3d_ensemble | F1 | 0.5177 | 0.0246 |
| 103 | graph_seq3d_ensemble | Precision | 0.4526 | 0.0199 |
| 103 | graph_seq3d_ensemble | Recall | 0.6161 | 0.0826 |
| 103 | graph_seq3d_ensemble | GMean | 0.6083 | 0.0196 |
| 103 | grassp_rnafm | MCC | 0.2542 | 0.0157 |
| 103 | grassp_rnafm | AUC | 0.6780 | 0.0075 |
| 103 | grassp_rnafm | AUPRC | 0.4930 | 0.0111 |
| 103 | grassp_rnafm | F1 | 0.5433 | 0.0094 |
| 103 | grassp_rnafm | Precision | 0.4659 | 0.0123 |
| 103 | grassp_rnafm | Recall | 0.6538 | 0.0319 |
| 103 | grassp_rnafm | GMean | 0.6321 | 0.0088 |
