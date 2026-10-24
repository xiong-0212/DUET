# Large-data shuffled-3D control

The shuffle preserves every node's structural out-degree, sequence edges, node features, labels and folds. Three small graphs required duplicate targets because their degree exceeded the number of non-self candidates; this is recorded in `graph_audit.json`.

## Method summaries

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
| graph_seq3d | MCC | 0.2123 | 0.0220 |
| graph_seq3d | AUC | 0.6455 | 0.0115 |
| graph_seq3d | AUPRC | 0.4719 | 0.0178 |
| graph_seq3d | F1 | 0.4624 | 0.0339 |
| graph_seq3d | Precision | 0.4924 | 0.0279 |
| graph_seq3d | Recall | 0.4443 | 0.0737 |
| graph_seq3d | GMean | 0.5770 | 0.0275 |
| graph_seq3d_shuffle | MCC | 0.1757 | 0.0260 |
| graph_seq3d_shuffle | AUC | 0.6320 | 0.0113 |
| graph_seq3d_shuffle | AUPRC | 0.4605 | 0.0128 |
| graph_seq3d_shuffle | F1 | 0.3924 | 0.0533 |
| graph_seq3d_shuffle | Precision | 0.4942 | 0.0274 |
| graph_seq3d_shuffle | Recall | 0.3346 | 0.0796 |
| graph_seq3d_shuffle | GMean | 0.5182 | 0.0466 |
| graph_seqss3d | MCC | 0.1803 | 0.0251 |
| graph_seqss3d | AUC | 0.6217 | 0.0169 |
| graph_seqss3d | AUPRC | 0.4501 | 0.0199 |
| graph_seqss3d | F1 | 0.4535 | 0.0416 |
| graph_seqss3d | Precision | 0.4601 | 0.0138 |
| graph_seqss3d | Recall | 0.4557 | 0.0823 |
| graph_seqss3d | GMean | 0.5693 | 0.0323 |

## Paired comparisons

| Metric | Comparison | Delta | 95% CI | raw p | Holm p |
|:--|:--|--:|:--|--:|--:|
| MCC | shuffled_vs_real3d | -0.0366 | -0.0462 to -0.0270 | 1.235e-05 | 7.411e-05 |
| MCC | shuffled_vs_seqss3d | -0.0046 | -0.0280 to +0.0187 | 0.6659 | 1 |
| MCC | shuffled_vs_seqss | +0.0053 | -0.0100 to +0.0205 | 0.4549 | 1 |
| MCC | shuffled_vs_cobra | -0.0036 | -0.0453 to +0.0381 | 0.8495 | 1 |
| MCC | real3d_vs_seqss | +0.0418 | +0.0277 to +0.0560 | 8.954e-05 | 0.0004477 |
| MCC | real3d_vs_seqss3d | +0.0320 | +0.0132 to +0.0508 | 0.003928 | 0.01571 |
| AUC | shuffled_vs_real3d | -0.0135 | -0.0190 to -0.0079 | 0.0004107 | 0.002054 |
| AUC | shuffled_vs_seqss3d | +0.0103 | -0.0029 to +0.0234 | 0.1103 | 0.2206 |
| AUC | shuffled_vs_seqss | +0.0063 | +0.0002 to +0.0124 | 0.04371 | 0.1311 |
| AUC | shuffled_vs_cobra | -0.0170 | -0.0401 to +0.0061 | 0.13 | 0.2206 |
| AUC | real3d_vs_seqss | +0.0198 | +0.0129 to +0.0267 | 0.0001132 | 0.0006789 |
| AUC | real3d_vs_seqss3d | +0.0237 | +0.0107 to +0.0368 | 0.002585 | 0.01034 |
| AUPRC | shuffled_vs_real3d | -0.0114 | -0.0187 to -0.0041 | 0.006503 | 0.03902 |
| AUPRC | shuffled_vs_seqss3d | +0.0105 | -0.0043 to +0.0253 | 0.1445 | 0.4335 |
| AUPRC | shuffled_vs_seqss | +0.0022 | -0.0062 to +0.0106 | 0.5656 | 1 |
| AUPRC | shuffled_vs_cobra | -0.0055 | -0.0275 to +0.0166 | 0.5881 | 1 |
| AUPRC | real3d_vs_seqss | +0.0136 | +0.0035 to +0.0237 | 0.01378 | 0.06891 |
| AUPRC | real3d_vs_seqss3d | +0.0218 | +0.0050 to +0.0386 | 0.01643 | 0.06891 |
| F1 | shuffled_vs_real3d | -0.0700 | -0.0852 to -0.0548 | 2.531e-06 | 1.518e-05 |
| F1 | shuffled_vs_seqss3d | -0.0611 | -0.1069 to -0.0153 | 0.01453 | 0.0581 |
| F1 | shuffled_vs_seqss | -0.0164 | -0.0629 to +0.0301 | 0.4459 | 1 |
| F1 | shuffled_vs_cobra | -0.0237 | -0.1052 to +0.0578 | 0.5269 | 1 |
| F1 | real3d_vs_seqss | +0.0536 | +0.0199 to +0.0873 | 0.005738 | 0.02869 |
| F1 | real3d_vs_seqss3d | +0.0089 | -0.0276 to +0.0454 | 0.5936 | 1 |
| Precision | shuffled_vs_real3d | +0.0018 | -0.0073 to +0.0109 | 0.661 | 0.7325 |
| Precision | shuffled_vs_seqss3d | +0.0342 | +0.0129 to +0.0554 | 0.005391 | 0.03235 |
| Precision | shuffled_vs_seqss | +0.0169 | -0.0056 to +0.0394 | 0.1234 | 0.4935 |
| Precision | shuffled_vs_cobra | +0.0150 | -0.0206 to +0.0505 | 0.3663 | 0.7325 |
| Precision | real3d_vs_seqss | +0.0151 | -0.0100 to +0.0402 | 0.207 | 0.621 |
| Precision | real3d_vs_seqss3d | +0.0323 | +0.0120 to +0.0526 | 0.005699 | 0.03235 |
| Recall | shuffled_vs_real3d | -0.1097 | -0.1212 to -0.0982 | 4.766e-09 | 2.859e-08 |
| Recall | shuffled_vs_seqss3d | -0.1211 | -0.2012 to -0.0410 | 0.007635 | 0.03817 |
| Recall | shuffled_vs_seqss | -0.0269 | -0.1014 to +0.0475 | 0.4344 | 0.9581 |
| Recall | shuffled_vs_cobra | -0.0593 | -0.1867 to +0.0680 | 0.3194 | 0.9581 |
| Recall | real3d_vs_seqss | +0.0828 | +0.0131 to +0.1525 | 0.02491 | 0.09965 |
| Recall | real3d_vs_seqss3d | -0.0114 | -0.0898 to +0.0670 | 0.7499 | 0.9581 |
| GMean | shuffled_vs_real3d | -0.0588 | -0.0734 to -0.0442 | 7.747e-06 | 4.648e-05 |
| GMean | shuffled_vs_seqss3d | -0.0511 | -0.0896 to -0.0127 | 0.01473 | 0.05892 |
| GMean | shuffled_vs_seqss | -0.0154 | -0.0565 to +0.0256 | 0.4162 | 1 |
| GMean | shuffled_vs_cobra | -0.0189 | -0.0886 to +0.0507 | 0.5537 | 1 |
| GMean | real3d_vs_seqss | +0.0433 | +0.0147 to +0.0720 | 0.007629 | 0.03814 |
| GMean | real3d_vs_seqss3d | +0.0077 | -0.0213 to +0.0367 | 0.5643 | 1 |

## Chain bootstrap

| Metric | Comparison | Delta | 95% CI | raw p | Holm p | valid B |
|:--|:--|--:|:--|--:|--:|--:|
| MCC | shuffled_vs_real3d | -0.0366 | -0.0702 to -0.0012 | 0.04098 | 0.2049 | 2000 |
| AUC | shuffled_vs_real3d | -0.0135 | -0.0319 to +0.0065 | 0.1689 | 0.6757 | 2000 |
| AUPRC | shuffled_vs_real3d | -0.0114 | -0.0366 to +0.0124 | 0.3188 | 1 | 2000 |
| F1 | shuffled_vs_real3d | -0.0700 | -0.0948 to -0.0416 | 0.0009995 | 0.005997 | 2000 |
| Precision | shuffled_vs_real3d | +0.0018 | -0.0210 to +0.0267 | 0.8506 | 1 | 2000 |
| Recall | shuffled_vs_real3d | -0.1097 | -0.1394 to -0.0789 | 0.0009995 | 0.005997 | 2000 |
| GMean | shuffled_vs_real3d | -0.0588 | -0.0795 to -0.0353 | 0.0009995 | 0.005997 | 2000 |
| MCC | shuffled_vs_seqss3d | -0.0046 | -0.0566 to +0.0453 | 0.8456 | 1 | 2000 |
| AUC | shuffled_vs_seqss3d | +0.0103 | -0.0203 to +0.0396 | 0.5387 | 1 | 2000 |
| AUPRC | shuffled_vs_seqss3d | +0.0105 | -0.0230 to +0.0391 | 0.6197 | 1 | 2000 |
| F1 | shuffled_vs_seqss3d | -0.0611 | -0.0965 to -0.0238 | 0.001999 | 0.007996 | 2000 |
| Precision | shuffled_vs_seqss3d | +0.0342 | -0.0009 to +0.0683 | 0.06197 | 0.3098 | 2000 |
| Recall | shuffled_vs_seqss3d | -0.1211 | -0.1657 to -0.0811 | 0.0009995 | 0.005997 | 2000 |
| GMean | shuffled_vs_seqss3d | -0.0511 | -0.0838 to -0.0189 | 0.002999 | 0.01199 | 2000 |
| MCC | shuffled_vs_seqss | +0.0053 | -0.0228 to +0.0327 | 0.7496 | 1 | 2000 |
| AUC | shuffled_vs_seqss | +0.0063 | -0.0092 to +0.0204 | 0.4308 | 1 | 2000 |
| AUPRC | shuffled_vs_seqss | +0.0022 | -0.0176 to +0.0177 | 0.9505 | 1 | 2000 |
| F1 | shuffled_vs_seqss | -0.0164 | -0.0362 to +0.0032 | 0.1219 | 0.3658 | 2000 |
| Precision | shuffled_vs_seqss | +0.0169 | -0.0068 to +0.0385 | 0.1519 | 0.6077 | 2000 |
| Recall | shuffled_vs_seqss | -0.0269 | -0.0476 to -0.0072 | 0.007996 | 0.02399 | 2000 |
| GMean | shuffled_vs_seqss | -0.0154 | -0.0323 to +0.0011 | 0.07496 | 0.2249 | 2000 |
| MCC | shuffled_vs_cobra | -0.0036 | -0.0747 to +0.0651 | 0.9025 | 1 | 2000 |
| AUC | shuffled_vs_cobra | -0.0170 | -0.0610 to +0.0299 | 0.4708 | 1 | 2000 |
| AUPRC | shuffled_vs_cobra | -0.0055 | -0.0600 to +0.0405 | 0.7716 | 1 | 2000 |
| F1 | shuffled_vs_cobra | -0.0237 | -0.0794 to +0.0391 | 0.4778 | 0.8816 | 2000 |
| Precision | shuffled_vs_cobra | +0.0150 | -0.0381 to +0.0629 | 0.6257 | 1 | 2000 |
| Recall | shuffled_vs_cobra | -0.0593 | -0.1309 to +0.0115 | 0.1089 | 0.2179 | 2000 |
| GMean | shuffled_vs_cobra | -0.0189 | -0.0676 to +0.0357 | 0.5197 | 0.9235 | 2000 |
| MCC | real3d_vs_seqss | +0.0418 | +0.0052 to +0.0740 | 0.02499 | 0.1499 | 2000 |
| AUC | real3d_vs_seqss | +0.0198 | -0.0001 to +0.0378 | 0.05297 | 0.2649 | 2000 |
| AUPRC | real3d_vs_seqss | +0.0136 | -0.0126 to +0.0367 | 0.3198 | 1 | 2000 |
| F1 | real3d_vs_seqss | +0.0536 | +0.0283 to +0.0766 | 0.0009995 | 0.005997 | 2000 |
| Precision | real3d_vs_seqss | +0.0151 | -0.0147 to +0.0399 | 0.3098 | 0.9295 | 2000 |
| Recall | real3d_vs_seqss | +0.0828 | +0.0569 to +0.1062 | 0.0009995 | 0.005997 | 2000 |
| GMean | real3d_vs_seqss | +0.0433 | +0.0227 to +0.0614 | 0.0009995 | 0.005997 | 2000 |
| MCC | real3d_vs_seqss3d | +0.0320 | -0.0057 to +0.0640 | 0.09295 | 0.3718 | 2000 |
| AUC | real3d_vs_seqss3d | +0.0237 | +0.0018 to +0.0438 | 0.03698 | 0.2219 | 2000 |
| AUPRC | real3d_vs_seqss3d | +0.0218 | +0.0005 to +0.0377 | 0.04698 | 0.2819 | 2000 |
| F1 | real3d_vs_seqss3d | +0.0089 | -0.0155 to +0.0305 | 0.4408 | 0.8816 | 2000 |
| Precision | real3d_vs_seqss3d | +0.0323 | +0.0037 to +0.0549 | 0.02899 | 0.1739 | 2000 |
| Recall | real3d_vs_seqss3d | -0.0114 | -0.0370 to +0.0126 | 0.3478 | 0.3478 | 2000 |
| GMean | real3d_vs_seqss3d | +0.0077 | -0.0137 to +0.0261 | 0.4618 | 0.9235 | 2000 |
