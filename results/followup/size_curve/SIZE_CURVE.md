# Fixed-composition training-size curve

All rows use the same canonical 44-chain test set. Sizes 80 and 160 are nested whole-cluster subsets of the 290-chain pool; 290 reuses the accepted large-data results.

## Method summaries

| Size | Method | Metric | Mean | SD |
|--:|:--|:--|--:|--:|
| 80 | cobra | MCC | 0.0983 | 0.0190 |
| 80 | cobra | AUC | 0.5869 | 0.0167 |
| 80 | cobra | AUPRC | 0.3953 | 0.0102 |
| 80 | cobra | F1 | 0.2785 | 0.1118 |
| 80 | cobra | Precision | 0.4328 | 0.0600 |
| 80 | cobra | Recall | 0.2619 | 0.1733 |
| 80 | cobra | GMean | 0.4164 | 0.1063 |
| 80 | duet_graph | MCC | 0.1229 | 0.0233 |
| 80 | duet_graph | AUC | 0.6050 | 0.0132 |
| 80 | duet_graph | AUPRC | 0.3867 | 0.0104 |
| 80 | duet_graph | F1 | 0.3740 | 0.0290 |
| 80 | duet_graph | Precision | 0.4047 | 0.0125 |
| 80 | duet_graph | Recall | 0.3561 | 0.0437 |
| 80 | duet_graph | GMean | 0.5182 | 0.0250 |
| 80 | duet_tcn | MCC | 0.0872 | 0.0309 |
| 80 | duet_tcn | AUC | 0.5814 | 0.0171 |
| 80 | duet_tcn | AUPRC | 0.3866 | 0.0188 |
| 80 | duet_tcn | F1 | 0.3243 | 0.0333 |
| 80 | duet_tcn | Precision | 0.3900 | 0.0349 |
| 80 | duet_tcn | Recall | 0.2896 | 0.0539 |
| 80 | duet_tcn | GMean | 0.4726 | 0.0320 |
| 80 | duet_dense | MCC | 0.0943 | 0.0207 |
| 80 | duet_dense | AUC | 0.5926 | 0.0225 |
| 80 | duet_dense | AUPRC | 0.3943 | 0.0171 |
| 80 | duet_dense | F1 | 0.2941 | 0.0651 |
| 80 | duet_dense | Precision | 0.4127 | 0.0455 |
| 80 | duet_dense | Recall | 0.2534 | 0.0893 |
| 80 | duet_dense | GMean | 0.4405 | 0.0665 |
| 80 | duet_ensemble | MCC | 0.1049 | 0.0133 |
| 80 | duet_ensemble | AUC | 0.6093 | 0.0111 |
| 80 | duet_ensemble | AUPRC | 0.4046 | 0.0125 |
| 80 | duet_ensemble | F1 | 0.3296 | 0.0315 |
| 80 | duet_ensemble | Precision | 0.4051 | 0.0206 |
| 80 | duet_ensemble | Recall | 0.2910 | 0.0545 |
| 80 | duet_ensemble | GMean | 0.4766 | 0.0303 |
| 80 | graph_seq3d | MCC | 0.1055 | 0.0171 |
| 80 | graph_seq3d | AUC | 0.5868 | 0.0068 |
| 80 | graph_seq3d | AUPRC | 0.4083 | 0.0062 |
| 80 | graph_seq3d | F1 | 0.3816 | 0.0180 |
| 80 | graph_seq3d | Precision | 0.4200 | 0.0151 |
| 80 | graph_seq3d | Recall | 0.3520 | 0.0312 |
| 80 | graph_seq3d | GMean | 0.5116 | 0.0148 |
| 80 | graph_seq3d_ensemble | MCC | 0.1109 | 0.0136 |
| 80 | graph_seq3d_ensemble | AUC | 0.5952 | 0.0100 |
| 80 | graph_seq3d_ensemble | AUPRC | 0.4213 | 0.0104 |
| 80 | graph_seq3d_ensemble | F1 | 0.3833 | 0.0279 |
| 80 | graph_seq3d_ensemble | Precision | 0.4239 | 0.0187 |
| 80 | graph_seq3d_ensemble | Recall | 0.3563 | 0.0554 |
| 80 | graph_seq3d_ensemble | GMean | 0.5119 | 0.0230 |
| 80 | grassp_rnafm | MCC | 0.1337 | 0.0384 |
| 80 | grassp_rnafm | AUC | 0.5983 | 0.0266 |
| 80 | grassp_rnafm | AUPRC | 0.4246 | 0.0181 |
| 80 | grassp_rnafm | F1 | 0.4651 | 0.0292 |
| 80 | grassp_rnafm | Precision | 0.4089 | 0.0287 |
| 80 | grassp_rnafm | Recall | 0.5630 | 0.1416 |
| 80 | grassp_rnafm | GMean | 0.5444 | 0.0575 |
| 160 | cobra | MCC | 0.1379 | 0.0203 |
| 160 | cobra | AUC | 0.6187 | 0.0107 |
| 160 | cobra | AUPRC | 0.4122 | 0.0128 |
| 160 | cobra | F1 | 0.3635 | 0.0461 |
| 160 | cobra | Precision | 0.4283 | 0.0195 |
| 160 | cobra | Recall | 0.3457 | 0.0972 |
| 160 | cobra | GMean | 0.5069 | 0.0384 |
| 160 | duet_graph | MCC | 0.1868 | 0.0193 |
| 160 | duet_graph | AUC | 0.6488 | 0.0080 |
| 160 | duet_graph | AUPRC | 0.4515 | 0.0089 |
| 160 | duet_graph | F1 | 0.4434 | 0.0270 |
| 160 | duet_graph | Precision | 0.4371 | 0.0166 |
| 160 | duet_graph | Recall | 0.4591 | 0.0643 |
| 160 | duet_graph | GMean | 0.5748 | 0.0229 |
| 160 | duet_tcn | MCC | 0.1678 | 0.0179 |
| 160 | duet_tcn | AUC | 0.6323 | 0.0112 |
| 160 | duet_tcn | AUPRC | 0.4351 | 0.0117 |
| 160 | duet_tcn | F1 | 0.3878 | 0.0674 |
| 160 | duet_tcn | Precision | 0.4520 | 0.0338 |
| 160 | duet_tcn | Recall | 0.3726 | 0.1237 |
| 160 | duet_tcn | GMean | 0.5223 | 0.0599 |
| 160 | duet_dense | MCC | 0.1466 | 0.0250 |
| 160 | duet_dense | AUC | 0.6164 | 0.0174 |
| 160 | duet_dense | AUPRC | 0.4164 | 0.0118 |
| 160 | duet_dense | F1 | 0.3817 | 0.0706 |
| 160 | duet_dense | Precision | 0.4320 | 0.0342 |
| 160 | duet_dense | Recall | 0.3856 | 0.1237 |
| 160 | duet_dense | GMean | 0.5191 | 0.0630 |
| 160 | duet_ensemble | MCC | 0.1983 | 0.0218 |
| 160 | duet_ensemble | AUC | 0.6542 | 0.0102 |
| 160 | duet_ensemble | AUPRC | 0.4553 | 0.0092 |
| 160 | duet_ensemble | F1 | 0.4993 | 0.0083 |
| 160 | duet_ensemble | Precision | 0.4025 | 0.0234 |
| 160 | duet_ensemble | Recall | 0.6769 | 0.0688 |
| 160 | duet_ensemble | GMean | 0.5953 | 0.0178 |
| 160 | graph_seq3d | MCC | 0.1179 | 0.0373 |
| 160 | graph_seq3d | AUC | 0.6533 | 0.0058 |
| 160 | graph_seq3d | AUPRC | 0.4901 | 0.0066 |
| 160 | graph_seq3d | F1 | 0.1493 | 0.0573 |
| 160 | graph_seq3d | Precision | 0.5770 | 0.0395 |
| 160 | graph_seq3d | Recall | 0.0878 | 0.0379 |
| 160 | graph_seq3d | GMean | 0.2763 | 0.0660 |
| 160 | graph_seq3d_ensemble | MCC | 0.1982 | 0.0230 |
| 160 | graph_seq3d_ensemble | AUC | 0.6541 | 0.0154 |
| 160 | graph_seq3d_ensemble | AUPRC | 0.4902 | 0.0129 |
| 160 | graph_seq3d_ensemble | F1 | 0.4652 | 0.0346 |
| 160 | graph_seq3d_ensemble | Precision | 0.4749 | 0.0399 |
| 160 | graph_seq3d_ensemble | Recall | 0.4765 | 0.0946 |
| 160 | graph_seq3d_ensemble | GMean | 0.5740 | 0.0274 |
| 160 | grassp_rnafm | MCC | 0.2195 | 0.0126 |
| 160 | grassp_rnafm | AUC | 0.6620 | 0.0090 |
| 160 | grassp_rnafm | AUPRC | 0.4997 | 0.0148 |
| 160 | grassp_rnafm | F1 | 0.4977 | 0.0082 |
| 160 | grassp_rnafm | Precision | 0.4682 | 0.0093 |
| 160 | grassp_rnafm | Recall | 0.5324 | 0.0159 |
| 160 | grassp_rnafm | GMean | 0.6048 | 0.0067 |
| 290 | cobra | MCC | 0.1922 | 0.0308 |
| 290 | cobra | AUC | 0.6571 | 0.0226 |
| 290 | cobra | AUPRC | 0.4507 | 0.0242 |
| 290 | cobra | F1 | 0.4252 | 0.0539 |
| 290 | cobra | Precision | 0.4458 | 0.0328 |
| 290 | cobra | Recall | 0.4690 | 0.1075 |
| 290 | cobra | GMean | 0.5574 | 0.0453 |
| 290 | duet_graph | MCC | 0.1928 | 0.0169 |
| 290 | duet_graph | AUC | 0.6410 | 0.0136 |
| 290 | duet_graph | AUPRC | 0.4498 | 0.0141 |
| 290 | duet_graph | F1 | 0.4217 | 0.0285 |
| 290 | duet_graph | Precision | 0.4566 | 0.0113 |
| 290 | duet_graph | Recall | 0.4106 | 0.0502 |
| 290 | duet_graph | GMean | 0.5593 | 0.0256 |
| 290 | duet_tcn | MCC | 0.2078 | 0.0238 |
| 290 | duet_tcn | AUC | 0.6592 | 0.0114 |
| 290 | duet_tcn | AUPRC | 0.4495 | 0.0126 |
| 290 | duet_tcn | F1 | 0.4380 | 0.0445 |
| 290 | duet_tcn | Precision | 0.4574 | 0.0149 |
| 290 | duet_tcn | Recall | 0.4645 | 0.0880 |
| 290 | duet_tcn | GMean | 0.5727 | 0.0369 |
| 290 | duet_dense | MCC | 0.2132 | 0.0119 |
| 290 | duet_dense | AUC | 0.6663 | 0.0068 |
| 290 | duet_dense | AUPRC | 0.4535 | 0.0105 |
| 290 | duet_dense | F1 | 0.4223 | 0.0297 |
| 290 | duet_dense | Precision | 0.4751 | 0.0197 |
| 290 | duet_dense | Recall | 0.4257 | 0.0532 |
| 290 | duet_dense | GMean | 0.5601 | 0.0271 |
| 290 | duet_ensemble | MCC | 0.2396 | 0.0125 |
| 290 | duet_ensemble | AUC | 0.6840 | 0.0045 |
| 290 | duet_ensemble | AUPRC | 0.4855 | 0.0107 |
| 290 | duet_ensemble | F1 | 0.4755 | 0.0223 |
| 290 | duet_ensemble | Precision | 0.4668 | 0.0176 |
| 290 | duet_ensemble | Recall | 0.5277 | 0.0589 |
| 290 | duet_ensemble | GMean | 0.6039 | 0.0173 |
| 290 | graph_seq3d | MCC | 0.2123 | 0.0220 |
| 290 | graph_seq3d | AUC | 0.6455 | 0.0115 |
| 290 | graph_seq3d | AUPRC | 0.4719 | 0.0178 |
| 290 | graph_seq3d | F1 | 0.4624 | 0.0339 |
| 290 | graph_seq3d | Precision | 0.4924 | 0.0279 |
| 290 | graph_seq3d | Recall | 0.4443 | 0.0737 |
| 290 | graph_seq3d | GMean | 0.5770 | 0.0275 |
| 290 | graph_seq3d_ensemble | MCC | 0.2392 | 0.0198 |
| 290 | graph_seq3d_ensemble | AUC | 0.6782 | 0.0088 |
| 290 | graph_seq3d_ensemble | AUPRC | 0.5008 | 0.0150 |
| 290 | graph_seq3d_ensemble | F1 | 0.5001 | 0.0312 |
| 290 | graph_seq3d_ensemble | Precision | 0.4933 | 0.0162 |
| 290 | graph_seq3d_ensemble | Recall | 0.5177 | 0.0743 |
| 290 | graph_seq3d_ensemble | GMean | 0.6074 | 0.0234 |
| 290 | grassp_rnafm | MCC | 0.2607 | 0.0135 |
| 290 | grassp_rnafm | AUC | 0.6860 | 0.0076 |
| 290 | grassp_rnafm | AUPRC | 0.5093 | 0.0114 |
| 290 | grassp_rnafm | F1 | 0.5480 | 0.0070 |
| 290 | grassp_rnafm | Precision | 0.4675 | 0.0108 |
| 290 | grassp_rnafm | Recall | 0.6635 | 0.0220 |
| 290 | grassp_rnafm | GMean | 0.6359 | 0.0070 |

## Comparisons against 80 chains

| Metric | Size | Method | Delta | 95% CI | raw p | Holm p |
|:--|--:|:--|--:|:--|--:|--:|
| MCC | 160 | cobra | +0.0396 | +0.0243 to +0.0548 | 0.0002345 | 0.0009382 |
| MCC | 160 | duet_graph | +0.0639 | +0.0496 to +0.0782 | 3.2e-06 | 3.2e-05 |
| MCC | 160 | duet_tcn | +0.0805 | +0.0489 to +0.1121 | 0.0002699 | 0.0009382 |
| MCC | 160 | duet_dense | +0.0523 | +0.0294 to +0.0752 | 0.0005893 | 0.001179 |
| MCC | 160 | duet_ensemble | +0.0934 | +0.0744 to +0.1124 | 1.468e-06 | 1.615e-05 |
| MCC | 160 | graph_seq3d | +0.0124 | -0.0135 to +0.0383 | 0.3069 | 0.3069 |
| MCC | 160 | graph_seq3d_ensemble | +0.0873 | +0.0652 to +0.1094 | 9.065e-06 | 6.805e-05 |
| MCC | 160 | grassp_rnafm | +0.0858 | +0.0567 to +0.1149 | 9.21e-05 | 0.0004605 |
| MCC | 290 | cobra | +0.0938 | +0.0650 to +0.1226 | 4.262e-05 | 0.0002557 |
| MCC | 290 | duet_graph | +0.0700 | +0.0524 to +0.0875 | 8.506e-06 | 6.805e-05 |
| MCC | 290 | duet_tcn | +0.1206 | +0.0930 to +0.1482 | 3.97e-06 | 3.573e-05 |
| MCC | 290 | duet_dense | +0.1189 | +0.1034 to +0.1345 | 3.22e-08 | 4.83e-07 |
| MCC | 290 | duet_ensemble | +0.1347 | +0.1191 to +0.1502 | 1.074e-08 | 1.718e-07 |
| MCC | 290 | graph_seq3d | +0.1068 | +0.0899 to +0.1237 | 1.692e-07 | 2.2e-06 |
| MCC | 290 | graph_seq3d_ensemble | +0.1283 | +0.1105 to +0.1461 | 5.449e-08 | 7.628e-07 |
| MCC | 290 | grassp_rnafm | +0.1269 | +0.1025 to +0.1514 | 9.263e-07 | 1.112e-05 |
| AUC | 160 | cobra | +0.0318 | +0.0178 to +0.0459 | 0.0006185 | 0.001237 |
| AUC | 160 | duet_graph | +0.0437 | +0.0332 to +0.0543 | 6.124e-06 | 6.124e-05 |
| AUC | 160 | duet_tcn | +0.0509 | +0.0367 to +0.0651 | 1.986e-05 | 0.0001396 |
| AUC | 160 | duet_dense | +0.0238 | +0.0010 to +0.0466 | 0.04259 | 0.04259 |
| AUC | 160 | duet_ensemble | +0.0449 | +0.0323 to +0.0575 | 2.049e-05 | 0.0001396 |
| AUC | 160 | graph_seq3d | +0.0665 | +0.0602 to +0.0728 | 1.941e-09 | 3.105e-08 |
| AUC | 160 | graph_seq3d_ensemble | +0.0589 | +0.0427 to +0.0751 | 1.745e-05 | 0.0001396 |
| AUC | 160 | grassp_rnafm | +0.0637 | +0.0429 to +0.0845 | 6.801e-05 | 0.000272 |
| AUC | 290 | cobra | +0.0703 | +0.0471 to +0.0934 | 7.314e-05 | 0.000272 |
| AUC | 290 | duet_graph | +0.0359 | +0.0254 to +0.0464 | 2.94e-05 | 0.000147 |
| AUC | 290 | duet_tcn | +0.0778 | +0.0635 to +0.0922 | 6.287e-07 | 7.544e-06 |
| AUC | 290 | duet_dense | +0.0737 | +0.0554 to +0.0920 | 7.839e-06 | 7.055e-05 |
| AUC | 290 | duet_ensemble | +0.0748 | +0.0655 to +0.0840 | 2.064e-08 | 3.096e-07 |
| AUC | 290 | graph_seq3d | +0.0587 | +0.0494 to +0.0680 | 1.796e-07 | 2.335e-06 |
| AUC | 290 | graph_seq3d_ensemble | +0.0830 | +0.0724 to +0.0936 | 2.667e-08 | 3.734e-07 |
| AUC | 290 | grassp_rnafm | +0.0877 | +0.0686 to +0.1067 | 2.563e-06 | 2.819e-05 |
| AUPRC | 160 | cobra | +0.0169 | +0.0033 to +0.0305 | 0.02043 | 0.04086 |
| AUPRC | 160 | duet_graph | +0.0648 | +0.0556 to +0.0739 | 6.3e-08 | 8.82e-07 |
| AUPRC | 160 | duet_tcn | +0.0485 | +0.0325 to +0.0644 | 7.21e-05 | 0.0002884 |
| AUPRC | 160 | duet_dense | +0.0221 | +0.0042 to +0.0400 | 0.02089 | 0.04086 |
| AUPRC | 160 | duet_ensemble | +0.0508 | +0.0369 to +0.0646 | 1.644e-05 | 8.218e-05 |
| AUPRC | 160 | graph_seq3d | +0.0818 | +0.0758 to +0.0878 | 1.877e-10 | 3.002e-09 |
| AUPRC | 160 | graph_seq3d_ensemble | +0.0690 | +0.0555 to +0.0824 | 1.059e-06 | 1.031e-05 |
| AUPRC | 160 | grassp_rnafm | +0.0751 | +0.0580 to +0.0922 | 3.811e-06 | 2.975e-05 |
| AUPRC | 290 | cobra | +0.0554 | +0.0350 to +0.0758 | 0.0001719 | 0.0005158 |
| AUPRC | 290 | duet_graph | +0.0631 | +0.0513 to +0.0750 | 7.485e-07 | 8.234e-06 |
| AUPRC | 290 | duet_tcn | +0.0629 | +0.0462 to +0.0796 | 1.333e-05 | 7.997e-05 |
| AUPRC | 290 | duet_dense | +0.0592 | +0.0458 to +0.0727 | 3.719e-06 | 2.975e-05 |
| AUPRC | 290 | duet_ensemble | +0.0809 | +0.0698 to +0.0920 | 5.029e-08 | 7.543e-07 |
| AUPRC | 290 | graph_seq3d | +0.0636 | +0.0512 to +0.0760 | 1.031e-06 | 1.031e-05 |
| AUPRC | 290 | graph_seq3d_ensemble | +0.0795 | +0.0659 to +0.0930 | 3.229e-07 | 4.197e-06 |
| AUPRC | 290 | grassp_rnafm | +0.0848 | +0.0696 to +0.0999 | 4.798e-07 | 5.757e-06 |
| F1 | 160 | cobra | +0.0850 | +0.0034 to +0.1666 | 0.04285 | 0.04285 |
| F1 | 160 | duet_graph | +0.0694 | +0.0478 to +0.0910 | 4.726e-05 | 0.0004726 |
| F1 | 160 | duet_tcn | +0.0636 | +0.0182 to +0.1089 | 0.01132 | 0.03781 |
| F1 | 160 | duet_dense | +0.0875 | +0.0382 to +0.1369 | 0.003062 | 0.01531 |
| F1 | 160 | duet_ensemble | +0.1696 | +0.1488 to +0.1905 | 1.885e-08 | 3.016e-07 |
| F1 | 160 | graph_seq3d | -0.2323 | -0.2746 to -0.1900 | 5.681e-07 | 8.521e-06 |
| F1 | 160 | graph_seq3d_ensemble | +0.0819 | +0.0479 to +0.1159 | 0.0004058 | 0.003247 |
| F1 | 160 | grassp_rnafm | +0.0325 | +0.0078 to +0.0573 | 0.01553 | 0.03781 |
| F1 | 290 | cobra | +0.1467 | +0.0457 to +0.2478 | 0.009453 | 0.03781 |
| F1 | 290 | duet_graph | +0.0477 | +0.0242 to +0.0711 | 0.001284 | 0.007769 |
| F1 | 290 | duet_tcn | +0.1137 | +0.0822 to +0.1452 | 1.878e-05 | 0.0002066 |
| F1 | 290 | duet_dense | +0.1281 | +0.0666 to +0.1897 | 0.00111 | 0.007769 |
| F1 | 290 | duet_ensemble | +0.1459 | +0.1186 to +0.1731 | 7.153e-07 | 1.001e-05 |
| F1 | 290 | graph_seq3d | +0.0808 | +0.0520 to +0.1097 | 0.0001343 | 0.001208 |
| F1 | 290 | graph_seq3d_ensemble | +0.1168 | +0.0935 to +0.1401 | 1.245e-06 | 1.619e-05 |
| F1 | 290 | grassp_rnafm | +0.0828 | +0.0623 to +0.1034 | 7.578e-06 | 9.093e-05 |
| Precision | 160 | cobra | -0.0045 | -0.0435 to +0.0346 | 0.8025 | 1 |
| Precision | 160 | duet_graph | +0.0324 | +0.0180 to +0.0468 | 0.0006496 | 0.005479 |
| Precision | 160 | duet_tcn | +0.0619 | +0.0312 to +0.0926 | 0.001357 | 0.009498 |
| Precision | 160 | duet_dense | +0.0193 | -0.0200 to +0.0585 | 0.2963 | 1 |
| Precision | 160 | duet_ensemble | -0.0027 | -0.0257 to +0.0204 | 0.7995 | 1 |
| Precision | 160 | graph_seq3d | +0.1571 | +0.1271 to +0.1870 | 8.52e-07 | 1.363e-05 |
| Precision | 160 | graph_seq3d_ensemble | +0.0510 | +0.0185 to +0.0835 | 0.006251 | 0.03302 |
| Precision | 160 | grassp_rnafm | +0.0593 | +0.0409 to +0.0776 | 4.491e-05 | 0.000539 |
| Precision | 290 | cobra | +0.0130 | -0.0363 to +0.0624 | 0.5657 | 1 |
| Precision | 290 | duet_graph | +0.0519 | +0.0386 to +0.0651 | 9.667e-06 | 0.0001353 |
| Precision | 290 | duet_tcn | +0.0674 | +0.0378 to +0.0970 | 0.0006088 | 0.005479 |
| Precision | 290 | duet_dense | +0.0623 | +0.0235 to +0.1012 | 0.005504 | 0.03302 |
| Precision | 290 | duet_ensemble | +0.0617 | +0.0400 to +0.0834 | 0.0001205 | 0.001205 |
| Precision | 290 | graph_seq3d | +0.0725 | +0.0482 to +0.0967 | 8.337e-05 | 0.0009171 |
| Precision | 290 | graph_seq3d_ensemble | +0.0694 | +0.0521 to +0.0867 | 7.944e-06 | 0.0001192 |
| Precision | 290 | grassp_rnafm | +0.0585 | +0.0417 to +0.0754 | 2.509e-05 | 0.0003262 |
| Recall | 160 | cobra | +0.0838 | -0.0364 to +0.2040 | 0.1492 | 0.2983 |
| Recall | 160 | duet_graph | +0.1030 | +0.0545 to +0.1514 | 0.0009628 | 0.01059 |
| Recall | 160 | duet_tcn | +0.0830 | +0.0004 to +0.1655 | 0.04907 | 0.1737 |
| Recall | 160 | duet_dense | +0.1322 | +0.0409 to +0.2234 | 0.009569 | 0.06664 |
| Recall | 160 | duet_ensemble | +0.3859 | +0.3278 to +0.4440 | 1.117e-07 | 1.675e-06 |
| Recall | 160 | graph_seq3d | -0.2642 | -0.3012 to -0.2272 | 5.941e-08 | 9.505e-07 |
| Recall | 160 | graph_seq3d_ensemble | +0.1201 | +0.0395 to +0.2007 | 0.008235 | 0.06664 |
| Recall | 160 | grassp_rnafm | -0.0306 | -0.1311 to +0.0699 | 0.5081 | 0.5081 |
| Recall | 290 | cobra | +0.2071 | +0.0399 to +0.3744 | 0.02064 | 0.1032 |
| Recall | 290 | duet_graph | +0.0545 | +0.0186 to +0.0903 | 0.007404 | 0.06664 |
| Recall | 290 | duet_tcn | +0.1749 | +0.1151 to +0.2347 | 9.716e-05 | 0.001263 |
| Recall | 290 | duet_dense | +0.1723 | +0.0773 to +0.2672 | 0.002657 | 0.02657 |
| Recall | 290 | duet_ensemble | +0.2367 | +0.1746 to +0.2989 | 1.221e-05 | 0.000171 |
| Recall | 290 | graph_seq3d | +0.0923 | +0.0297 to +0.1549 | 0.008737 | 0.06664 |
| Recall | 290 | graph_seq3d_ensemble | +0.1613 | +0.1016 to +0.2211 | 0.000178 | 0.002136 |
| Recall | 290 | grassp_rnafm | +0.1005 | +0.0037 to +0.1972 | 0.04342 | 0.1737 |
| GMean | 160 | cobra | +0.0904 | +0.0124 to +0.1685 | 0.02775 | 0.03858 |
| GMean | 160 | duet_graph | +0.0565 | +0.0384 to +0.0747 | 6.041e-05 | 0.0006645 |
| GMean | 160 | duet_tcn | +0.0496 | +0.0101 to +0.0891 | 0.01929 | 0.03858 |
| GMean | 160 | duet_dense | +0.0786 | +0.0322 to +0.1251 | 0.004027 | 0.02014 |
| GMean | 160 | duet_ensemble | +0.1187 | +0.0963 to +0.1411 | 7.847e-07 | 1.099e-05 |
| GMean | 160 | graph_seq3d | -0.2353 | -0.2831 to -0.1875 | 1.459e-06 | 1.897e-05 |
| GMean | 160 | graph_seq3d_ensemble | +0.0621 | +0.0356 to +0.0887 | 0.0004973 | 0.004129 |
| GMean | 160 | grassp_rnafm | +0.0604 | +0.0185 to +0.1023 | 0.009862 | 0.02959 |
| GMean | 290 | cobra | +0.1410 | +0.0483 to +0.2337 | 0.007392 | 0.02957 |
| GMean | 290 | duet_graph | +0.0411 | +0.0208 to +0.0614 | 0.001312 | 0.009186 |
| GMean | 290 | duet_tcn | +0.1000 | +0.0717 to +0.1284 | 2.248e-05 | 0.0002698 |
| GMean | 290 | duet_dense | +0.1197 | +0.0590 to +0.1804 | 0.00158 | 0.009477 |
| GMean | 290 | duet_ensemble | +0.1273 | +0.1035 to +0.1511 | 7.206e-07 | 1.081e-05 |
| GMean | 290 | graph_seq3d | +0.0654 | +0.0418 to +0.0890 | 0.0001462 | 0.001462 |
| GMean | 290 | graph_seq3d_ensemble | +0.0955 | +0.0790 to +0.1119 | 3.536e-07 | 5.657e-06 |
| GMean | 290 | grassp_rnafm | +0.0915 | +0.0528 to +0.1301 | 0.0004588 | 0.004129 |

## Chain bootstrap

| Metric | Comparison | Delta | 95% CI | raw p | Holm p |
|:--|:--|--:|:--|--:|--:|
| MCC | 160/cobra | +0.0396 | -0.0137 to +0.0939 | 0.1538 | 0.3077 |
| AUC | 160/cobra | +0.0318 | -0.0050 to +0.0690 | 0.0999 | 0.2278 |
| AUPRC | 160/cobra | +0.0169 | -0.0176 to +0.0525 | 0.3137 | 0.3317 |
| F1 | 160/cobra | +0.0850 | +0.0291 to +0.1435 | 0.001998 | 0.03197 |
| Precision | 160/cobra | -0.0045 | -0.0499 to +0.0463 | 0.9491 | 1 |
| Recall | 160/cobra | +0.0838 | +0.0180 to +0.1550 | 0.00999 | 0.03996 |
| GMean | 160/cobra | +0.0904 | +0.0418 to +0.1436 | 0.001998 | 0.03197 |
| MCC | 160/duet_graph | +0.0639 | +0.0148 to +0.1065 | 0.01399 | 0.1079 |
| AUC | 160/duet_graph | +0.0437 | +0.0142 to +0.0682 | 0.005994 | 0.05994 |
| AUPRC | 160/duet_graph | +0.0648 | +0.0299 to +0.0958 | 0.001998 | 0.03197 |
| F1 | 160/duet_graph | +0.0694 | +0.0354 to +0.1028 | 0.001998 | 0.03197 |
| Precision | 160/duet_graph | +0.0324 | +0.0012 to +0.0613 | 0.04795 | 0.2398 |
| Recall | 160/duet_graph | +0.1030 | +0.0544 to +0.1429 | 0.001998 | 0.03197 |
| GMean | 160/duet_graph | +0.0565 | +0.0261 to +0.0854 | 0.001998 | 0.03197 |
| MCC | 160/duet_tcn | +0.0805 | +0.0079 to +0.1504 | 0.02997 | 0.1399 |
| AUC | 160/duet_tcn | +0.0509 | +0.0044 to +0.0939 | 0.03197 | 0.1279 |
| AUPRC | 160/duet_tcn | +0.0485 | +0.0007 to +0.0922 | 0.04995 | 0.1499 |
| F1 | 160/duet_tcn | +0.0636 | -0.0066 to +0.1328 | 0.09391 | 0.2577 |
| Precision | 160/duet_tcn | +0.0619 | +0.0092 to +0.1136 | 0.01598 | 0.1279 |
| Recall | 160/duet_tcn | +0.0830 | -0.0268 to +0.1883 | 0.1499 | 0.4496 |
| GMean | 160/duet_tcn | +0.0496 | -0.0173 to +0.1102 | 0.1818 | 0.2238 |
| MCC | 160/duet_dense | +0.0523 | -0.0038 to +0.1077 | 0.06593 | 0.1978 |
| AUC | 160/duet_dense | +0.0238 | -0.0136 to +0.0610 | 0.1638 | 0.2278 |
| AUPRC | 160/duet_dense | +0.0221 | -0.0130 to +0.0567 | 0.1658 | 0.3317 |
| F1 | 160/duet_dense | +0.0875 | +0.0330 to +0.1456 | 0.003996 | 0.03197 |
| Precision | 160/duet_dense | +0.0193 | -0.0233 to +0.0695 | 0.3357 | 1 |
| Recall | 160/duet_dense | +0.1322 | +0.0690 to +0.2012 | 0.001998 | 0.03197 |
| GMean | 160/duet_dense | +0.0786 | +0.0266 to +0.1336 | 0.003996 | 0.03197 |
| MCC | 160/duet_ensemble | +0.0934 | +0.0414 to +0.1496 | 0.001998 | 0.03197 |
| AUC | 160/duet_ensemble | +0.0449 | +0.0091 to +0.0786 | 0.01998 | 0.0999 |
| AUPRC | 160/duet_ensemble | +0.0508 | +0.0082 to +0.0872 | 0.01798 | 0.08991 |
| F1 | 160/duet_ensemble | +0.1696 | +0.1173 to +0.2311 | 0.001998 | 0.03197 |
| Precision | 160/duet_ensemble | -0.0027 | -0.0396 to +0.0402 | 0.979 | 1 |
| Recall | 160/duet_ensemble | +0.3859 | +0.3166 to +0.4521 | 0.001998 | 0.03197 |
| GMean | 160/duet_ensemble | +0.1187 | +0.0631 to +0.1754 | 0.001998 | 0.03197 |
| MCC | 160/graph_seq3d | +0.0124 | -0.0523 to +0.0896 | 0.7153 | 0.7153 |
| AUC | 160/graph_seq3d | +0.0665 | +0.0205 to +0.1197 | 0.00999 | 0.06993 |
| AUPRC | 160/graph_seq3d | +0.0818 | +0.0288 to +0.1353 | 0.003996 | 0.05195 |
| F1 | 160/graph_seq3d | -0.2323 | -0.2904 to -0.1648 | 0.001998 | 0.03197 |
| Precision | 160/graph_seq3d | +0.1571 | +0.0852 to +0.2614 | 0.001998 | 0.03197 |
| Recall | 160/graph_seq3d | -0.2642 | -0.3135 to -0.2052 | 0.001998 | 0.03197 |
| GMean | 160/graph_seq3d | -0.2353 | -0.2895 to -0.1814 | 0.001998 | 0.03197 |
| MCC | 160/graph_seq3d_ensemble | +0.0873 | +0.0187 to +0.1715 | 0.01199 | 0.1079 |
| AUC | 160/graph_seq3d_ensemble | +0.0589 | +0.0133 to +0.1140 | 0.00999 | 0.06993 |
| AUPRC | 160/graph_seq3d_ensemble | +0.0690 | +0.0192 to +0.1236 | 0.007992 | 0.07193 |
| F1 | 160/graph_seq3d_ensemble | +0.0819 | +0.0223 to +0.1463 | 0.005994 | 0.03197 |
| Precision | 160/graph_seq3d_ensemble | +0.0510 | +0.0074 to +0.1059 | 0.02398 | 0.1678 |
| Recall | 160/graph_seq3d_ensemble | +0.1201 | +0.0486 to +0.2017 | 0.001998 | 0.03197 |
| GMean | 160/graph_seq3d_ensemble | +0.0621 | +0.0164 to +0.1141 | 0.01199 | 0.03596 |
| MCC | 160/grassp_rnafm | +0.0858 | +0.0193 to +0.1631 | 0.01598 | 0.1079 |
| AUC | 160/grassp_rnafm | +0.0637 | +0.0212 to +0.1173 | 0.003996 | 0.04396 |
| AUPRC | 160/grassp_rnafm | +0.0751 | +0.0279 to +0.1315 | 0.003996 | 0.05195 |
| F1 | 160/grassp_rnafm | +0.0325 | -0.0125 to +0.0806 | 0.1518 | 0.2577 |
| Precision | 160/grassp_rnafm | +0.0593 | +0.0197 to +0.1027 | 0.003996 | 0.05594 |
| Recall | 160/grassp_rnafm | -0.0306 | -0.0877 to +0.0311 | 0.3177 | 0.5834 |
| GMean | 160/grassp_rnafm | +0.0604 | +0.0214 to +0.1065 | 0.003996 | 0.03197 |
| MCC | 290/cobra | +0.0938 | +0.0175 to +0.1708 | 0.01199 | 0.1079 |
| AUC | 290/cobra | +0.0703 | +0.0234 to +0.1113 | 0.005994 | 0.05994 |
| AUPRC | 290/cobra | +0.0554 | +0.0086 to +0.1055 | 0.02597 | 0.1039 |
| F1 | 290/cobra | +0.1467 | +0.0742 to +0.2170 | 0.001998 | 0.03197 |
| Precision | 290/cobra | +0.0130 | -0.0465 to +0.0797 | 0.5734 | 1 |
| Recall | 290/cobra | +0.2071 | +0.1068 to +0.2862 | 0.001998 | 0.03197 |
| GMean | 290/cobra | +0.1410 | +0.0716 to +0.2121 | 0.001998 | 0.03197 |
| MCC | 290/duet_graph | +0.0700 | +0.0060 to +0.1295 | 0.02797 | 0.1399 |
| AUC | 290/duet_graph | +0.0359 | -0.0029 to +0.0707 | 0.07592 | 0.2278 |
| AUPRC | 290/duet_graph | +0.0631 | +0.0165 to +0.1033 | 0.003996 | 0.05195 |
| F1 | 290/duet_graph | +0.0477 | -0.0032 to +0.1000 | 0.08591 | 0.2577 |
| Precision | 290/duet_graph | +0.0519 | +0.0139 to +0.0913 | 0.005994 | 0.06593 |
| Recall | 290/duet_graph | +0.0545 | -0.0402 to +0.1306 | 0.2917 | 0.5834 |
| GMean | 290/duet_graph | +0.0411 | -0.0064 to +0.0872 | 0.1119 | 0.2238 |
| MCC | 290/duet_tcn | +0.1206 | +0.0517 to +0.1896 | 0.001998 | 0.03197 |
| AUC | 290/duet_tcn | +0.0778 | +0.0316 to +0.1162 | 0.001998 | 0.03197 |
| AUPRC | 290/duet_tcn | +0.0629 | +0.0202 to +0.1131 | 0.00999 | 0.07992 |
| F1 | 290/duet_tcn | +0.1137 | +0.0473 to +0.1771 | 0.001998 | 0.03197 |
| Precision | 290/duet_tcn | +0.0674 | +0.0196 to +0.1237 | 0.00999 | 0.0999 |
| Recall | 290/duet_tcn | +0.1749 | +0.0620 to +0.2666 | 0.003996 | 0.03197 |
| GMean | 290/duet_tcn | +0.1000 | +0.0396 to +0.1619 | 0.001998 | 0.03197 |
| MCC | 290/duet_dense | +0.1189 | +0.0527 to +0.1874 | 0.001998 | 0.03197 |
| AUC | 290/duet_dense | +0.0737 | +0.0320 to +0.1077 | 0.001998 | 0.03197 |
| AUPRC | 290/duet_dense | +0.0592 | +0.0159 to +0.1059 | 0.00999 | 0.07992 |
| F1 | 290/duet_dense | +0.1281 | +0.0588 to +0.1954 | 0.001998 | 0.03197 |
| Precision | 290/duet_dense | +0.0623 | +0.0055 to +0.1261 | 0.03996 | 0.2398 |
| Recall | 290/duet_dense | +0.1723 | +0.0676 to +0.2569 | 0.001998 | 0.03197 |
| GMean | 290/duet_dense | +0.1197 | +0.0567 to +0.1895 | 0.001998 | 0.03197 |
| MCC | 290/duet_ensemble | +0.1347 | +0.0653 to +0.2060 | 0.001998 | 0.03197 |
| AUC | 290/duet_ensemble | +0.0748 | +0.0331 to +0.1129 | 0.001998 | 0.03197 |
| AUPRC | 290/duet_ensemble | +0.0809 | +0.0285 to +0.1281 | 0.001998 | 0.03197 |
| F1 | 290/duet_ensemble | +0.1459 | +0.0829 to +0.2114 | 0.001998 | 0.03197 |
| Precision | 290/duet_ensemble | +0.0617 | +0.0142 to +0.1181 | 0.01199 | 0.1079 |
| Recall | 290/duet_ensemble | +0.2367 | +0.1238 to +0.3289 | 0.001998 | 0.03197 |
| GMean | 290/duet_ensemble | +0.1273 | +0.0678 to +0.1925 | 0.001998 | 0.03197 |
| MCC | 290/graph_seq3d | +0.1068 | +0.0452 to +0.1869 | 0.001998 | 0.03197 |
| AUC | 290/graph_seq3d | +0.0587 | +0.0154 to +0.1141 | 0.007992 | 0.06394 |
| AUPRC | 290/graph_seq3d | +0.0636 | +0.0172 to +0.1191 | 0.01199 | 0.07992 |
| F1 | 290/graph_seq3d | +0.0808 | +0.0314 to +0.1377 | 0.001998 | 0.03197 |
| Precision | 290/graph_seq3d | +0.0725 | +0.0278 to +0.1291 | 0.003996 | 0.05594 |
| Recall | 290/graph_seq3d | +0.0923 | +0.0358 to +0.1541 | 0.001998 | 0.03197 |
| GMean | 290/graph_seq3d | +0.0654 | +0.0253 to +0.1130 | 0.001998 | 0.03197 |
| MCC | 290/graph_seq3d_ensemble | +0.1283 | +0.0531 to +0.2184 | 0.001998 | 0.03197 |
| AUC | 290/graph_seq3d_ensemble | +0.0830 | +0.0356 to +0.1379 | 0.001998 | 0.03197 |
| AUPRC | 290/graph_seq3d_ensemble | +0.0795 | +0.0285 to +0.1493 | 0.005994 | 0.05994 |
| F1 | 290/graph_seq3d_ensemble | +0.1168 | +0.0622 to +0.1790 | 0.001998 | 0.03197 |
| Precision | 290/graph_seq3d_ensemble | +0.0694 | +0.0220 to +0.1326 | 0.001998 | 0.03197 |
| Recall | 290/graph_seq3d_ensemble | +0.1613 | +0.0891 to +0.2481 | 0.001998 | 0.03197 |
| GMean | 290/graph_seq3d_ensemble | +0.0955 | +0.0481 to +0.1519 | 0.001998 | 0.03197 |
| MCC | 290/grassp_rnafm | +0.1269 | +0.0551 to +0.2203 | 0.001998 | 0.03197 |
| AUC | 290/grassp_rnafm | +0.0877 | +0.0450 to +0.1467 | 0.001998 | 0.03197 |
| AUPRC | 290/grassp_rnafm | +0.0848 | +0.0352 to +0.1542 | 0.001998 | 0.03197 |
| F1 | 290/grassp_rnafm | +0.0828 | +0.0397 to +0.1363 | 0.001998 | 0.03197 |
| Precision | 290/grassp_rnafm | +0.0585 | +0.0212 to +0.1065 | 0.003996 | 0.05594 |
| Recall | 290/grassp_rnafm | +0.1005 | +0.0450 to +0.1722 | 0.001998 | 0.03197 |
| GMean | 290/grassp_rnafm | +0.0915 | +0.0517 to +0.1446 | 0.001998 | 0.03197 |

Figure: `SIZE_CURVE.pdf` / `.svg` / `.png` / `.tiff`. Curves show mean across ten seeds; bands show one seed SD.
