# Follow-up experiment results

All comparisons use paired training seeds. Chain bootstrap keeps seeds paired and resamples benchmark chains. Raw and Holm-adjusted p values are reported for all seven metrics in the machine-readable files.

## All-metric significance

The existing comparison families were preserved and expanded from MCC to MCC, AUC, AUPRC, F1, Precision, Recall and GMean.

| Family | Metric | Comparison | Delta | Holm p |
|:--|:--|:--|--:|--:|
| crossed_controls_3split | MCC | grassp_ernie_vs_grassp_rnafm | -0.0062 | 0.3562 |
| crossed_controls_3split | MCC | grassp_ernie_vs_rgcn_ernie | +0.0935 | 0.0001039 |
| crossed_controls_3split | MCC | grassp_rnafm_vs_graph_seq3d | +0.0484 | 0.0009543 |
| crossed_controls_3split | MCC | rgcn_ernie_vs_graph_seq3d | -0.0513 | 0.007463 |
| crossed_controls_3split | AUC | grassp_ernie_vs_grassp_rnafm | -0.0150 | 0.0002393 |
| crossed_controls_3split | AUC | grassp_ernie_vs_rgcn_ernie | +0.0698 | 2.412e-05 |
| crossed_controls_3split | AUC | grassp_rnafm_vs_graph_seq3d | +0.0405 | 1.165e-05 |
| crossed_controls_3split | AUC | rgcn_ernie_vs_graph_seq3d | -0.0443 | 0.0006559 |
| crossed_controls_3split | AUPRC | grassp_ernie_vs_grassp_rnafm | -0.0158 | 0.04939 |
| crossed_controls_3split | AUPRC | grassp_ernie_vs_rgcn_ernie | +0.0430 | 0.003845 |
| crossed_controls_3split | AUPRC | grassp_rnafm_vs_graph_seq3d | +0.0374 | 0.001133 |
| crossed_controls_3split | AUPRC | rgcn_ernie_vs_graph_seq3d | -0.0214 | 0.04939 |
| crossed_controls_3split | F1 | grassp_ernie_vs_grassp_rnafm | -0.0104 | 0.04795 |
| crossed_controls_3split | F1 | grassp_ernie_vs_rgcn_ernie | +0.1149 | 1.138e-05 |
| crossed_controls_3split | F1 | grassp_rnafm_vs_graph_seq3d | +0.0856 | 9.187e-05 |
| crossed_controls_3split | F1 | rgcn_ernie_vs_graph_seq3d | -0.0397 | 0.04795 |
| crossed_controls_3split | Precision | grassp_ernie_vs_grassp_rnafm | +0.0068 | 0.2512 |
| crossed_controls_3split | Precision | grassp_ernie_vs_rgcn_ernie | +0.0164 | 0.2317 |
| crossed_controls_3split | Precision | grassp_rnafm_vs_graph_seq3d | -0.0250 | 0.06266 |
| crossed_controls_3split | Precision | rgcn_ernie_vs_graph_seq3d | -0.0345 | 0.0937 |
| crossed_controls_3split | Recall | grassp_ernie_vs_grassp_rnafm | -0.0405 | 0.02817 |
| crossed_controls_3split | Recall | grassp_ernie_vs_rgcn_ernie | +0.2276 | 8.598e-06 |
| crossed_controls_3split | Recall | grassp_rnafm_vs_graph_seq3d | +0.2192 | 1.44e-05 |
| crossed_controls_3split | Recall | rgcn_ernie_vs_graph_seq3d | -0.0489 | 0.1229 |
| crossed_controls_3split | GMean | grassp_ernie_vs_grassp_rnafm | -0.0038 | 0.3053 |
| crossed_controls_3split | GMean | grassp_ernie_vs_rgcn_ernie | +0.0862 | 2.103e-05 |
| crossed_controls_3split | GMean | grassp_rnafm_vs_graph_seq3d | +0.0589 | 0.0004742 |
| crossed_controls_3split | GMean | rgcn_ernie_vs_graph_seq3d | -0.0311 | 0.06135 |
| external_independent_2split | MCC | duet_ensemble_vs_cobra | +0.1093 | 0.01398 |
| external_independent_2split | MCC | graph_seqss3d_vs_graph_seq3d_external | -0.0124 | 0.9523 |
| external_independent_2split | MCC | grassp_ernie_vs_duet_ensemble | -0.1195 | 0.002999 |
| external_independent_2split | MCC | grassp_rnafm_vs_cobra | +0.1066 | 0.004599 |
| external_independent_2split | MCC | grassp_rnafm_vs_duet_ensemble | -0.0027 | 0.9523 |
| external_independent_2split | AUC | duet_ensemble_vs_cobra | +0.0729 | 0.01009 |
| external_independent_2split | AUC | graph_seqss3d_vs_graph_seq3d_external | -0.0184 | 0.256 |
| external_independent_2split | AUC | grassp_ernie_vs_duet_ensemble | -0.1390 | 5.248e-06 |
| external_independent_2split | AUC | grassp_rnafm_vs_cobra | +0.0830 | 0.003159 |
| external_independent_2split | AUC | grassp_rnafm_vs_duet_ensemble | +0.0101 | 0.256 |
| external_independent_2split | AUPRC | duet_ensemble_vs_cobra | +0.0627 | 0.003279 |
| external_independent_2split | AUPRC | graph_seqss3d_vs_graph_seq3d_external | -0.0026 | 0.8957 |
| external_independent_2split | AUPRC | grassp_ernie_vs_duet_ensemble | -0.1179 | 1.668e-07 |
| external_independent_2split | AUPRC | grassp_rnafm_vs_cobra | +0.1021 | 0.0002498 |
| external_independent_2split | AUPRC | grassp_rnafm_vs_duet_ensemble | +0.0394 | 0.02908 |
| external_independent_2split | F1 | duet_ensemble_vs_cobra | +0.0801 | 0.0111 |
| external_independent_2split | F1 | graph_seqss3d_vs_graph_seq3d_external | -0.0104 | 0.7597 |
| external_independent_2split | F1 | grassp_ernie_vs_duet_ensemble | -0.0739 | 0.001521 |
| external_independent_2split | F1 | grassp_rnafm_vs_cobra | +0.0710 | 0.0111 |
| external_independent_2split | F1 | grassp_rnafm_vs_duet_ensemble | -0.0091 | 0.7597 |
| external_independent_2split | Precision | duet_ensemble_vs_cobra | +0.0522 | 0.006749 |
| external_independent_2split | Precision | graph_seqss3d_vs_graph_seq3d_external | -0.0149 | 0.1737 |
| external_independent_2split | Precision | grassp_ernie_vs_duet_ensemble | -0.0636 | 0.0002698 |
| external_independent_2split | Precision | grassp_rnafm_vs_cobra | +0.0290 | 0.0279 |
| external_independent_2split | Precision | grassp_rnafm_vs_duet_ensemble | -0.0233 | 0.08534 |
| external_independent_2split | Recall | duet_ensemble_vs_cobra | +0.1349 | 0.32 |
| external_independent_2split | Recall | graph_seqss3d_vs_graph_seq3d_external | +0.0229 | 1 |
| external_independent_2split | Recall | grassp_ernie_vs_duet_ensemble | +0.0073 | 1 |
| external_independent_2split | Recall | grassp_rnafm_vs_cobra | +0.2665 | 0.006706 |
| external_independent_2split | Recall | grassp_rnafm_vs_duet_ensemble | +0.1316 | 0.01983 |
| external_independent_2split | GMean | duet_ensemble_vs_cobra | +0.1042 | 0.02138 |
| external_independent_2split | GMean | graph_seqss3d_vs_graph_seq3d_external | +0.0010 | 0.9515 |
| external_independent_2split | GMean | grassp_ernie_vs_duet_ensemble | -0.0688 | 0.01018 |
| external_independent_2split | GMean | grassp_rnafm_vs_cobra | +0.1213 | 0.00845 |
| external_independent_2split | GMean | grassp_rnafm_vs_duet_ensemble | +0.0171 | 0.2399 |
| primary_4split | MCC | duet_dense_vs_cobra | +0.0211 | 0.3107 |
| primary_4split | MCC | duet_ensemble_vs_cobra | +0.0474 | 0.007333 |
| primary_4split | MCC | duet_ensemble_vs_duet_graph | +0.0468 | 0.000473 |
| primary_4split | MCC | duet_graph_vs_cobra | +0.0007 | 0.9491 |
| primary_4split | MCC | duet_tcn_vs_cobra | +0.0156 | 0.5391 |
| primary_4split | AUC | duet_dense_vs_cobra | +0.0091 | 0.5552 |
| primary_4split | AUC | duet_ensemble_vs_cobra | +0.0269 | 0.02805 |
| primary_4split | AUC | duet_ensemble_vs_duet_graph | +0.0431 | 1.235e-05 |
| primary_4split | AUC | duet_graph_vs_cobra | -0.0162 | 0.2298 |
| primary_4split | AUC | duet_tcn_vs_cobra | +0.0021 | 0.8125 |
| primary_4split | AUPRC | duet_dense_vs_cobra | +0.0029 | 1 |
| primary_4split | AUPRC | duet_ensemble_vs_cobra | +0.0348 | 0.01459 |
| primary_4split | AUPRC | duet_ensemble_vs_duet_graph | +0.0356 | 0.0001603 |
| primary_4split | AUPRC | duet_graph_vs_cobra | -0.0008 | 1 |
| primary_4split | AUPRC | duet_tcn_vs_cobra | -0.0012 | 1 |
| primary_4split | F1 | duet_dense_vs_cobra | -0.0030 | 1 |
| primary_4split | F1 | duet_ensemble_vs_cobra | +0.0502 | 0.1443 |
| primary_4split | F1 | duet_ensemble_vs_duet_graph | +0.0538 | 0.005312 |
| primary_4split | F1 | duet_graph_vs_cobra | -0.0036 | 1 |
| primary_4split | F1 | duet_tcn_vs_cobra | +0.0128 | 1 |
| primary_4split | Precision | duet_dense_vs_cobra | +0.0293 | 0.2993 |
| primary_4split | Precision | duet_ensemble_vs_cobra | +0.0211 | 0.5144 |
| primary_4split | Precision | duet_ensemble_vs_duet_graph | +0.0103 | 0.5144 |
| primary_4split | Precision | duet_graph_vs_cobra | +0.0108 | 0.5144 |
| primary_4split | Precision | duet_tcn_vs_cobra | +0.0116 | 0.5144 |
| primary_4split | Recall | duet_dense_vs_cobra | -0.0433 | 0.7637 |
| primary_4split | Recall | duet_ensemble_vs_cobra | +0.0587 | 0.6553 |
| primary_4split | Recall | duet_ensemble_vs_duet_graph | +0.1171 | 0.005595 |
| primary_4split | Recall | duet_graph_vs_cobra | -0.0585 | 0.566 |
| primary_4split | Recall | duet_tcn_vs_cobra | -0.0045 | 0.9278 |
| primary_4split | GMean | duet_dense_vs_cobra | +0.0027 | 1 |
| primary_4split | GMean | duet_ensemble_vs_cobra | +0.0464 | 0.08223 |
| primary_4split | GMean | duet_ensemble_vs_duet_graph | +0.0445 | 0.006374 |
| primary_4split | GMean | duet_graph_vs_cobra | +0.0019 | 1 |
| primary_4split | GMean | duet_tcn_vs_cobra | +0.0152 | 1 |
| primary_4split_fixed_0.5 | MCC | duet_dense_vs_cobra | +0.0322 | 0.09696 |
| primary_4split_fixed_0.5 | MCC | duet_ensemble_vs_cobra | +0.0524 | 0.01023 |
| primary_4split_fixed_0.5 | MCC | duet_ensemble_vs_duet_graph | +0.0434 | 0.0002918 |
| primary_4split_fixed_0.5 | MCC | duet_graph_vs_cobra | +0.0091 | 0.5214 |
| primary_4split_fixed_0.5 | MCC | duet_tcn_vs_cobra | +0.0240 | 0.228 |
| primary_4split_fixed_0.5 | AUC | duet_dense_vs_cobra | +0.0091 | 0.5552 |
| primary_4split_fixed_0.5 | AUC | duet_ensemble_vs_cobra | +0.0269 | 0.02805 |
| primary_4split_fixed_0.5 | AUC | duet_ensemble_vs_duet_graph | +0.0431 | 1.235e-05 |
| primary_4split_fixed_0.5 | AUC | duet_graph_vs_cobra | -0.0162 | 0.2298 |
| primary_4split_fixed_0.5 | AUC | duet_tcn_vs_cobra | +0.0021 | 0.8125 |
| primary_4split_fixed_0.5 | AUPRC | duet_dense_vs_cobra | +0.0029 | 1 |
| primary_4split_fixed_0.5 | AUPRC | duet_ensemble_vs_cobra | +0.0348 | 0.01459 |
| primary_4split_fixed_0.5 | AUPRC | duet_ensemble_vs_duet_graph | +0.0356 | 0.0001603 |
| primary_4split_fixed_0.5 | AUPRC | duet_graph_vs_cobra | -0.0008 | 1 |
| primary_4split_fixed_0.5 | AUPRC | duet_tcn_vs_cobra | -0.0012 | 1 |
| primary_4split_fixed_0.5 | F1 | duet_dense_vs_cobra | +0.0414 | 0.3672 |
| primary_4split_fixed_0.5 | F1 | duet_ensemble_vs_cobra | +0.0589 | 0.2616 |
| primary_4split_fixed_0.5 | F1 | duet_ensemble_vs_duet_graph | +0.0241 | 0.3672 |
| primary_4split_fixed_0.5 | F1 | duet_graph_vs_cobra | +0.0348 | 0.3672 |
| primary_4split_fixed_0.5 | F1 | duet_tcn_vs_cobra | +0.0511 | 0.3672 |
| primary_4split_fixed_0.5 | Precision | duet_dense_vs_cobra | +0.0131 | 1 |
| primary_4split_fixed_0.5 | Precision | duet_ensemble_vs_cobra | +0.0257 | 0.3622 |
| primary_4split_fixed_0.5 | Precision | duet_ensemble_vs_duet_graph | +0.0304 | 0.0005503 |
| primary_4split_fixed_0.5 | Precision | duet_graph_vs_cobra | -0.0047 | 1 |
| primary_4split_fixed_0.5 | Precision | duet_tcn_vs_cobra | -0.0038 | 1 |
| primary_4split_fixed_0.5 | Recall | duet_dense_vs_cobra | +0.0428 | 0.7488 |
| primary_4split_fixed_0.5 | Recall | duet_ensemble_vs_cobra | +0.0600 | 0.6539 |
| primary_4split_fixed_0.5 | Recall | duet_ensemble_vs_duet_graph | +0.0428 | 0.5744 |
| primary_4split_fixed_0.5 | Recall | duet_graph_vs_cobra | +0.0172 | 0.7488 |
| primary_4split_fixed_0.5 | Recall | duet_tcn_vs_cobra | +0.0711 | 0.6539 |
| primary_4split_fixed_0.5 | GMean | duet_dense_vs_cobra | +0.0390 | 0.3515 |
| primary_4split_fixed_0.5 | GMean | duet_ensemble_vs_cobra | +0.0538 | 0.2642 |
| primary_4split_fixed_0.5 | GMean | duet_ensemble_vs_duet_graph | +0.0206 | 0.3515 |
| primary_4split_fixed_0.5 | GMean | duet_graph_vs_cobra | +0.0331 | 0.3515 |
| primary_4split_fixed_0.5 | GMean | duet_tcn_vs_cobra | +0.0465 | 0.3515 |
| structure_3split | MCC | graph_seq3d_vs_duet_graph | +0.0418 | 0.0004477 |
| structure_3split | MCC | graph_seqss3d_vs_graph_seq3d | -0.0320 | 0.007856 |
| structure_3split | MCC | grassp_ernie_vs_cobra | +0.0752 | 0.0007908 |
| structure_3split | MCC | grassp_ernie_vs_duet_ensemble | +0.0270 | 0.01274 |
| structure_3split | MCC | grassp_ernie_vs_graph_seqss3d | +0.0741 | 0.0006411 |
| structure_3split | MCC | grassp_rnafm_vs_duet_ensemble | +0.0332 | 6.699e-05 |
| structure_3split | AUC | graph_seq3d_vs_duet_graph | +0.0198 | 0.0005658 |
| structure_3split | AUC | graph_seqss3d_vs_graph_seq3d | -0.0237 | 0.01034 |
| structure_3split | AUC | grassp_ernie_vs_cobra | +0.0220 | 0.09201 |
| structure_3split | AUC | grassp_ernie_vs_duet_ensemble | -0.0013 | 0.6996 |
| structure_3split | AUC | grassp_ernie_vs_graph_seqss3d | +0.0493 | 0.0001219 |
| structure_3split | AUC | grassp_rnafm_vs_duet_ensemble | +0.0137 | 0.01047 |
| structure_3split | AUPRC | graph_seq3d_vs_duet_graph | +0.0136 | 0.06891 |
| structure_3split | AUPRC | graph_seqss3d_vs_graph_seq3d | -0.0218 | 0.06891 |
| structure_3split | AUPRC | grassp_ernie_vs_cobra | +0.0275 | 0.08534 |
| structure_3split | AUPRC | grassp_ernie_vs_duet_ensemble | -0.0004 | 0.9479 |
| structure_3split | AUPRC | grassp_ernie_vs_graph_seqss3d | +0.0435 | 0.001244 |
| structure_3split | AUPRC | grassp_rnafm_vs_duet_ensemble | +0.0154 | 0.06891 |
| structure_3split | F1 | graph_seq3d_vs_duet_graph | +0.0536 | 0.01148 |
| structure_3split | F1 | graph_seqss3d_vs_graph_seq3d | -0.0089 | 0.5936 |
| structure_3split | F1 | grassp_ernie_vs_cobra | +0.1214 | 0.001723 |
| structure_3split | F1 | grassp_ernie_vs_duet_ensemble | +0.0642 | 0.001723 |
| structure_3split | F1 | grassp_ernie_vs_graph_seqss3d | +0.0840 | 0.001308 |
| structure_3split | F1 | grassp_rnafm_vs_duet_ensemble | +0.0746 | 0.0002868 |
| structure_3split | Precision | graph_seq3d_vs_duet_graph | +0.0151 | 0.414 |
| structure_3split | Precision | graph_seqss3d_vs_graph_seq3d | -0.0323 | 0.0228 |
| structure_3split | Precision | grassp_ernie_vs_cobra | -0.0050 | 0.6941 |
| structure_3split | Precision | grassp_ernie_vs_duet_ensemble | -0.0285 | 0.008215 |
| structure_3split | Precision | grassp_ernie_vs_graph_seqss3d | +0.0142 | 0.335 |
| structure_3split | Precision | grassp_rnafm_vs_duet_ensemble | -0.0353 | 0.002542 |
| structure_3split | Recall | graph_seq3d_vs_duet_graph | +0.0828 | 0.04982 |
| structure_3split | Recall | graph_seqss3d_vs_graph_seq3d | +0.0114 | 0.7499 |
| structure_3split | Recall | grassp_ernie_vs_cobra | +0.2290 | 0.0007749 |
| structure_3split | Recall | grassp_ernie_vs_duet_ensemble | +0.1649 | 0.0004622 |
| structure_3split | Recall | grassp_ernie_vs_graph_seqss3d | +0.1673 | 0.001006 |
| structure_3split | Recall | grassp_rnafm_vs_duet_ensemble | +0.2054 | 0.0001376 |
| structure_3split | GMean | graph_seq3d_vs_duet_graph | +0.0433 | 0.01526 |
| structure_3split | GMean | graph_seqss3d_vs_graph_seq3d | -0.0077 | 0.5643 |
| structure_3split | GMean | grassp_ernie_vs_cobra | +0.0950 | 0.00261 |
| structure_3split | GMean | grassp_ernie_vs_duet_ensemble | +0.0454 | 0.003767 |
| structure_3split | GMean | grassp_ernie_vs_graph_seqss3d | +0.0628 | 0.001914 |
| structure_3split | GMean | grassp_rnafm_vs_duet_ensemble | +0.0492 | 0.0006672 |
| structure_3split_fixed_0.5 | MCC | graph_seq3d_vs_duet_graph | +0.0400 | 0.0002192 |
| structure_3split_fixed_0.5 | MCC | graph_seqss3d_vs_graph_seq3d | -0.0328 | 0.004222 |
| structure_3split_fixed_0.5 | MCC | grassp_ernie_vs_cobra | +0.0915 | 0.0002804 |
| structure_3split_fixed_0.5 | MCC | grassp_ernie_vs_duet_ensemble | +0.0384 | 0.005925 |
| structure_3split_fixed_0.5 | MCC | grassp_ernie_vs_graph_seqss3d | +0.0768 | 0.0006163 |
| structure_3split_fixed_0.5 | MCC | grassp_rnafm_vs_duet_ensemble | +0.0443 | 0.001456 |
| structure_3split_fixed_0.5 | AUC | graph_seq3d_vs_duet_graph | +0.0198 | 0.0005658 |
| structure_3split_fixed_0.5 | AUC | graph_seqss3d_vs_graph_seq3d | -0.0237 | 0.01034 |
| structure_3split_fixed_0.5 | AUC | grassp_ernie_vs_cobra | +0.0220 | 0.09201 |
| structure_3split_fixed_0.5 | AUC | grassp_ernie_vs_duet_ensemble | -0.0013 | 0.6996 |
| structure_3split_fixed_0.5 | AUC | grassp_ernie_vs_graph_seqss3d | +0.0493 | 0.0001219 |
| structure_3split_fixed_0.5 | AUC | grassp_rnafm_vs_duet_ensemble | +0.0137 | 0.01047 |
| structure_3split_fixed_0.5 | AUPRC | graph_seq3d_vs_duet_graph | +0.0136 | 0.06891 |
| structure_3split_fixed_0.5 | AUPRC | graph_seqss3d_vs_graph_seq3d | -0.0218 | 0.06891 |
| structure_3split_fixed_0.5 | AUPRC | grassp_ernie_vs_cobra | +0.0275 | 0.08534 |
| structure_3split_fixed_0.5 | AUPRC | grassp_ernie_vs_duet_ensemble | -0.0004 | 0.9479 |
| structure_3split_fixed_0.5 | AUPRC | grassp_ernie_vs_graph_seqss3d | +0.0435 | 0.001244 |
| structure_3split_fixed_0.5 | AUPRC | grassp_rnafm_vs_duet_ensemble | +0.0154 | 0.06891 |
| structure_3split_fixed_0.5 | F1 | graph_seq3d_vs_duet_graph | +0.0400 | 0.0647 |
| structure_3split_fixed_0.5 | F1 | graph_seqss3d_vs_graph_seq3d | -0.0075 | 0.6902 |
| structure_3split_fixed_0.5 | F1 | grassp_ernie_vs_cobra | +0.1736 | 0.0003798 |
| structure_3split_fixed_0.5 | F1 | grassp_ernie_vs_duet_ensemble | +0.1076 | 0.0003798 |
| structure_3split_fixed_0.5 | F1 | grassp_ernie_vs_graph_seqss3d | +0.0963 | 0.0005088 |
| structure_3split_fixed_0.5 | F1 | grassp_rnafm_vs_duet_ensemble | +0.1173 | 0.0001471 |
| structure_3split_fixed_0.5 | Precision | graph_seq3d_vs_duet_graph | +0.0229 | 0.1709 |
| structure_3split_fixed_0.5 | Precision | graph_seqss3d_vs_graph_seq3d | -0.0358 | 0.006594 |
| structure_3split_fixed_0.5 | Precision | grassp_ernie_vs_cobra | -0.0174 | 0.465 |
| structure_3split_fixed_0.5 | Precision | grassp_ernie_vs_duet_ensemble | -0.0459 | 0.0004285 |
| structure_3split_fixed_0.5 | Precision | grassp_ernie_vs_graph_seqss3d | +0.0098 | 0.465 |
| structure_3split_fixed_0.5 | Precision | grassp_rnafm_vs_duet_ensemble | -0.0519 | 0.0002055 |
| structure_3split_fixed_0.5 | Recall | graph_seq3d_vs_duet_graph | +0.0540 | 0.231 |
| structure_3split_fixed_0.5 | Recall | graph_seqss3d_vs_graph_seq3d | +0.0129 | 0.7264 |
| structure_3split_fixed_0.5 | Recall | grassp_ernie_vs_cobra | +0.3087 | 3.936e-05 |
| structure_3split_fixed_0.5 | Recall | grassp_ernie_vs_duet_ensemble | +0.2466 | 1.759e-05 |
| structure_3split_fixed_0.5 | Recall | grassp_ernie_vs_graph_seqss3d | +0.1945 | 0.0003094 |
| structure_3split_fixed_0.5 | Recall | grassp_rnafm_vs_duet_ensemble | +0.2831 | 1.067e-05 |
| structure_3split_fixed_0.5 | GMean | graph_seq3d_vs_duet_graph | +0.0321 | 0.08623 |
| structure_3split_fixed_0.5 | GMean | graph_seqss3d_vs_graph_seq3d | -0.0056 | 0.718 |
| structure_3split_fixed_0.5 | GMean | grassp_ernie_vs_cobra | +0.1400 | 0.0008935 |
| structure_3split_fixed_0.5 | GMean | grassp_ernie_vs_duet_ensemble | +0.0821 | 0.0008935 |
| structure_3split_fixed_0.5 | GMean | grassp_ernie_vs_graph_seqss3d | +0.0720 | 0.000908 |
| structure_3split_fixed_0.5 | GMean | grassp_rnafm_vs_duet_ensemble | +0.0858 | 0.0003493 |

## Shuffled-3D control

| Metric | Comparison | Delta | 95% CI | raw p | Holm p |
|:--|:--|--:|:--|--:|--:|
| MCC | real3d_vs_seqss | +0.0418 | +0.0277 to +0.0560 | 8.954e-05 | 0.0004477 |
| MCC | real3d_vs_seqss3d | +0.0320 | +0.0132 to +0.0508 | 0.003928 | 0.01571 |
| MCC | shuffled_vs_cobra | -0.0036 | -0.0453 to +0.0381 | 0.8495 | 1 |
| MCC | shuffled_vs_real3d | -0.0366 | -0.0462 to -0.0270 | 1.235e-05 | 7.411e-05 |
| MCC | shuffled_vs_seqss | +0.0053 | -0.0100 to +0.0205 | 0.4549 | 1 |
| MCC | shuffled_vs_seqss3d | -0.0046 | -0.0280 to +0.0187 | 0.6659 | 1 |
| AUC | real3d_vs_seqss | +0.0198 | +0.0129 to +0.0267 | 0.0001132 | 0.0006789 |
| AUC | real3d_vs_seqss3d | +0.0237 | +0.0107 to +0.0368 | 0.002585 | 0.01034 |
| AUC | shuffled_vs_cobra | -0.0170 | -0.0401 to +0.0061 | 0.13 | 0.2206 |
| AUC | shuffled_vs_real3d | -0.0135 | -0.0190 to -0.0079 | 0.0004107 | 0.002054 |
| AUC | shuffled_vs_seqss | +0.0063 | +0.0002 to +0.0124 | 0.04371 | 0.1311 |
| AUC | shuffled_vs_seqss3d | +0.0103 | -0.0029 to +0.0234 | 0.1103 | 0.2206 |
| AUPRC | real3d_vs_seqss | +0.0136 | +0.0035 to +0.0237 | 0.01378 | 0.06891 |
| AUPRC | real3d_vs_seqss3d | +0.0218 | +0.0050 to +0.0386 | 0.01643 | 0.06891 |
| AUPRC | shuffled_vs_cobra | -0.0055 | -0.0275 to +0.0166 | 0.5881 | 1 |
| AUPRC | shuffled_vs_real3d | -0.0114 | -0.0187 to -0.0041 | 0.006503 | 0.03902 |
| AUPRC | shuffled_vs_seqss | +0.0022 | -0.0062 to +0.0106 | 0.5656 | 1 |
| AUPRC | shuffled_vs_seqss3d | +0.0105 | -0.0043 to +0.0253 | 0.1445 | 0.4335 |
| F1 | real3d_vs_seqss | +0.0536 | +0.0199 to +0.0873 | 0.005738 | 0.02869 |
| F1 | real3d_vs_seqss3d | +0.0089 | -0.0276 to +0.0454 | 0.5936 | 1 |
| F1 | shuffled_vs_cobra | -0.0237 | -0.1052 to +0.0578 | 0.5269 | 1 |
| F1 | shuffled_vs_real3d | -0.0700 | -0.0852 to -0.0548 | 2.531e-06 | 1.518e-05 |
| F1 | shuffled_vs_seqss | -0.0164 | -0.0629 to +0.0301 | 0.4459 | 1 |
| F1 | shuffled_vs_seqss3d | -0.0611 | -0.1069 to -0.0153 | 0.01453 | 0.0581 |
| Precision | real3d_vs_seqss | +0.0151 | -0.0100 to +0.0402 | 0.207 | 0.621 |
| Precision | real3d_vs_seqss3d | +0.0323 | +0.0120 to +0.0526 | 0.005699 | 0.03235 |
| Precision | shuffled_vs_cobra | +0.0150 | -0.0206 to +0.0505 | 0.3663 | 0.7325 |
| Precision | shuffled_vs_real3d | +0.0018 | -0.0073 to +0.0109 | 0.661 | 0.7325 |
| Precision | shuffled_vs_seqss | +0.0169 | -0.0056 to +0.0394 | 0.1234 | 0.4935 |
| Precision | shuffled_vs_seqss3d | +0.0342 | +0.0129 to +0.0554 | 0.005391 | 0.03235 |
| Recall | real3d_vs_seqss | +0.0828 | +0.0131 to +0.1525 | 0.02491 | 0.09965 |
| Recall | real3d_vs_seqss3d | -0.0114 | -0.0898 to +0.0670 | 0.7499 | 0.9581 |
| Recall | shuffled_vs_cobra | -0.0593 | -0.1867 to +0.0680 | 0.3194 | 0.9581 |
| Recall | shuffled_vs_real3d | -0.1097 | -0.1212 to -0.0982 | 4.766e-09 | 2.859e-08 |
| Recall | shuffled_vs_seqss | -0.0269 | -0.1014 to +0.0475 | 0.4344 | 0.9581 |
| Recall | shuffled_vs_seqss3d | -0.1211 | -0.2012 to -0.0410 | 0.007635 | 0.03817 |
| GMean | real3d_vs_seqss | +0.0433 | +0.0147 to +0.0720 | 0.007629 | 0.03814 |
| GMean | real3d_vs_seqss3d | +0.0077 | -0.0213 to +0.0367 | 0.5643 | 1 |
| GMean | shuffled_vs_cobra | -0.0189 | -0.0886 to +0.0507 | 0.5537 | 1 |
| GMean | shuffled_vs_real3d | -0.0588 | -0.0734 to -0.0442 | 7.747e-06 | 4.648e-05 |
| GMean | shuffled_vs_seqss | -0.0154 | -0.0565 to +0.0256 | 0.4162 | 1 |
| GMean | shuffled_vs_seqss3d | -0.0511 | -0.0896 to -0.0127 | 0.01473 | 0.05892 |

## Fixed-composition training-size curve

| Method | 80 MCC | 160 MCC | 290 MCC | 160-80 | 290-80 |
|:--|--:|--:|--:|--:|--:|
| cobra | 0.098 | 0.138 | 0.192 | +0.040 (Holm 0.000938) | +0.094 (Holm 0.000256) |
| duet_graph | 0.123 | 0.187 | 0.193 | +0.064 (Holm 3.2e-05) | +0.070 (Holm 6.81e-05) |
| duet_tcn | 0.087 | 0.168 | 0.208 | +0.081 (Holm 0.000938) | +0.121 (Holm 3.57e-05) |
| duet_dense | 0.094 | 0.147 | 0.213 | +0.052 (Holm 0.00118) | +0.119 (Holm 4.83e-07) |
| duet_ensemble | 0.105 | 0.198 | 0.240 | +0.093 (Holm 1.61e-05) | +0.135 (Holm 1.72e-07) |
| graph_seq3d | 0.105 | 0.118 | 0.212 | +0.012 (Holm 0.307) | +0.107 (Holm 2.2e-06) |
| graph_seq3d_ensemble | 0.111 | 0.198 | 0.239 | +0.087 (Holm 6.81e-05) | +0.128 (Holm 7.63e-07) |
| grassp_rnafm | 0.134 | 0.220 | 0.261 | +0.086 (Holm 0.00046) | +0.127 (Holm 1.11e-05) |

Full seven-metric curves are in `SIZE_CURVE.md`, `SIZE_CURVE.pdf`, and `SIZE_CURVE_source_data.csv`.

## Repeated cluster-stratified CV

| Assignment | Positive-residue range | DUET ensemble - CoBRA (MCC) | Holm p |
|--:|--:|--:|--:|
| 101 | 1 | +0.0633 | 0.1227 |
| 102 | 10 | +0.0772 | 0.02361 |
| 103 | 3 | -0.0078 | 0.4199 |

## GRASSP hyperparameter sensitivity

| Encoder | Selected configuration | OOF MCC | Final - DUET (MCC) | Holm p |
|:--|:--|--:|--:|--:|
| ernie | lr0.0001_bs1_h128_d0.3 | 0.2510 | -0.0127 | 0.4185 |
| rnafm | lr0.001_bs16_h128_d0.3 | 0.2433 | -0.0192 | 0.02544 |

## Duplicate-weighting ablation

| Model | Comparison | MCC delta | Holm p |
|:--|:--|--:|--:|
| duet_graph | dedup_vs_group | -0.0502 | 0.000592 |
| duet_graph | dedup_vs_uniform | -0.0390 | 0.003564 |
| duet_graph | uniform_vs_group | -0.0112 | 0.1979 |
| grassp_rnafm | dedup_vs_group | -0.1422 | 0.01051 |
| grassp_rnafm | dedup_vs_uniform | -0.1362 | 0.01051 |
| grassp_rnafm | uniform_vs_group | -0.0060 | 0.3868 |
