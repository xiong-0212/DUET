# All-metric significance analysis

Each paired contrast uses ten paired training seeds. The independent resampling unit is the benchmark chain. Raw and Holm-adjusted p values are reported for both paired tests and chain bootstrap.

## Method summaries

| Context | Method | Metric | Mean | SD |
|:--|:--|:--|--:|--:|
| four_split | cobra | MCC | 0.1922 | 0.0308 |
| four_split | cobra | AUC | 0.6571 | 0.0226 |
| four_split | cobra | AUPRC | 0.4507 | 0.0242 |
| four_split | cobra | F1 | 0.4252 | 0.0539 |
| four_split | cobra | Precision | 0.4458 | 0.0328 |
| four_split | cobra | Recall | 0.4690 | 0.1075 |
| four_split | cobra | GMean | 0.5574 | 0.0453 |
| four_split | duet_graph | MCC | 0.1928 | 0.0169 |
| four_split | duet_graph | AUC | 0.6410 | 0.0136 |
| four_split | duet_graph | AUPRC | 0.4498 | 0.0141 |
| four_split | duet_graph | F1 | 0.4217 | 0.0285 |
| four_split | duet_graph | Precision | 0.4566 | 0.0113 |
| four_split | duet_graph | Recall | 0.4106 | 0.0502 |
| four_split | duet_graph | GMean | 0.5593 | 0.0256 |
| four_split | duet_tcn | MCC | 0.2078 | 0.0238 |
| four_split | duet_tcn | AUC | 0.6592 | 0.0114 |
| four_split | duet_tcn | AUPRC | 0.4495 | 0.0126 |
| four_split | duet_tcn | F1 | 0.4380 | 0.0445 |
| four_split | duet_tcn | Precision | 0.4574 | 0.0149 |
| four_split | duet_tcn | Recall | 0.4645 | 0.0880 |
| four_split | duet_tcn | GMean | 0.5727 | 0.0369 |
| four_split | duet_dense | MCC | 0.2132 | 0.0119 |
| four_split | duet_dense | AUC | 0.6663 | 0.0068 |
| four_split | duet_dense | AUPRC | 0.4535 | 0.0105 |
| four_split | duet_dense | F1 | 0.4223 | 0.0297 |
| four_split | duet_dense | Precision | 0.4751 | 0.0197 |
| four_split | duet_dense | Recall | 0.4257 | 0.0532 |
| four_split | duet_dense | GMean | 0.5601 | 0.0271 |
| four_split | duet_ensemble | MCC | 0.2396 | 0.0125 |
| four_split | duet_ensemble | AUC | 0.6840 | 0.0045 |
| four_split | duet_ensemble | AUPRC | 0.4855 | 0.0107 |
| four_split | duet_ensemble | F1 | 0.4755 | 0.0223 |
| four_split | duet_ensemble | Precision | 0.4668 | 0.0176 |
| four_split | duet_ensemble | Recall | 0.5277 | 0.0589 |
| four_split | duet_ensemble | GMean | 0.6039 | 0.0173 |
| three_split | graph_seq3d | MCC | 0.2123 | 0.0220 |
| three_split | graph_seq3d | AUC | 0.6455 | 0.0115 |
| three_split | graph_seq3d | AUPRC | 0.4719 | 0.0178 |
| three_split | graph_seq3d | F1 | 0.4624 | 0.0339 |
| three_split | graph_seq3d | Precision | 0.4924 | 0.0279 |
| three_split | graph_seq3d | Recall | 0.4443 | 0.0737 |
| three_split | graph_seq3d | GMean | 0.5770 | 0.0275 |
| three_split | graph_seqss3d | MCC | 0.1803 | 0.0251 |
| three_split | graph_seqss3d | AUC | 0.6217 | 0.0169 |
| three_split | graph_seqss3d | AUPRC | 0.4501 | 0.0199 |
| three_split | graph_seqss3d | F1 | 0.4535 | 0.0416 |
| three_split | graph_seqss3d | Precision | 0.4601 | 0.0138 |
| three_split | graph_seqss3d | Recall | 0.4557 | 0.0823 |
| three_split | graph_seqss3d | GMean | 0.5693 | 0.0323 |
| three_split | grassp_ernie | MCC | 0.2544 | 0.0183 |
| three_split | grassp_ernie | AUC | 0.6710 | 0.0053 |
| three_split | grassp_ernie | AUPRC | 0.4935 | 0.0180 |
| three_split | grassp_ernie | F1 | 0.5375 | 0.0122 |
| three_split | grassp_ernie | Precision | 0.4743 | 0.0139 |
| three_split | grassp_ernie | Recall | 0.6229 | 0.0346 |
| three_split | grassp_ernie | GMean | 0.6321 | 0.0099 |
| three_split | grassp_rnafm | MCC | 0.2607 | 0.0135 |
| three_split | grassp_rnafm | AUC | 0.6860 | 0.0076 |
| three_split | grassp_rnafm | AUPRC | 0.5093 | 0.0114 |
| three_split | grassp_rnafm | F1 | 0.5480 | 0.0070 |
| three_split | grassp_rnafm | Precision | 0.4675 | 0.0108 |
| three_split | grassp_rnafm | Recall | 0.6635 | 0.0220 |
| three_split | grassp_rnafm | GMean | 0.6359 | 0.0070 |
| three_split | rgcn_ernie | MCC | 0.1609 | 0.0307 |
| three_split | rgcn_ernie | AUC | 0.6011 | 0.0232 |
| three_split | rgcn_ernie | AUPRC | 0.4505 | 0.0214 |
| three_split | rgcn_ernie | F1 | 0.4227 | 0.0284 |
| three_split | rgcn_ernie | Precision | 0.4579 | 0.0232 |
| three_split | rgcn_ernie | Recall | 0.3954 | 0.0427 |
| three_split | rgcn_ernie | GMean | 0.5459 | 0.0236 |
| three_split | graph_seq3d_ensemble | MCC | 0.2392 | 0.0198 |
| three_split | graph_seq3d_ensemble | AUC | 0.6782 | 0.0088 |
| three_split | graph_seq3d_ensemble | AUPRC | 0.5008 | 0.0150 |
| three_split | graph_seq3d_ensemble | F1 | 0.5001 | 0.0312 |
| three_split | graph_seq3d_ensemble | Precision | 0.4933 | 0.0162 |
| three_split | graph_seq3d_ensemble | Recall | 0.5177 | 0.0743 |
| three_split | graph_seq3d_ensemble | GMean | 0.6074 | 0.0234 |
| three_split | graph_seqss3d_ensemble | MCC | 0.2361 | 0.0224 |
| three_split | graph_seqss3d_ensemble | AUC | 0.6693 | 0.0116 |
| three_split | graph_seqss3d_ensemble | AUPRC | 0.4940 | 0.0144 |
| three_split | graph_seqss3d_ensemble | F1 | 0.4691 | 0.0346 |
| three_split | graph_seqss3d_ensemble | Precision | 0.5169 | 0.0161 |
| three_split | graph_seqss3d_ensemble | Recall | 0.4378 | 0.0634 |
| three_split | graph_seqss3d_ensemble | GMean | 0.5834 | 0.0288 |
| external_independent | cobra | MCC | 0.1367 | 0.0660 |
| external_independent | cobra | AUC | 0.6573 | 0.0464 |
| external_independent | cobra | AUPRC | 0.2099 | 0.0317 |
| external_independent | cobra | F1 | 0.2634 | 0.0520 |
| external_independent | cobra | Precision | 0.1842 | 0.0211 |
| external_independent | cobra | Recall | 0.5204 | 0.1837 |
| external_independent | cobra | GMean | 0.5427 | 0.0832 |
| external_independent | duet_graph | MCC | 0.2130 | 0.0581 |
| external_independent | duet_graph | AUC | 0.6945 | 0.0229 |
| external_independent | duet_graph | AUPRC | 0.2660 | 0.0296 |
| external_independent | duet_graph | F1 | 0.3288 | 0.0465 |
| external_independent | duet_graph | Precision | 0.2656 | 0.0390 |
| external_independent | duet_graph | Recall | 0.4442 | 0.0782 |
| external_independent | duet_graph | GMean | 0.5930 | 0.0479 |
| external_independent | duet_tcn | MCC | 0.2156 | 0.0471 |
| external_independent | duet_tcn | AUC | 0.6947 | 0.0293 |
| external_independent | duet_tcn | AUPRC | 0.2498 | 0.0388 |
| external_independent | duet_tcn | F1 | 0.3257 | 0.0312 |
| external_independent | duet_tcn | Precision | 0.2333 | 0.0353 |
| external_independent | duet_tcn | Recall | 0.5713 | 0.0786 |
| external_independent | duet_tcn | GMean | 0.6153 | 0.0254 |
| external_independent | duet_dense | MCC | 0.1996 | 0.0539 |
| external_independent | duet_dense | AUC | 0.7188 | 0.0230 |
| external_independent | duet_dense | AUPRC | 0.2625 | 0.0388 |
| external_independent | duet_dense | F1 | 0.3113 | 0.0386 |
| external_independent | duet_dense | Precision | 0.2261 | 0.0225 |
| external_independent | duet_dense | Recall | 0.5288 | 0.1067 |
| external_independent | duet_dense | GMean | 0.5939 | 0.0545 |
| external_independent | duet_ensemble | MCC | 0.2460 | 0.0424 |
| external_independent | duet_ensemble | AUC | 0.7302 | 0.0160 |
| external_independent | duet_ensemble | AUPRC | 0.2726 | 0.0162 |
| external_independent | duet_ensemble | F1 | 0.3435 | 0.0268 |
| external_independent | duet_ensemble | Precision | 0.2365 | 0.0268 |
| external_independent | duet_ensemble | Recall | 0.6553 | 0.0804 |
| external_independent | duet_ensemble | GMean | 0.6468 | 0.0271 |
| external_independent | graph_seq3d | MCC | 0.2098 | 0.0307 |
| external_independent | graph_seq3d | AUC | 0.7059 | 0.0195 |
| external_independent | graph_seq3d | AUPRC | 0.2729 | 0.0292 |
| external_independent | graph_seq3d | F1 | 0.3265 | 0.0217 |
| external_independent | graph_seq3d | Precision | 0.2349 | 0.0276 |
| external_independent | graph_seq3d | Recall | 0.5565 | 0.0825 |
| external_independent | graph_seq3d | GMean | 0.6263 | 0.0247 |
| external_independent | graph_seqss3d | MCC | 0.1974 | 0.0452 |
| external_independent | graph_seqss3d | AUC | 0.6875 | 0.0298 |
| external_independent | graph_seqss3d | AUPRC | 0.2704 | 0.0377 |
| external_independent | graph_seqss3d | F1 | 0.3161 | 0.0307 |
| external_independent | graph_seqss3d | Precision | 0.2199 | 0.0152 |
| external_independent | graph_seqss3d | Recall | 0.5794 | 0.1196 |
| external_independent | graph_seqss3d | GMean | 0.6273 | 0.0481 |
| external_independent | grassp_ernie | MCC | 0.1265 | 0.0580 |
| external_independent | grassp_ernie | AUC | 0.5911 | 0.0289 |
| external_independent | grassp_ernie | AUPRC | 0.1547 | 0.0129 |
| external_independent | grassp_ernie | F1 | 0.2695 | 0.0328 |
| external_independent | grassp_ernie | Precision | 0.1728 | 0.0199 |
| external_independent | grassp_ernie | Recall | 0.6625 | 0.1205 |
| external_independent | grassp_ernie | GMean | 0.5781 | 0.0400 |
| external_independent | grassp_rnafm | MCC | 0.2433 | 0.0243 |
| external_independent | grassp_rnafm | AUC | 0.7403 | 0.0157 |
| external_independent | grassp_rnafm | AUPRC | 0.3120 | 0.0387 |
| external_independent | grassp_rnafm | F1 | 0.3344 | 0.0157 |
| external_independent | grassp_rnafm | Precision | 0.2132 | 0.0127 |
| external_independent | grassp_rnafm | Recall | 0.7868 | 0.0431 |
| external_independent | grassp_rnafm | GMean | 0.6640 | 0.0191 |
| external_independent | rgcn_ernie | MCC | -0.1226 | 0.0336 |
| external_independent | rgcn_ernie | AUC | 0.3708 | 0.0448 |
| external_independent | rgcn_ernie | AUPRC | 0.1052 | 0.0130 |
| external_independent | rgcn_ernie | F1 | 0.0674 | 0.0367 |
| external_independent | rgcn_ernie | Precision | 0.0473 | 0.0288 |
| external_independent | rgcn_ernie | Recall | 0.1432 | 0.0654 |
| external_independent | rgcn_ernie | GMean | 0.2403 | 0.0926 |
| external_independent | graph_seq3d_ensemble | MCC | 0.2400 | 0.0365 |
| external_independent | graph_seq3d_ensemble | AUC | 0.7379 | 0.0193 |
| external_independent | graph_seq3d_ensemble | AUPRC | 0.2818 | 0.0370 |
| external_independent | graph_seq3d_ensemble | F1 | 0.3369 | 0.0240 |
| external_independent | graph_seq3d_ensemble | Precision | 0.2243 | 0.0237 |
| external_independent | graph_seq3d_ensemble | Recall | 0.7005 | 0.0748 |
| external_independent | graph_seq3d_ensemble | GMean | 0.6461 | 0.0266 |
| external_independent | graph_seqss3d_ensemble | MCC | 0.2507 | 0.0390 |
| external_independent | graph_seqss3d_ensemble | AUC | 0.7345 | 0.0211 |
| external_independent | graph_seqss3d_ensemble | AUPRC | 0.2803 | 0.0305 |
| external_independent | graph_seqss3d_ensemble | F1 | 0.3493 | 0.0262 |
| external_independent | graph_seqss3d_ensemble | Precision | 0.2438 | 0.0252 |
| external_independent | graph_seqss3d_ensemble | Recall | 0.6392 | 0.0692 |
| external_independent | graph_seqss3d_ensemble | GMean | 0.6509 | 0.0254 |
| source_matched_control | cobra | MCC | 0.0955 | 0.0335 |
| source_matched_control | cobra | AUC | 0.6476 | 0.0522 |
| source_matched_control | cobra | AUPRC | 0.1134 | 0.0295 |
| source_matched_control | cobra | F1 | 0.1310 | 0.0279 |
| source_matched_control | cobra | Precision | 0.0897 | 0.0256 |
| source_matched_control | cobra | Recall | 0.2730 | 0.0726 |
| source_matched_control | cobra | GMean | 0.4862 | 0.0596 |
| source_matched_control | duet_graph | MCC | 0.0701 | 0.0303 |
| source_matched_control | duet_graph | AUC | 0.5816 | 0.0248 |
| source_matched_control | duet_graph | AUPRC | 0.0797 | 0.0192 |
| source_matched_control | duet_graph | F1 | 0.1060 | 0.0198 |
| source_matched_control | duet_graph | Precision | 0.0631 | 0.0114 |
| source_matched_control | duet_graph | Recall | 0.3432 | 0.0883 |
| source_matched_control | duet_graph | GMean | 0.5200 | 0.0660 |
| source_matched_control | duet_tcn | MCC | 0.0948 | 0.0404 |
| source_matched_control | duet_tcn | AUC | 0.6355 | 0.0229 |
| source_matched_control | duet_tcn | AUPRC | 0.0993 | 0.0241 |
| source_matched_control | duet_tcn | F1 | 0.1308 | 0.0350 |
| source_matched_control | duet_tcn | Precision | 0.0895 | 0.0272 |
| source_matched_control | duet_tcn | Recall | 0.2622 | 0.0754 |
| source_matched_control | duet_tcn | GMean | 0.4777 | 0.0692 |
| source_matched_control | duet_dense | MCC | 0.1279 | 0.0309 |
| source_matched_control | duet_dense | AUC | 0.6852 | 0.0487 |
| source_matched_control | duet_dense | AUPRC | 0.1113 | 0.0147 |
| source_matched_control | duet_dense | F1 | 0.1623 | 0.0287 |
| source_matched_control | duet_dense | Precision | 0.1232 | 0.0286 |
| source_matched_control | duet_dense | Recall | 0.2486 | 0.0438 |
| source_matched_control | duet_dense | GMean | 0.4786 | 0.0402 |
| source_matched_control | duet_ensemble | MCC | 0.1194 | 0.0295 |
| source_matched_control | duet_ensemble | AUC | 0.6620 | 0.0225 |
| source_matched_control | duet_ensemble | AUPRC | 0.1025 | 0.0185 |
| source_matched_control | duet_ensemble | F1 | 0.1502 | 0.0265 |
| source_matched_control | duet_ensemble | Precision | 0.1004 | 0.0228 |
| source_matched_control | duet_ensemble | Recall | 0.3135 | 0.0463 |
| source_matched_control | duet_ensemble | GMean | 0.5256 | 0.0363 |
| source_matched_control | graph_seq3d | MCC | 0.0379 | 0.0271 |
| source_matched_control | graph_seq3d | AUC | 0.5811 | 0.0293 |
| source_matched_control | graph_seq3d | AUPRC | 0.0532 | 0.0082 |
| source_matched_control | graph_seq3d | F1 | 0.0853 | 0.0149 |
| source_matched_control | graph_seq3d | Precision | 0.0491 | 0.0086 |
| source_matched_control | graph_seq3d | Recall | 0.3378 | 0.0857 |
| source_matched_control | graph_seq3d | GMean | 0.4979 | 0.0517 |
| source_matched_control | graph_seqss3d | MCC | 0.0456 | 0.0333 |
| source_matched_control | graph_seqss3d | AUC | 0.5826 | 0.0459 |
| source_matched_control | graph_seqss3d | AUPRC | 0.0540 | 0.0121 |
| source_matched_control | graph_seqss3d | F1 | 0.0899 | 0.0182 |
| source_matched_control | graph_seqss3d | Precision | 0.0514 | 0.0108 |
| source_matched_control | graph_seqss3d | Recall | 0.3676 | 0.0837 |
| source_matched_control | graph_seqss3d | GMean | 0.5166 | 0.0558 |
| source_matched_control | grassp_ernie | MCC | 0.0845 | 0.0602 |
| source_matched_control | grassp_ernie | AUC | 0.6131 | 0.0717 |
| source_matched_control | grassp_ernie | AUPRC | 0.0504 | 0.0076 |
| source_matched_control | grassp_ernie | F1 | 0.0908 | 0.0153 |
| source_matched_control | grassp_ernie | Precision | 0.0478 | 0.0082 |
| source_matched_control | grassp_ernie | Recall | 0.9027 | 0.1060 |
| source_matched_control | grassp_ernie | GMean | 0.5206 | 0.0884 |
| source_matched_control | grassp_rnafm | MCC | 0.1291 | 0.0253 |
| source_matched_control | grassp_rnafm | AUC | 0.7308 | 0.0284 |
| source_matched_control | grassp_rnafm | AUPRC | 0.1385 | 0.0181 |
| source_matched_control | grassp_rnafm | F1 | 0.1268 | 0.0134 |
| source_matched_control | grassp_rnafm | Precision | 0.0703 | 0.0078 |
| source_matched_control | grassp_rnafm | Recall | 0.6595 | 0.0756 |
| source_matched_control | grassp_rnafm | GMean | 0.6607 | 0.0327 |
| source_matched_control | rgcn_ernie | MCC | -0.1382 | 0.0116 |
| source_matched_control | rgcn_ernie | AUC | 0.3652 | 0.0764 |
| source_matched_control | rgcn_ernie | AUPRC | 0.0320 | 0.0115 |
| source_matched_control | rgcn_ernie | F1 | 0.0574 | 0.0011 |
| source_matched_control | rgcn_ernie | Precision | 0.0298 | 0.0006 |
| source_matched_control | rgcn_ernie | Recall | 0.7622 | 0.0171 |
| source_matched_control | rgcn_ernie | GMean | 0.2094 | 0.0110 |
| source_matched_control | graph_seq3d_ensemble | MCC | 0.1128 | 0.0313 |
| source_matched_control | graph_seq3d_ensemble | AUC | 0.6536 | 0.0281 |
| source_matched_control | graph_seq3d_ensemble | AUPRC | 0.0954 | 0.0189 |
| source_matched_control | graph_seq3d_ensemble | F1 | 0.1414 | 0.0268 |
| source_matched_control | graph_seq3d_ensemble | Precision | 0.0898 | 0.0210 |
| source_matched_control | graph_seq3d_ensemble | Recall | 0.3486 | 0.0517 |
| source_matched_control | graph_seq3d_ensemble | GMean | 0.5455 | 0.0380 |
| source_matched_control | graph_seqss3d_ensemble | MCC | 0.1167 | 0.0399 |
| source_matched_control | graph_seqss3d_ensemble | AUC | 0.6605 | 0.0317 |
| source_matched_control | graph_seqss3d_ensemble | AUPRC | 0.0887 | 0.0151 |
| source_matched_control | graph_seqss3d_ensemble | F1 | 0.1489 | 0.0341 |
| source_matched_control | graph_seqss3d_ensemble | Precision | 0.1020 | 0.0249 |
| source_matched_control | graph_seqss3d_ensemble | Recall | 0.2865 | 0.0734 |
| source_matched_control | graph_seqss3d_ensemble | GMean | 0.5038 | 0.0655 |

## Paired comparisons

| Family | Metric | Comparison | Delta | 95% CI | raw p | Holm p |
|:--|:--|:--|--:|:--|--:|--:|
| primary_4split | MCC | duet_ensemble_vs_cobra | +0.0474 | +0.0228 to +0.0721 | 0.001833 | 0.007333 |
| primary_4split | MCC | duet_graph_vs_cobra | +0.0007 | -0.0231 to +0.0245 | 0.9491 | 0.9491 |
| primary_4split | MCC | duet_tcn_vs_cobra | +0.0156 | -0.0144 to +0.0457 | 0.2695 | 0.5391 |
| primary_4split | MCC | duet_dense_vs_cobra | +0.0211 | -0.0052 to +0.0474 | 0.1036 | 0.3107 |
| primary_4split | MCC | duet_ensemble_vs_duet_graph | +0.0468 | +0.0308 to +0.0627 | 9.46e-05 | 0.000473 |
| primary_4split | AUC | duet_ensemble_vs_cobra | +0.0269 | +0.0094 to +0.0444 | 0.007011 | 0.02805 |
| primary_4split | AUC | duet_graph_vs_cobra | -0.0162 | -0.0345 to +0.0021 | 0.0766 | 0.2298 |
| primary_4split | AUC | duet_tcn_vs_cobra | +0.0021 | -0.0172 to +0.0214 | 0.8125 | 0.8125 |
| primary_4split | AUC | duet_dense_vs_cobra | +0.0091 | -0.0087 to +0.0270 | 0.2776 | 0.5552 |
| primary_4split | AUC | duet_ensemble_vs_duet_graph | +0.0431 | +0.0338 to +0.0524 | 2.47e-06 | 1.235e-05 |
| primary_4split | AUPRC | duet_ensemble_vs_cobra | +0.0348 | +0.0146 to +0.0550 | 0.003648 | 0.01459 |
| primary_4split | AUPRC | duet_graph_vs_cobra | -0.0008 | -0.0232 to +0.0215 | 0.9338 | 1 |
| primary_4split | AUPRC | duet_tcn_vs_cobra | -0.0012 | -0.0205 to +0.0182 | 0.8941 | 1 |
| primary_4split | AUPRC | duet_dense_vs_cobra | +0.0029 | -0.0170 to +0.0228 | 0.7525 | 1 |
| primary_4split | AUPRC | duet_ensemble_vs_duet_graph | +0.0356 | +0.0251 to +0.0462 | 3.207e-05 | 0.0001603 |
| primary_4split | F1 | duet_ensemble_vs_cobra | +0.0502 | +0.0041 to +0.0964 | 0.03607 | 0.1443 |
| primary_4split | F1 | duet_graph_vs_cobra | -0.0036 | -0.0492 to +0.0421 | 0.8639 | 1 |
| primary_4split | F1 | duet_tcn_vs_cobra | +0.0128 | -0.0423 to +0.0678 | 0.6132 | 1 |
| primary_4split | F1 | duet_dense_vs_cobra | -0.0030 | -0.0574 to +0.0515 | 0.9047 | 1 |
| primary_4split | F1 | duet_ensemble_vs_duet_graph | +0.0538 | +0.0281 to +0.0795 | 0.001062 | 0.005312 |
| primary_4split | Precision | duet_ensemble_vs_cobra | +0.0211 | -0.0074 to +0.0495 | 0.1286 | 0.5144 |
| primary_4split | Precision | duet_graph_vs_cobra | +0.0108 | -0.0079 to +0.0294 | 0.2232 | 0.5144 |
| primary_4split | Precision | duet_tcn_vs_cobra | +0.0116 | -0.0122 to +0.0354 | 0.2976 | 0.5144 |
| primary_4split | Precision | duet_dense_vs_cobra | +0.0293 | -0.0015 to +0.0601 | 0.05986 | 0.2993 |
| primary_4split | Precision | duet_ensemble_vs_duet_graph | +0.0103 | -0.0037 to +0.0242 | 0.1298 | 0.5144 |
| primary_4split | Recall | duet_ensemble_vs_cobra | +0.0587 | -0.0417 to +0.1590 | 0.2184 | 0.6553 |
| primary_4split | Recall | duet_graph_vs_cobra | -0.0585 | -0.1405 to +0.0236 | 0.1415 | 0.566 |
| primary_4split | Recall | duet_tcn_vs_cobra | -0.0045 | -0.1134 to +0.1044 | 0.9278 | 0.9278 |
| primary_4split | Recall | duet_dense_vs_cobra | -0.0433 | -0.1498 to +0.0632 | 0.3819 | 0.7637 |
| primary_4split | Recall | duet_ensemble_vs_duet_graph | +0.1171 | +0.0608 to +0.1735 | 0.001119 | 0.005595 |
| primary_4split | GMean | duet_ensemble_vs_cobra | +0.0464 | +0.0090 to +0.0839 | 0.02056 | 0.08223 |
| primary_4split | GMean | duet_graph_vs_cobra | +0.0019 | -0.0370 to +0.0408 | 0.9148 | 1 |
| primary_4split | GMean | duet_tcn_vs_cobra | +0.0152 | -0.0308 to +0.0612 | 0.4731 | 1 |
| primary_4split | GMean | duet_dense_vs_cobra | +0.0027 | -0.0439 to +0.0492 | 0.8996 | 1 |
| primary_4split | GMean | duet_ensemble_vs_duet_graph | +0.0445 | +0.0227 to +0.0664 | 0.001275 | 0.006374 |
| primary_4split_fixed_0.5 | MCC | duet_ensemble_vs_cobra | +0.0524 | +0.0237 to +0.0812 | 0.002557 | 0.01023 |
| primary_4split_fixed_0.5 | MCC | duet_graph_vs_cobra | +0.0091 | -0.0217 to +0.0398 | 0.5214 | 0.5214 |
| primary_4split_fixed_0.5 | MCC | duet_tcn_vs_cobra | +0.0240 | -0.0070 to +0.0551 | 0.114 | 0.228 |
| primary_4split_fixed_0.5 | MCC | duet_dense_vs_cobra | +0.0322 | +0.0034 to +0.0610 | 0.03232 | 0.09696 |
| primary_4split_fixed_0.5 | MCC | duet_ensemble_vs_duet_graph | +0.0434 | +0.0295 to +0.0572 | 5.837e-05 | 0.0002918 |
| primary_4split_fixed_0.5 | AUC | duet_ensemble_vs_cobra | +0.0269 | +0.0094 to +0.0444 | 0.007011 | 0.02805 |
| primary_4split_fixed_0.5 | AUC | duet_graph_vs_cobra | -0.0162 | -0.0345 to +0.0021 | 0.0766 | 0.2298 |
| primary_4split_fixed_0.5 | AUC | duet_tcn_vs_cobra | +0.0021 | -0.0172 to +0.0214 | 0.8125 | 0.8125 |
| primary_4split_fixed_0.5 | AUC | duet_dense_vs_cobra | +0.0091 | -0.0087 to +0.0270 | 0.2776 | 0.5552 |
| primary_4split_fixed_0.5 | AUC | duet_ensemble_vs_duet_graph | +0.0431 | +0.0338 to +0.0524 | 2.47e-06 | 1.235e-05 |
| primary_4split_fixed_0.5 | AUPRC | duet_ensemble_vs_cobra | +0.0348 | +0.0146 to +0.0550 | 0.003648 | 0.01459 |
| primary_4split_fixed_0.5 | AUPRC | duet_graph_vs_cobra | -0.0008 | -0.0232 to +0.0215 | 0.9338 | 1 |
| primary_4split_fixed_0.5 | AUPRC | duet_tcn_vs_cobra | -0.0012 | -0.0205 to +0.0182 | 0.8941 | 1 |
| primary_4split_fixed_0.5 | AUPRC | duet_dense_vs_cobra | +0.0029 | -0.0170 to +0.0228 | 0.7525 | 1 |
| primary_4split_fixed_0.5 | AUPRC | duet_ensemble_vs_duet_graph | +0.0356 | +0.0251 to +0.0462 | 3.207e-05 | 0.0001603 |
| primary_4split_fixed_0.5 | F1 | duet_ensemble_vs_cobra | +0.0589 | -0.0007 to +0.1186 | 0.05232 | 0.2616 |
| primary_4split_fixed_0.5 | F1 | duet_graph_vs_cobra | +0.0348 | -0.0209 to +0.0906 | 0.191 | 0.3672 |
| primary_4split_fixed_0.5 | F1 | duet_tcn_vs_cobra | +0.0511 | -0.0102 to +0.1125 | 0.09202 | 0.3672 |
| primary_4split_fixed_0.5 | F1 | duet_dense_vs_cobra | +0.0414 | -0.0206 to +0.1034 | 0.165 | 0.3672 |
| primary_4split_fixed_0.5 | F1 | duet_ensemble_vs_duet_graph | +0.0241 | -0.0048 to +0.0530 | 0.0918 | 0.3672 |
| primary_4split_fixed_0.5 | Precision | duet_ensemble_vs_cobra | +0.0257 | -0.0050 to +0.0564 | 0.09055 | 0.3622 |
| primary_4split_fixed_0.5 | Precision | duet_graph_vs_cobra | -0.0047 | -0.0304 to +0.0211 | 0.6917 | 1 |
| primary_4split_fixed_0.5 | Precision | duet_tcn_vs_cobra | -0.0038 | -0.0330 to +0.0254 | 0.7753 | 1 |
| primary_4split_fixed_0.5 | Precision | duet_dense_vs_cobra | +0.0131 | -0.0224 to +0.0486 | 0.4249 | 1 |
| primary_4split_fixed_0.5 | Precision | duet_ensemble_vs_duet_graph | +0.0304 | +0.0198 to +0.0409 | 0.0001101 | 0.0005503 |
| primary_4split_fixed_0.5 | Recall | duet_ensemble_vs_cobra | +0.0600 | -0.0379 to +0.1579 | 0.1993 | 0.6539 |
| primary_4split_fixed_0.5 | Recall | duet_graph_vs_cobra | +0.0172 | -0.0655 to +0.0999 | 0.6497 | 0.7488 |
| primary_4split_fixed_0.5 | Recall | duet_tcn_vs_cobra | +0.0711 | -0.0349 to +0.1772 | 0.1635 | 0.6539 |
| primary_4split_fixed_0.5 | Recall | duet_dense_vs_cobra | +0.0428 | -0.0608 to +0.1463 | 0.3744 | 0.7488 |
| primary_4split_fixed_0.5 | Recall | duet_ensemble_vs_duet_graph | +0.0428 | -0.0127 to +0.0983 | 0.1149 | 0.5744 |
| primary_4split_fixed_0.5 | GMean | duet_ensemble_vs_cobra | +0.0538 | -0.0008 to +0.1084 | 0.05284 | 0.2642 |
| primary_4split_fixed_0.5 | GMean | duet_graph_vs_cobra | +0.0331 | -0.0175 to +0.0838 | 0.173 | 0.3515 |
| primary_4split_fixed_0.5 | GMean | duet_tcn_vs_cobra | +0.0465 | -0.0084 to +0.1014 | 0.08787 | 0.3515 |
| primary_4split_fixed_0.5 | GMean | duet_dense_vs_cobra | +0.0390 | -0.0176 to +0.0957 | 0.1537 | 0.3515 |
| primary_4split_fixed_0.5 | GMean | duet_ensemble_vs_duet_graph | +0.0206 | -0.0050 to +0.0463 | 0.102 | 0.3515 |
| structure_3split | MCC | grassp_ernie_vs_duet_ensemble | +0.0270 | +0.0073 to +0.0467 | 0.01274 | 0.01274 |
| structure_3split | MCC | grassp_rnafm_vs_duet_ensemble | +0.0332 | +0.0246 to +0.0418 | 1.117e-05 | 6.699e-05 |
| structure_3split | MCC | grassp_ernie_vs_graph_seqss3d | +0.0741 | +0.0471 to +0.1012 | 0.0001603 | 0.0006411 |
| structure_3split | MCC | graph_seq3d_vs_duet_graph | +0.0418 | +0.0277 to +0.0560 | 8.954e-05 | 0.0004477 |
| structure_3split | MCC | graph_seqss3d_vs_graph_seq3d | -0.0320 | -0.0508 to -0.0132 | 0.003928 | 0.007856 |
| structure_3split | MCC | grassp_ernie_vs_cobra | +0.0752 | +0.0458 to +0.1045 | 0.0002636 | 0.0007908 |
| structure_3split | AUC | grassp_ernie_vs_duet_ensemble | -0.0013 | -0.0084 to +0.0059 | 0.6996 | 0.6996 |
| structure_3split | AUC | grassp_rnafm_vs_duet_ensemble | +0.0137 | +0.0058 to +0.0216 | 0.003491 | 0.01047 |
| structure_3split | AUC | grassp_ernie_vs_graph_seqss3d | +0.0493 | +0.0355 to +0.0630 | 2.032e-05 | 0.0001219 |
| structure_3split | AUC | graph_seq3d_vs_duet_graph | +0.0198 | +0.0129 to +0.0267 | 0.0001132 | 0.0005658 |
| structure_3split | AUC | graph_seqss3d_vs_graph_seq3d | -0.0237 | -0.0368 to -0.0107 | 0.002585 | 0.01034 |
| structure_3split | AUC | grassp_ernie_vs_cobra | +0.0220 | +0.0005 to +0.0435 | 0.046 | 0.09201 |
| structure_3split | AUPRC | grassp_ernie_vs_duet_ensemble | -0.0004 | -0.0129 to +0.0121 | 0.9479 | 0.9479 |
| structure_3split | AUPRC | grassp_rnafm_vs_duet_ensemble | +0.0154 | +0.0029 to +0.0280 | 0.0212 | 0.06891 |
| structure_3split | AUPRC | grassp_ernie_vs_graph_seqss3d | +0.0435 | +0.0270 to +0.0599 | 0.0002074 | 0.001244 |
| structure_3split | AUPRC | graph_seq3d_vs_duet_graph | +0.0136 | +0.0035 to +0.0237 | 0.01378 | 0.06891 |
| structure_3split | AUPRC | graph_seqss3d_vs_graph_seq3d | -0.0218 | -0.0386 to -0.0050 | 0.01643 | 0.06891 |
| structure_3split | AUPRC | grassp_ernie_vs_cobra | +0.0275 | +0.0011 to +0.0539 | 0.04267 | 0.08534 |
| structure_3split | F1 | grassp_ernie_vs_duet_ensemble | +0.0642 | +0.0371 to +0.0913 | 0.0004586 | 0.001723 |
| structure_3split | F1 | grassp_rnafm_vs_duet_ensemble | +0.0746 | +0.0514 to +0.0979 | 4.779e-05 | 0.0002868 |
| structure_3split | F1 | grassp_ernie_vs_graph_seqss3d | +0.0840 | +0.0512 to +0.1168 | 0.0002616 | 0.001308 |
| structure_3split | F1 | graph_seq3d_vs_duet_graph | +0.0536 | +0.0199 to +0.0873 | 0.005738 | 0.01148 |
| structure_3split | F1 | graph_seqss3d_vs_graph_seq3d | -0.0089 | -0.0454 to +0.0276 | 0.5936 | 0.5936 |
| structure_3split | F1 | grassp_ernie_vs_cobra | +0.1214 | +0.0706 to +0.1722 | 0.0004308 | 0.001723 |
| structure_3split | Precision | grassp_ernie_vs_duet_ensemble | -0.0285 | -0.0430 to -0.0139 | 0.001643 | 0.008215 |
| structure_3split | Precision | grassp_rnafm_vs_duet_ensemble | -0.0353 | -0.0500 to -0.0206 | 0.0004237 | 0.002542 |
| structure_3split | Precision | grassp_ernie_vs_graph_seqss3d | +0.0142 | -0.0040 to +0.0324 | 0.1117 | 0.335 |
| structure_3split | Precision | graph_seq3d_vs_duet_graph | +0.0151 | -0.0100 to +0.0402 | 0.207 | 0.414 |
| structure_3split | Precision | graph_seqss3d_vs_graph_seq3d | -0.0323 | -0.0526 to -0.0120 | 0.005699 | 0.0228 |
| structure_3split | Precision | grassp_ernie_vs_cobra | -0.0050 | -0.0329 to +0.0229 | 0.6941 | 0.6941 |
| structure_3split | Recall | grassp_ernie_vs_duet_ensemble | +0.1649 | +0.1089 to +0.2209 | 9.244e-05 | 0.0004622 |
| structure_3split | Recall | grassp_rnafm_vs_duet_ensemble | +0.2054 | +0.1471 to +0.2638 | 2.293e-05 | 0.0001376 |
| structure_3split | Recall | grassp_ernie_vs_graph_seqss3d | +0.1673 | +0.0997 to +0.2349 | 0.0003353 | 0.001006 |
| structure_3split | Recall | graph_seq3d_vs_duet_graph | +0.0828 | +0.0131 to +0.1525 | 0.02491 | 0.04982 |
| structure_3split | Recall | graph_seqss3d_vs_graph_seq3d | +0.0114 | -0.0670 to +0.0898 | 0.7499 | 0.7499 |
| structure_3split | Recall | grassp_ernie_vs_cobra | +0.2290 | +0.1432 to +0.3149 | 0.0001937 | 0.0007749 |
| structure_3split | GMean | grassp_ernie_vs_duet_ensemble | +0.0454 | +0.0232 to +0.0677 | 0.001256 | 0.003767 |
| structure_3split | GMean | grassp_rnafm_vs_duet_ensemble | +0.0492 | +0.0321 to +0.0663 | 0.0001112 | 0.0006672 |
| structure_3split | GMean | grassp_ernie_vs_graph_seqss3d | +0.0628 | +0.0369 to +0.0886 | 0.0003829 | 0.001914 |
| structure_3split | GMean | graph_seq3d_vs_duet_graph | +0.0433 | +0.0147 to +0.0720 | 0.007629 | 0.01526 |
| structure_3split | GMean | graph_seqss3d_vs_graph_seq3d | -0.0077 | -0.0367 to +0.0213 | 0.5643 | 0.5643 |
| structure_3split | GMean | grassp_ernie_vs_cobra | +0.0950 | +0.0528 to +0.1372 | 0.0006524 | 0.00261 |
| structure_3split_fixed_0.5 | MCC | grassp_ernie_vs_duet_ensemble | +0.0384 | +0.0142 to +0.0627 | 0.005925 | 0.005925 |
| structure_3split_fixed_0.5 | MCC | grassp_rnafm_vs_duet_ensemble | +0.0443 | +0.0255 to +0.0632 | 0.0004853 | 0.001456 |
| structure_3split_fixed_0.5 | MCC | grassp_ernie_vs_graph_seqss3d | +0.0768 | +0.0489 to +0.1047 | 0.0001541 | 0.0006163 |
| structure_3split_fixed_0.5 | MCC | graph_seq3d_vs_duet_graph | +0.0400 | +0.0280 to +0.0521 | 3.653e-05 | 0.0002192 |
| structure_3split_fixed_0.5 | MCC | graph_seqss3d_vs_graph_seq3d | -0.0328 | -0.0502 to -0.0154 | 0.002111 | 0.004222 |
| structure_3split_fixed_0.5 | MCC | grassp_ernie_vs_cobra | +0.0915 | +0.0624 to +0.1206 | 5.608e-05 | 0.0002804 |
| structure_3split_fixed_0.5 | AUC | grassp_ernie_vs_duet_ensemble | -0.0013 | -0.0084 to +0.0059 | 0.6996 | 0.6996 |
| structure_3split_fixed_0.5 | AUC | grassp_rnafm_vs_duet_ensemble | +0.0137 | +0.0058 to +0.0216 | 0.003491 | 0.01047 |
| structure_3split_fixed_0.5 | AUC | grassp_ernie_vs_graph_seqss3d | +0.0493 | +0.0355 to +0.0630 | 2.032e-05 | 0.0001219 |
| structure_3split_fixed_0.5 | AUC | graph_seq3d_vs_duet_graph | +0.0198 | +0.0129 to +0.0267 | 0.0001132 | 0.0005658 |
| structure_3split_fixed_0.5 | AUC | graph_seqss3d_vs_graph_seq3d | -0.0237 | -0.0368 to -0.0107 | 0.002585 | 0.01034 |
| structure_3split_fixed_0.5 | AUC | grassp_ernie_vs_cobra | +0.0220 | +0.0005 to +0.0435 | 0.046 | 0.09201 |
| structure_3split_fixed_0.5 | AUPRC | grassp_ernie_vs_duet_ensemble | -0.0004 | -0.0129 to +0.0121 | 0.9479 | 0.9479 |
| structure_3split_fixed_0.5 | AUPRC | grassp_rnafm_vs_duet_ensemble | +0.0154 | +0.0029 to +0.0280 | 0.0212 | 0.06891 |
| structure_3split_fixed_0.5 | AUPRC | grassp_ernie_vs_graph_seqss3d | +0.0435 | +0.0270 to +0.0599 | 0.0002074 | 0.001244 |
| structure_3split_fixed_0.5 | AUPRC | graph_seq3d_vs_duet_graph | +0.0136 | +0.0035 to +0.0237 | 0.01378 | 0.06891 |
| structure_3split_fixed_0.5 | AUPRC | graph_seqss3d_vs_graph_seq3d | -0.0218 | -0.0386 to -0.0050 | 0.01643 | 0.06891 |
| structure_3split_fixed_0.5 | AUPRC | grassp_ernie_vs_cobra | +0.0275 | +0.0011 to +0.0539 | 0.04267 | 0.08534 |
| structure_3split_fixed_0.5 | F1 | grassp_ernie_vs_duet_ensemble | +0.1076 | +0.0720 to +0.1433 | 7.595e-05 | 0.0003798 |
| structure_3split_fixed_0.5 | F1 | grassp_rnafm_vs_duet_ensemble | +0.1173 | +0.0837 to +0.1508 | 2.452e-05 | 0.0001471 |
| structure_3split_fixed_0.5 | F1 | grassp_ernie_vs_graph_seqss3d | +0.0963 | +0.0608 to +0.1317 | 0.0001696 | 0.0005088 |
| structure_3split_fixed_0.5 | F1 | graph_seq3d_vs_duet_graph | +0.0400 | +0.0042 to +0.0758 | 0.03235 | 0.0647 |
| structure_3split_fixed_0.5 | F1 | graph_seqss3d_vs_graph_seq3d | -0.0075 | -0.0488 to +0.0338 | 0.6902 | 0.6902 |
| structure_3split_fixed_0.5 | F1 | grassp_ernie_vs_cobra | +0.1736 | +0.1153 to +0.2318 | 8.465e-05 | 0.0003798 |
| structure_3split_fixed_0.5 | Precision | grassp_ernie_vs_duet_ensemble | -0.0459 | -0.0613 to -0.0305 | 8.571e-05 | 0.0004285 |
| structure_3split_fixed_0.5 | Precision | grassp_rnafm_vs_duet_ensemble | -0.0519 | -0.0674 to -0.0364 | 3.425e-05 | 0.0002055 |
| structure_3split_fixed_0.5 | Precision | grassp_ernie_vs_graph_seqss3d | +0.0098 | -0.0075 to +0.0271 | 0.2325 | 0.465 |
| structure_3split_fixed_0.5 | Precision | graph_seq3d_vs_duet_graph | +0.0229 | -0.0008 to +0.0467 | 0.05698 | 0.1709 |
| structure_3split_fixed_0.5 | Precision | graph_seqss3d_vs_graph_seq3d | -0.0358 | -0.0541 to -0.0175 | 0.001648 | 0.006594 |
| structure_3split_fixed_0.5 | Precision | grassp_ernie_vs_cobra | -0.0174 | -0.0502 to +0.0154 | 0.2613 | 0.465 |
| structure_3split_fixed_0.5 | Recall | grassp_ernie_vs_duet_ensemble | +0.2466 | +0.1910 to +0.3023 | 3.518e-06 | 1.759e-05 |
| structure_3split_fixed_0.5 | Recall | grassp_rnafm_vs_duet_ensemble | +0.2831 | +0.2242 to +0.3420 | 1.778e-06 | 1.067e-05 |
| structure_3split_fixed_0.5 | Recall | grassp_ernie_vs_graph_seqss3d | +0.1945 | +0.1275 to +0.2615 | 0.0001031 | 0.0003094 |
| structure_3split_fixed_0.5 | Recall | graph_seq3d_vs_duet_graph | +0.0540 | -0.0161 to +0.1241 | 0.1155 | 0.231 |
| structure_3split_fixed_0.5 | Recall | graph_seqss3d_vs_graph_seq3d | +0.0129 | -0.0682 to +0.0940 | 0.7264 | 0.7264 |
| structure_3split_fixed_0.5 | Recall | grassp_ernie_vs_cobra | +0.3087 | +0.2298 to +0.3877 | 9.841e-06 | 3.936e-05 |
| structure_3split_fixed_0.5 | GMean | grassp_ernie_vs_duet_ensemble | +0.0821 | +0.0512 to +0.1129 | 0.0001974 | 0.0008935 |
| structure_3split_fixed_0.5 | GMean | grassp_rnafm_vs_duet_ensemble | +0.0858 | +0.0584 to +0.1132 | 5.821e-05 | 0.0003493 |
| structure_3split_fixed_0.5 | GMean | grassp_ernie_vs_graph_seqss3d | +0.0720 | +0.0433 to +0.1006 | 0.0003027 | 0.000908 |
| structure_3split_fixed_0.5 | GMean | graph_seq3d_vs_duet_graph | +0.0321 | +0.0012 to +0.0630 | 0.04311 | 0.08623 |
| structure_3split_fixed_0.5 | GMean | graph_seqss3d_vs_graph_seq3d | -0.0056 | -0.0396 to +0.0284 | 0.718 | 0.718 |
| structure_3split_fixed_0.5 | GMean | grassp_ernie_vs_cobra | +0.1400 | +0.0881 to +0.1919 | 0.0001787 | 0.0008935 |
| crossed_controls_3split | MCC | grassp_ernie_vs_grassp_rnafm | -0.0062 | -0.0206 to +0.0082 | 0.3562 | 0.3562 |
| crossed_controls_3split | MCC | rgcn_ernie_vs_graph_seq3d | -0.0513 | -0.0813 to -0.0214 | 0.003732 | 0.007463 |
| crossed_controls_3split | MCC | grassp_rnafm_vs_graph_seq3d | +0.0484 | +0.0290 to +0.0678 | 0.0003181 | 0.0009543 |
| crossed_controls_3split | MCC | grassp_ernie_vs_rgcn_ernie | +0.0935 | +0.0665 to +0.1205 | 2.597e-05 | 0.0001039 |
| crossed_controls_3split | AUC | grassp_ernie_vs_grassp_rnafm | -0.0150 | -0.0202 to -0.0097 | 0.0001197 | 0.0002393 |
| crossed_controls_3split | AUC | rgcn_ernie_vs_graph_seq3d | -0.0443 | -0.0640 to -0.0246 | 0.0006559 | 0.0006559 |
| crossed_controls_3split | AUC | grassp_rnafm_vs_graph_seq3d | +0.0405 | +0.0316 to +0.0494 | 2.914e-06 | 1.165e-05 |
| crossed_controls_3split | AUC | grassp_ernie_vs_rgcn_ernie | +0.0698 | +0.0524 to +0.0873 | 8.041e-06 | 2.412e-05 |
| crossed_controls_3split | AUPRC | grassp_ernie_vs_grassp_rnafm | -0.0158 | -0.0291 to -0.0025 | 0.0247 | 0.04939 |
| crossed_controls_3split | AUPRC | rgcn_ernie_vs_graph_seq3d | -0.0214 | -0.0402 to -0.0027 | 0.02958 | 0.04939 |
| crossed_controls_3split | AUPRC | grassp_rnafm_vs_graph_seq3d | +0.0374 | +0.0227 to +0.0522 | 0.0002833 | 0.001133 |
| crossed_controls_3split | AUPRC | grassp_ernie_vs_rgcn_ernie | +0.0430 | +0.0219 to +0.0642 | 0.001282 | 0.003845 |
| crossed_controls_3split | F1 | grassp_ernie_vs_grassp_rnafm | -0.0104 | -0.0196 to -0.0013 | 0.02981 | 0.04795 |
| crossed_controls_3split | F1 | rgcn_ernie_vs_graph_seq3d | -0.0397 | -0.0729 to -0.0066 | 0.02398 | 0.04795 |
| crossed_controls_3split | F1 | grassp_rnafm_vs_graph_seq3d | +0.0856 | +0.0604 to +0.1108 | 3.062e-05 | 9.187e-05 |
| crossed_controls_3split | F1 | grassp_ernie_vs_rgcn_ernie | +0.1149 | +0.0896 to +0.1401 | 2.845e-06 | 1.138e-05 |
| crossed_controls_3split | Precision | grassp_ernie_vs_grassp_rnafm | +0.0068 | -0.0058 to +0.0194 | 0.2512 | 0.2512 |
| crossed_controls_3split | Precision | rgcn_ernie_vs_graph_seq3d | -0.0345 | -0.0652 to -0.0039 | 0.03123 | 0.0937 |
| crossed_controls_3split | Precision | grassp_rnafm_vs_graph_seq3d | -0.0250 | -0.0440 to -0.0060 | 0.01566 | 0.06266 |
| crossed_controls_3split | Precision | grassp_ernie_vs_rgcn_ernie | +0.0164 | -0.0049 to +0.0377 | 0.1159 | 0.2317 |
| crossed_controls_3split | Recall | grassp_ernie_vs_grassp_rnafm | -0.0405 | -0.0707 to -0.0103 | 0.01408 | 0.02817 |
| crossed_controls_3split | Recall | rgcn_ernie_vs_graph_seq3d | -0.0489 | -0.1140 to +0.0161 | 0.1229 | 0.1229 |
| crossed_controls_3split | Recall | grassp_rnafm_vs_graph_seq3d | +0.2192 | +0.1678 to +0.2706 | 4.801e-06 | 1.44e-05 |
| crossed_controls_3split | Recall | grassp_ernie_vs_rgcn_ernie | +0.2276 | +0.1791 to +0.2760 | 2.149e-06 | 8.598e-06 |
| crossed_controls_3split | GMean | grassp_ernie_vs_grassp_rnafm | -0.0038 | -0.0116 to +0.0041 | 0.3053 | 0.3053 |
| crossed_controls_3split | GMean | rgcn_ernie_vs_graph_seq3d | -0.0311 | -0.0586 to -0.0036 | 0.03068 | 0.06135 |
| crossed_controls_3split | GMean | grassp_rnafm_vs_graph_seq3d | +0.0589 | +0.0374 to +0.0803 | 0.0001581 | 0.0004742 |
| crossed_controls_3split | GMean | grassp_ernie_vs_rgcn_ernie | +0.0862 | +0.0658 to +0.1066 | 5.257e-06 | 2.103e-05 |
| external_independent_2split | MCC | duet_ensemble_vs_cobra | +0.1093 | +0.0431 to +0.1754 | 0.00466 | 0.01398 |
| external_independent_2split | MCC | grassp_ernie_vs_duet_ensemble | -0.1195 | -0.1720 to -0.0671 | 0.0005997 | 0.002999 |
| external_independent_2split | MCC | grassp_rnafm_vs_duet_ensemble | -0.0027 | -0.0376 to +0.0321 | 0.8644 | 0.9523 |
| external_independent_2split | MCC | grassp_rnafm_vs_cobra | +0.1066 | +0.0551 to +0.1580 | 0.00115 | 0.004599 |
| external_independent_2split | MCC | graph_seqss3d_vs_graph_seq3d_external | -0.0124 | -0.0501 to +0.0253 | 0.4761 | 0.9523 |
| external_independent_2split | AUC | duet_ensemble_vs_cobra | +0.0729 | +0.0311 to +0.1146 | 0.003362 | 0.01009 |
| external_independent_2split | AUC | grassp_ernie_vs_duet_ensemble | -0.1390 | -0.1662 to -0.1119 | 1.05e-06 | 5.248e-06 |
| external_independent_2split | AUC | grassp_rnafm_vs_duet_ensemble | +0.0101 | -0.0035 to +0.0238 | 0.128 | 0.256 |
| external_independent_2split | AUC | grassp_rnafm_vs_cobra | +0.0830 | +0.0451 to +0.1209 | 0.0007897 | 0.003159 |
| external_independent_2split | AUC | graph_seqss3d_vs_graph_seq3d_external | -0.0184 | -0.0443 to +0.0075 | 0.1431 | 0.256 |
| external_independent_2split | AUPRC | duet_ensemble_vs_cobra | +0.0627 | +0.0327 to +0.0928 | 0.001093 | 0.003279 |
| external_independent_2split | AUPRC | grassp_ernie_vs_duet_ensemble | -0.1179 | -0.1334 to -0.1024 | 3.336e-08 | 1.668e-07 |
| external_independent_2split | AUPRC | grassp_rnafm_vs_duet_ensemble | +0.0394 | +0.0099 to +0.0689 | 0.01454 | 0.02908 |
| external_independent_2split | AUPRC | grassp_rnafm_vs_cobra | +0.1021 | +0.0692 to +0.1351 | 6.244e-05 | 0.0002498 |
| external_independent_2split | AUPRC | graph_seqss3d_vs_graph_seq3d_external | -0.0026 | -0.0459 to +0.0407 | 0.8957 | 0.8957 |
| external_independent_2split | F1 | duet_ensemble_vs_cobra | +0.0801 | +0.0357 to +0.1246 | 0.002774 | 0.0111 |
| external_independent_2split | F1 | grassp_ernie_vs_duet_ensemble | -0.0739 | -0.1034 to -0.0445 | 0.0003041 | 0.001521 |
| external_independent_2split | F1 | grassp_rnafm_vs_duet_ensemble | -0.0091 | -0.0313 to +0.0132 | 0.3799 | 0.7597 |
| external_independent_2split | F1 | grassp_rnafm_vs_cobra | +0.0710 | +0.0302 to +0.1118 | 0.003416 | 0.0111 |
| external_independent_2split | F1 | graph_seqss3d_vs_graph_seq3d_external | -0.0104 | -0.0362 to +0.0154 | 0.3863 | 0.7597 |
| external_independent_2split | Precision | duet_ensemble_vs_cobra | +0.0522 | +0.0255 to +0.0790 | 0.001687 | 0.006749 |
| external_independent_2split | Precision | grassp_ernie_vs_duet_ensemble | -0.0636 | -0.0838 to -0.0435 | 5.396e-05 | 0.0002698 |
| external_independent_2split | Precision | grassp_rnafm_vs_duet_ensemble | -0.0233 | -0.0456 to -0.0010 | 0.04267 | 0.08534 |
| external_independent_2split | Precision | grassp_rnafm_vs_cobra | +0.0290 | +0.0091 to +0.0489 | 0.009301 | 0.0279 |
| external_independent_2split | Precision | graph_seqss3d_vs_graph_seq3d_external | -0.0149 | -0.0378 to +0.0079 | 0.1737 | 0.1737 |
| external_independent_2split | Recall | duet_ensemble_vs_cobra | +0.1349 | -0.0354 to +0.3052 | 0.1067 | 0.32 |
| external_independent_2split | Recall | grassp_ernie_vs_duet_ensemble | +0.0073 | -0.1131 to +0.1276 | 0.8945 | 1 |
| external_independent_2split | Recall | grassp_rnafm_vs_duet_ensemble | +0.1316 | +0.0510 to +0.2121 | 0.004958 | 0.01983 |
| external_independent_2split | Recall | grassp_rnafm_vs_cobra | +0.2665 | +0.1347 to +0.3983 | 0.001341 | 0.006706 |
| external_independent_2split | Recall | graph_seqss3d_vs_graph_seq3d_external | +0.0229 | -0.0825 to +0.1284 | 0.6345 | 1 |
| external_independent_2split | GMean | duet_ensemble_vs_cobra | +0.1042 | +0.0361 to +0.1722 | 0.007126 | 0.02138 |
| external_independent_2split | GMean | grassp_ernie_vs_duet_ensemble | -0.0688 | -0.1064 to -0.0311 | 0.002544 | 0.01018 |
| external_independent_2split | GMean | grassp_rnafm_vs_duet_ensemble | +0.0171 | -0.0054 to +0.0397 | 0.12 | 0.2399 |
| external_independent_2split | GMean | grassp_rnafm_vs_cobra | +0.1213 | +0.0591 to +0.1835 | 0.00169 | 0.00845 |
| external_independent_2split | GMean | graph_seqss3d_vs_graph_seq3d_external | +0.0010 | -0.0365 to +0.0386 | 0.9515 | 0.9515 |

## Chain bootstrap

| Family | Metric | Comparison | Delta | 95% CI | p | Holm p | valid B |
|:--|:--|:--|--:|:--|--:|--:|--:|
| primary_4split | MCC | duet_ensemble_vs_cobra | +0.0474 | +0.0200 to +0.0774 | 0.001998 | 0.05195 | 1000 |
| primary_4split | AUC | duet_ensemble_vs_cobra | +0.0269 | +0.0102 to +0.0477 | 0.001998 | 0.05195 | 1000 |
| primary_4split | AUPRC | duet_ensemble_vs_cobra | +0.0348 | +0.0178 to +0.0520 | 0.001998 | 0.05195 | 1000 |
| primary_4split | F1 | duet_ensemble_vs_cobra | +0.0502 | +0.0311 to +0.0729 | 0.001998 | 0.05195 | 1000 |
| primary_4split | Precision | duet_ensemble_vs_cobra | +0.0211 | -0.0007 to +0.0408 | 0.06394 | 1 | 1000 |
| primary_4split | Recall | duet_ensemble_vs_cobra | +0.0587 | +0.0362 to +0.0835 | 0.001998 | 0.05195 | 1000 |
| primary_4split | GMean | duet_ensemble_vs_cobra | +0.0464 | +0.0294 to +0.0711 | 0.001998 | 0.05195 | 1000 |
| primary_4split | MCC | duet_graph_vs_cobra | +0.0007 | -0.0708 to +0.0648 | 0.999 | 1 | 1000 |
| primary_4split | AUC | duet_graph_vs_cobra | -0.0162 | -0.0576 to +0.0287 | 0.4695 | 1 | 1000 |
| primary_4split | AUPRC | duet_graph_vs_cobra | -0.0008 | -0.0501 to +0.0390 | 0.8412 | 1 | 1000 |
| primary_4split | F1 | duet_graph_vs_cobra | -0.0036 | -0.0536 to +0.0504 | 0.9231 | 1 | 1000 |
| primary_4split | Precision | duet_graph_vs_cobra | +0.0108 | -0.0434 to +0.0566 | 0.7053 | 1 | 1000 |
| primary_4split | Recall | duet_graph_vs_cobra | -0.0585 | -0.1127 to -0.0031 | 0.03397 | 0.2398 | 1000 |
| primary_4split | GMean | duet_graph_vs_cobra | +0.0019 | -0.0409 to +0.0497 | 0.8751 | 1 | 1000 |
| primary_4split | MCC | duet_tcn_vs_cobra | +0.0156 | -0.0104 to +0.0428 | 0.2737 | 1 | 1000 |
| primary_4split | AUC | duet_tcn_vs_cobra | +0.0021 | -0.0135 to +0.0177 | 0.7732 | 1 | 1000 |
| primary_4split | AUPRC | duet_tcn_vs_cobra | -0.0012 | -0.0191 to +0.0220 | 0.991 | 1 | 1000 |
| primary_4split | F1 | duet_tcn_vs_cobra | +0.0128 | -0.0066 to +0.0336 | 0.1958 | 1 | 1000 |
| primary_4split | Precision | duet_tcn_vs_cobra | +0.0116 | -0.0094 to +0.0297 | 0.2917 | 1 | 1000 |
| primary_4split | Recall | duet_tcn_vs_cobra | -0.0045 | -0.0274 to +0.0179 | 0.7133 | 1 | 1000 |
| primary_4split | GMean | duet_tcn_vs_cobra | +0.0152 | -0.0018 to +0.0376 | 0.08392 | 0.6713 | 1000 |
| primary_4split | MCC | duet_dense_vs_cobra | +0.0211 | -0.0068 to +0.0493 | 0.1419 | 1 | 1000 |
| primary_4split | AUC | duet_dense_vs_cobra | +0.0091 | -0.0070 to +0.0250 | 0.2837 | 1 | 1000 |
| primary_4split | AUPRC | duet_dense_vs_cobra | +0.0029 | -0.0171 to +0.0228 | 0.8132 | 1 | 1000 |
| primary_4split | F1 | duet_dense_vs_cobra | -0.0030 | -0.0250 to +0.0187 | 0.7612 | 1 | 1000 |
| primary_4split | Precision | duet_dense_vs_cobra | +0.0293 | +0.0047 to +0.0498 | 0.01399 | 0.3497 | 1000 |
| primary_4split | Recall | duet_dense_vs_cobra | -0.0433 | -0.0651 to -0.0205 | 0.001998 | 0.05195 | 1000 |
| primary_4split | GMean | duet_dense_vs_cobra | +0.0027 | -0.0165 to +0.0261 | 0.8052 | 1 | 1000 |
| primary_4split | MCC | duet_ensemble_vs_duet_graph | +0.0468 | -0.0088 to +0.1094 | 0.1139 | 1 | 1000 |
| primary_4split | AUC | duet_ensemble_vs_duet_graph | +0.0431 | +0.0101 to +0.0788 | 0.003996 | 0.08791 | 1000 |
| primary_4split | AUPRC | duet_ensemble_vs_duet_graph | +0.0356 | +0.0012 to +0.0803 | 0.04196 | 0.8392 | 1000 |
| primary_4split | F1 | duet_ensemble_vs_duet_graph | +0.0538 | +0.0089 to +0.0996 | 0.02198 | 0.2198 | 1000 |
| primary_4split | Precision | duet_ensemble_vs_duet_graph | +0.0103 | -0.0294 to +0.0541 | 0.6074 | 1 | 1000 |
| primary_4split | Recall | duet_ensemble_vs_duet_graph | +0.1171 | +0.0654 to +0.1739 | 0.001998 | 0.05195 | 1000 |
| primary_4split | GMean | duet_ensemble_vs_duet_graph | +0.0445 | +0.0072 to +0.0836 | 0.02398 | 0.2398 | 1000 |
| primary_4split_fixed_0.5 | MCC | duet_ensemble_vs_cobra | +0.0524 | +0.0238 to +0.0826 | 0.001998 | 0.05195 | 1000 |
| primary_4split_fixed_0.5 | AUC | duet_ensemble_vs_cobra | +0.0269 | +0.0102 to +0.0477 | 0.001998 | 0.05195 | 1000 |
| primary_4split_fixed_0.5 | AUPRC | duet_ensemble_vs_cobra | +0.0348 | +0.0178 to +0.0520 | 0.001998 | 0.05195 | 1000 |
| primary_4split_fixed_0.5 | F1 | duet_ensemble_vs_cobra | +0.0589 | +0.0361 to +0.0844 | 0.001998 | 0.05195 | 1000 |
| primary_4split_fixed_0.5 | Precision | duet_ensemble_vs_cobra | +0.0257 | -0.0039 to +0.0490 | 0.08991 | 1 | 1000 |
| primary_4split_fixed_0.5 | Recall | duet_ensemble_vs_cobra | +0.0600 | +0.0351 to +0.0872 | 0.001998 | 0.05195 | 1000 |
| primary_4split_fixed_0.5 | GMean | duet_ensemble_vs_cobra | +0.0538 | +0.0341 to +0.0777 | 0.001998 | 0.05195 | 1000 |
| primary_4split_fixed_0.5 | MCC | duet_graph_vs_cobra | +0.0091 | -0.0558 to +0.0664 | 0.7852 | 1 | 1000 |
| primary_4split_fixed_0.5 | AUC | duet_graph_vs_cobra | -0.0162 | -0.0576 to +0.0287 | 0.4695 | 1 | 1000 |
| primary_4split_fixed_0.5 | AUPRC | duet_graph_vs_cobra | -0.0008 | -0.0501 to +0.0390 | 0.8412 | 1 | 1000 |
| primary_4split_fixed_0.5 | F1 | duet_graph_vs_cobra | +0.0348 | -0.0143 to +0.0860 | 0.1459 | 1 | 1000 |
| primary_4split_fixed_0.5 | Precision | duet_graph_vs_cobra | -0.0047 | -0.0598 to +0.0422 | 0.7532 | 1 | 1000 |
| primary_4split_fixed_0.5 | Recall | duet_graph_vs_cobra | +0.0172 | -0.0314 to +0.0675 | 0.4555 | 1 | 1000 |
| primary_4split_fixed_0.5 | GMean | duet_graph_vs_cobra | +0.0331 | -0.0072 to +0.0767 | 0.09391 | 0.6713 | 1000 |
| primary_4split_fixed_0.5 | MCC | duet_tcn_vs_cobra | +0.0240 | -0.0012 to +0.0492 | 0.06993 | 1 | 1000 |
| primary_4split_fixed_0.5 | AUC | duet_tcn_vs_cobra | +0.0021 | -0.0135 to +0.0177 | 0.7732 | 1 | 1000 |
| primary_4split_fixed_0.5 | AUPRC | duet_tcn_vs_cobra | -0.0012 | -0.0191 to +0.0220 | 0.991 | 1 | 1000 |
| primary_4split_fixed_0.5 | F1 | duet_tcn_vs_cobra | +0.0511 | +0.0305 to +0.0735 | 0.001998 | 0.05195 | 1000 |
| primary_4split_fixed_0.5 | Precision | duet_tcn_vs_cobra | -0.0038 | -0.0330 to +0.0165 | 0.6274 | 1 | 1000 |
| primary_4split_fixed_0.5 | Recall | duet_tcn_vs_cobra | +0.0711 | +0.0463 to +0.0947 | 0.001998 | 0.05195 | 1000 |
| primary_4split_fixed_0.5 | GMean | duet_tcn_vs_cobra | +0.0465 | +0.0295 to +0.0668 | 0.001998 | 0.05195 | 1000 |
| primary_4split_fixed_0.5 | MCC | duet_dense_vs_cobra | +0.0322 | +0.0089 to +0.0570 | 0.00999 | 0.2198 | 1000 |
| primary_4split_fixed_0.5 | AUC | duet_dense_vs_cobra | +0.0091 | -0.0070 to +0.0250 | 0.2837 | 1 | 1000 |
| primary_4split_fixed_0.5 | AUPRC | duet_dense_vs_cobra | +0.0029 | -0.0171 to +0.0228 | 0.8132 | 1 | 1000 |
| primary_4split_fixed_0.5 | F1 | duet_dense_vs_cobra | +0.0414 | +0.0213 to +0.0634 | 0.001998 | 0.05195 | 1000 |
| primary_4split_fixed_0.5 | Precision | duet_dense_vs_cobra | +0.0131 | -0.0159 to +0.0332 | 0.3836 | 1 | 1000 |
| primary_4split_fixed_0.5 | Recall | duet_dense_vs_cobra | +0.0428 | +0.0191 to +0.0699 | 0.001998 | 0.05195 | 1000 |
| primary_4split_fixed_0.5 | GMean | duet_dense_vs_cobra | +0.0390 | +0.0218 to +0.0593 | 0.001998 | 0.05195 | 1000 |
| primary_4split_fixed_0.5 | MCC | duet_ensemble_vs_duet_graph | +0.0434 | -0.0132 to +0.1069 | 0.1299 | 1 | 1000 |
| primary_4split_fixed_0.5 | AUC | duet_ensemble_vs_duet_graph | +0.0431 | +0.0101 to +0.0788 | 0.003996 | 0.08791 | 1000 |
| primary_4split_fixed_0.5 | AUPRC | duet_ensemble_vs_duet_graph | +0.0356 | +0.0012 to +0.0803 | 0.04196 | 0.8392 | 1000 |
| primary_4split_fixed_0.5 | F1 | duet_ensemble_vs_duet_graph | +0.0241 | -0.0232 to +0.0703 | 0.3217 | 1 | 1000 |
| primary_4split_fixed_0.5 | Precision | duet_ensemble_vs_duet_graph | +0.0304 | -0.0126 to +0.0779 | 0.1658 | 1 | 1000 |
| primary_4split_fixed_0.5 | Recall | duet_ensemble_vs_duet_graph | +0.0428 | -0.0061 to +0.0955 | 0.08991 | 0.4496 | 1000 |
| primary_4split_fixed_0.5 | GMean | duet_ensemble_vs_duet_graph | +0.0206 | -0.0186 to +0.0594 | 0.3057 | 1 | 1000 |
| structure_3split | MCC | grassp_ernie_vs_duet_ensemble | +0.0270 | -0.0316 to +0.0965 | 0.4276 | 1 | 1000 |
| structure_3split | AUC | grassp_ernie_vs_duet_ensemble | -0.0013 | -0.0353 to +0.0399 | 0.961 | 1 | 1000 |
| structure_3split | AUPRC | grassp_ernie_vs_duet_ensemble | -0.0004 | -0.0363 to +0.0398 | 0.967 | 1 | 1000 |
| structure_3split | F1 | grassp_ernie_vs_duet_ensemble | +0.0642 | +0.0250 to +0.1111 | 0.001998 | 0.05195 | 1000 |
| structure_3split | Precision | grassp_ernie_vs_duet_ensemble | -0.0285 | -0.0721 to +0.0172 | 0.2198 | 1 | 1000 |
| structure_3split | Recall | grassp_ernie_vs_duet_ensemble | +0.1649 | +0.1075 to +0.2241 | 0.001998 | 0.05195 | 1000 |
| structure_3split | GMean | grassp_ernie_vs_duet_ensemble | +0.0454 | +0.0106 to +0.0845 | 0.005994 | 0.07193 | 1000 |
| structure_3split | MCC | grassp_rnafm_vs_duet_ensemble | +0.0332 | -0.0295 to +0.1046 | 0.2857 | 1 | 1000 |
| structure_3split | AUC | grassp_rnafm_vs_duet_ensemble | +0.0137 | -0.0174 to +0.0524 | 0.3576 | 1 | 1000 |
| structure_3split | AUPRC | grassp_rnafm_vs_duet_ensemble | +0.0154 | -0.0232 to +0.0582 | 0.4496 | 1 | 1000 |
| structure_3split | F1 | grassp_rnafm_vs_duet_ensemble | +0.0746 | +0.0264 to +0.1285 | 0.003996 | 0.05195 | 1000 |
| structure_3split | Precision | grassp_rnafm_vs_duet_ensemble | -0.0353 | -0.0729 to +0.0046 | 0.08991 | 1 | 1000 |
| structure_3split | Recall | grassp_rnafm_vs_duet_ensemble | +0.2054 | +0.1479 to +0.2690 | 0.001998 | 0.05195 | 1000 |
| structure_3split | GMean | grassp_rnafm_vs_duet_ensemble | +0.0492 | +0.0105 to +0.0964 | 0.01798 | 0.1978 | 1000 |
| structure_3split | MCC | grassp_ernie_vs_graph_seqss3d | +0.0741 | +0.0105 to +0.1363 | 0.03197 | 0.5435 | 1000 |
| structure_3split | AUC | grassp_ernie_vs_graph_seqss3d | +0.0493 | +0.0116 to +0.0859 | 0.01399 | 0.2797 | 1000 |
| structure_3split | AUPRC | grassp_ernie_vs_graph_seqss3d | +0.0435 | +0.0042 to +0.0804 | 0.02797 | 0.6154 | 1000 |
| structure_3split | F1 | grassp_ernie_vs_graph_seqss3d | +0.0840 | +0.0462 to +0.1201 | 0.001998 | 0.05195 | 1000 |
| structure_3split | Precision | grassp_ernie_vs_graph_seqss3d | +0.0142 | -0.0286 to +0.0534 | 0.5275 | 1 | 1000 |
| structure_3split | Recall | grassp_ernie_vs_graph_seqss3d | +0.1673 | +0.1236 to +0.2141 | 0.001998 | 0.05195 | 1000 |
| structure_3split | GMean | grassp_ernie_vs_graph_seqss3d | +0.0628 | +0.0308 to +0.0943 | 0.001998 | 0.05195 | 1000 |
| structure_3split | MCC | graph_seq3d_vs_duet_graph | +0.0418 | +0.0064 to +0.0758 | 0.02198 | 0.4396 | 1000 |
| structure_3split | AUC | graph_seq3d_vs_duet_graph | +0.0198 | +0.0002 to +0.0381 | 0.04995 | 0.7493 | 1000 |
| structure_3split | AUPRC | graph_seq3d_vs_duet_graph | +0.0136 | -0.0126 to +0.0371 | 0.3297 | 1 | 1000 |
| structure_3split | F1 | graph_seq3d_vs_duet_graph | +0.0536 | +0.0283 to +0.0772 | 0.001998 | 0.05195 | 1000 |
| structure_3split | Precision | graph_seq3d_vs_duet_graph | +0.0151 | -0.0131 to +0.0406 | 0.3257 | 1 | 1000 |
| structure_3split | Recall | graph_seq3d_vs_duet_graph | +0.0828 | +0.0571 to +0.1074 | 0.001998 | 0.05195 | 1000 |
| structure_3split | GMean | graph_seq3d_vs_duet_graph | +0.0433 | +0.0236 to +0.0622 | 0.001998 | 0.05195 | 1000 |
| structure_3split | MCC | graph_seqss3d_vs_graph_seq3d | -0.0320 | -0.0650 to +0.0063 | 0.0999 | 1 | 1000 |
| structure_3split | AUC | graph_seqss3d_vs_graph_seq3d | -0.0237 | -0.0439 to -0.0022 | 0.02797 | 0.4755 | 1000 |
| structure_3split | AUPRC | graph_seqss3d_vs_graph_seq3d | -0.0218 | -0.0386 to -0.0006 | 0.04396 | 0.8392 | 1000 |
| structure_3split | F1 | graph_seqss3d_vs_graph_seq3d | -0.0089 | -0.0314 to +0.0158 | 0.4735 | 1 | 1000 |
| structure_3split | Precision | graph_seqss3d_vs_graph_seq3d | -0.0323 | -0.0550 to -0.0040 | 0.03197 | 0.7353 | 1000 |
| structure_3split | Recall | graph_seqss3d_vs_graph_seq3d | +0.0114 | -0.0136 to +0.0367 | 0.3516 | 1 | 1000 |
| structure_3split | GMean | graph_seqss3d_vs_graph_seq3d | -0.0077 | -0.0269 to +0.0138 | 0.4935 | 1 | 1000 |
| structure_3split | MCC | grassp_ernie_vs_cobra | +0.0752 | +0.0159 to +0.1420 | 0.01798 | 0.3776 | 1000 |
| structure_3split | AUC | grassp_ernie_vs_cobra | +0.0220 | -0.0170 to +0.0687 | 0.2797 | 1 | 1000 |
| structure_3split | AUPRC | grassp_ernie_vs_cobra | +0.0275 | -0.0140 to +0.0729 | 0.2198 | 1 | 1000 |
| structure_3split | F1 | grassp_ernie_vs_cobra | +0.1214 | +0.0765 to +0.1717 | 0.001998 | 0.05195 | 1000 |
| structure_3split | Precision | grassp_ernie_vs_cobra | -0.0050 | -0.0542 to +0.0414 | 0.7293 | 1 | 1000 |
| structure_3split | Recall | grassp_ernie_vs_cobra | +0.2290 | +0.1690 to +0.2910 | 0.001998 | 0.05195 | 1000 |
| structure_3split | GMean | grassp_ernie_vs_cobra | +0.0950 | +0.0625 to +0.1394 | 0.001998 | 0.05195 | 1000 |
| structure_3split_fixed_0.5 | MCC | grassp_ernie_vs_duet_ensemble | +0.0384 | -0.0211 to +0.1050 | 0.2238 | 1 | 1000 |
| structure_3split_fixed_0.5 | AUC | grassp_ernie_vs_duet_ensemble | -0.0013 | -0.0353 to +0.0399 | 0.961 | 1 | 1000 |
| structure_3split_fixed_0.5 | AUPRC | grassp_ernie_vs_duet_ensemble | -0.0004 | -0.0363 to +0.0398 | 0.967 | 1 | 1000 |
| structure_3split_fixed_0.5 | F1 | grassp_ernie_vs_duet_ensemble | +0.1076 | +0.0580 to +0.1604 | 0.001998 | 0.05195 | 1000 |
| structure_3split_fixed_0.5 | Precision | grassp_ernie_vs_duet_ensemble | -0.0459 | -0.0977 to +0.0034 | 0.06793 | 1 | 1000 |
| structure_3split_fixed_0.5 | Recall | grassp_ernie_vs_duet_ensemble | +0.2466 | +0.1872 to +0.3057 | 0.001998 | 0.05195 | 1000 |
| structure_3split_fixed_0.5 | GMean | grassp_ernie_vs_duet_ensemble | +0.0821 | +0.0438 to +0.1244 | 0.001998 | 0.05195 | 1000 |
| structure_3split_fixed_0.5 | MCC | grassp_rnafm_vs_duet_ensemble | +0.0443 | -0.0184 to +0.1162 | 0.1658 | 1 | 1000 |
| structure_3split_fixed_0.5 | AUC | grassp_rnafm_vs_duet_ensemble | +0.0137 | -0.0174 to +0.0524 | 0.3576 | 1 | 1000 |
| structure_3split_fixed_0.5 | AUPRC | grassp_rnafm_vs_duet_ensemble | +0.0154 | -0.0232 to +0.0582 | 0.4496 | 1 | 1000 |
| structure_3split_fixed_0.5 | F1 | grassp_rnafm_vs_duet_ensemble | +0.1173 | +0.0634 to +0.1752 | 0.001998 | 0.05195 | 1000 |
| structure_3split_fixed_0.5 | Precision | grassp_rnafm_vs_duet_ensemble | -0.0519 | -0.0957 to -0.0075 | 0.02797 | 0.6713 | 1000 |
| structure_3split_fixed_0.5 | Recall | grassp_rnafm_vs_duet_ensemble | +0.2831 | +0.2262 to +0.3451 | 0.001998 | 0.05195 | 1000 |
| structure_3split_fixed_0.5 | GMean | grassp_rnafm_vs_duet_ensemble | +0.0858 | +0.0416 to +0.1349 | 0.001998 | 0.05195 | 1000 |
| structure_3split_fixed_0.5 | MCC | grassp_ernie_vs_graph_seqss3d | +0.0768 | +0.0144 to +0.1380 | 0.02797 | 0.5035 | 1000 |
| structure_3split_fixed_0.5 | AUC | grassp_ernie_vs_graph_seqss3d | +0.0493 | +0.0116 to +0.0859 | 0.01399 | 0.2797 | 1000 |
| structure_3split_fixed_0.5 | AUPRC | grassp_ernie_vs_graph_seqss3d | +0.0435 | +0.0042 to +0.0804 | 0.02797 | 0.6154 | 1000 |
| structure_3split_fixed_0.5 | F1 | grassp_ernie_vs_graph_seqss3d | +0.0963 | +0.0576 to +0.1343 | 0.001998 | 0.05195 | 1000 |
| structure_3split_fixed_0.5 | Precision | grassp_ernie_vs_graph_seqss3d | +0.0098 | -0.0350 to +0.0497 | 0.6813 | 1 | 1000 |
| structure_3split_fixed_0.5 | Recall | grassp_ernie_vs_graph_seqss3d | +0.1945 | +0.1493 to +0.2425 | 0.001998 | 0.05195 | 1000 |
| structure_3split_fixed_0.5 | GMean | grassp_ernie_vs_graph_seqss3d | +0.0720 | +0.0398 to +0.1042 | 0.001998 | 0.05195 | 1000 |
| structure_3split_fixed_0.5 | MCC | graph_seq3d_vs_duet_graph | +0.0400 | +0.0060 to +0.0739 | 0.02398 | 0.4555 | 1000 |
| structure_3split_fixed_0.5 | AUC | graph_seq3d_vs_duet_graph | +0.0198 | +0.0002 to +0.0381 | 0.04995 | 0.7493 | 1000 |
| structure_3split_fixed_0.5 | AUPRC | graph_seq3d_vs_duet_graph | +0.0136 | -0.0126 to +0.0371 | 0.3297 | 1 | 1000 |
| structure_3split_fixed_0.5 | F1 | graph_seq3d_vs_duet_graph | +0.0400 | +0.0166 to +0.0629 | 0.003996 | 0.05195 | 1000 |
| structure_3split_fixed_0.5 | Precision | graph_seq3d_vs_duet_graph | +0.0229 | -0.0056 to +0.0494 | 0.1259 | 1 | 1000 |
| structure_3split_fixed_0.5 | Recall | graph_seq3d_vs_duet_graph | +0.0540 | +0.0291 to +0.0775 | 0.001998 | 0.05195 | 1000 |
| structure_3split_fixed_0.5 | GMean | graph_seq3d_vs_duet_graph | +0.0321 | +0.0129 to +0.0504 | 0.003996 | 0.05195 | 1000 |
| structure_3split_fixed_0.5 | MCC | graph_seqss3d_vs_graph_seq3d | -0.0328 | -0.0655 to +0.0040 | 0.07393 | 1 | 1000 |
| structure_3split_fixed_0.5 | AUC | graph_seqss3d_vs_graph_seq3d | -0.0237 | -0.0439 to -0.0022 | 0.02797 | 0.4755 | 1000 |
| structure_3split_fixed_0.5 | AUPRC | graph_seqss3d_vs_graph_seq3d | -0.0218 | -0.0386 to -0.0006 | 0.04396 | 0.8392 | 1000 |
| structure_3split_fixed_0.5 | F1 | graph_seqss3d_vs_graph_seq3d | -0.0075 | -0.0320 to +0.0158 | 0.5355 | 1 | 1000 |
| structure_3split_fixed_0.5 | Precision | graph_seqss3d_vs_graph_seq3d | -0.0358 | -0.0594 to -0.0074 | 0.01199 | 0.3117 | 1000 |
| structure_3split_fixed_0.5 | Recall | graph_seqss3d_vs_graph_seq3d | +0.0129 | -0.0138 to +0.0388 | 0.3217 | 1 | 1000 |
| structure_3split_fixed_0.5 | GMean | graph_seqss3d_vs_graph_seq3d | -0.0056 | -0.0262 to +0.0156 | 0.6014 | 1 | 1000 |
| structure_3split_fixed_0.5 | MCC | grassp_ernie_vs_cobra | +0.0915 | +0.0306 to +0.1553 | 0.003996 | 0.09191 | 1000 |
| structure_3split_fixed_0.5 | AUC | grassp_ernie_vs_cobra | +0.0220 | -0.0170 to +0.0687 | 0.2797 | 1 | 1000 |
| structure_3split_fixed_0.5 | AUPRC | grassp_ernie_vs_cobra | +0.0275 | -0.0140 to +0.0729 | 0.2198 | 1 | 1000 |
| structure_3split_fixed_0.5 | F1 | grassp_ernie_vs_cobra | +0.1736 | +0.1242 to +0.2290 | 0.001998 | 0.05195 | 1000 |
| structure_3split_fixed_0.5 | Precision | grassp_ernie_vs_cobra | -0.0174 | -0.0804 to +0.0354 | 0.4256 | 1 | 1000 |
| structure_3split_fixed_0.5 | Recall | grassp_ernie_vs_cobra | +0.3087 | +0.2488 to +0.3651 | 0.001998 | 0.05195 | 1000 |
| structure_3split_fixed_0.5 | GMean | grassp_ernie_vs_cobra | +0.1400 | +0.1038 to +0.1877 | 0.001998 | 0.05195 | 1000 |
| crossed_controls_3split | MCC | grassp_ernie_vs_grassp_rnafm | -0.0062 | -0.0577 to +0.0407 | 0.7892 | 1 | 1000 |
| crossed_controls_3split | AUC | grassp_ernie_vs_grassp_rnafm | -0.0150 | -0.0448 to +0.0131 | 0.2717 | 1 | 1000 |
| crossed_controls_3split | AUPRC | grassp_ernie_vs_grassp_rnafm | -0.0158 | -0.0482 to +0.0135 | 0.3017 | 1 | 1000 |
| crossed_controls_3split | F1 | grassp_ernie_vs_grassp_rnafm | -0.0104 | -0.0399 to +0.0172 | 0.4655 | 1 | 1000 |
| crossed_controls_3split | Precision | grassp_ernie_vs_grassp_rnafm | +0.0068 | -0.0217 to +0.0340 | 0.6853 | 1 | 1000 |
| crossed_controls_3split | Recall | grassp_ernie_vs_grassp_rnafm | -0.0405 | -0.0805 to -0.0059 | 0.02997 | 0.2398 | 1000 |
| crossed_controls_3split | GMean | grassp_ernie_vs_grassp_rnafm | -0.0038 | -0.0306 to +0.0218 | 0.7592 | 1 | 1000 |
| crossed_controls_3split | MCC | rgcn_ernie_vs_graph_seq3d | -0.0513 | -0.1102 to +0.0071 | 0.09191 | 1 | 1000 |
| crossed_controls_3split | AUC | rgcn_ernie_vs_graph_seq3d | -0.0443 | -0.0851 to -0.0100 | 0.02398 | 0.4316 | 1000 |
| crossed_controls_3split | AUPRC | rgcn_ernie_vs_graph_seq3d | -0.0214 | -0.0599 to +0.0183 | 0.3037 | 1 | 1000 |
| crossed_controls_3split | F1 | rgcn_ernie_vs_graph_seq3d | -0.0397 | -0.0797 to +0.0018 | 0.05794 | 0.5215 | 1000 |
| crossed_controls_3split | Precision | rgcn_ernie_vs_graph_seq3d | -0.0345 | -0.0786 to +0.0111 | 0.1479 | 1 | 1000 |
| crossed_controls_3split | Recall | rgcn_ernie_vs_graph_seq3d | -0.0489 | -0.0895 to +0.0002 | 0.05395 | 0.3237 | 1000 |
| crossed_controls_3split | GMean | rgcn_ernie_vs_graph_seq3d | -0.0311 | -0.0632 to +0.0029 | 0.07393 | 0.6653 | 1000 |
| crossed_controls_3split | MCC | grassp_rnafm_vs_graph_seq3d | +0.0484 | +0.0003 to +0.0997 | 0.04995 | 0.7992 | 1000 |
| crossed_controls_3split | AUC | grassp_rnafm_vs_graph_seq3d | +0.0405 | +0.0159 to +0.0671 | 0.001998 | 0.05195 | 1000 |
| crossed_controls_3split | AUPRC | grassp_rnafm_vs_graph_seq3d | +0.0374 | +0.0118 to +0.0692 | 0.007992 | 0.1838 | 1000 |
| crossed_controls_3split | F1 | grassp_rnafm_vs_graph_seq3d | +0.0856 | +0.0548 to +0.1173 | 0.001998 | 0.05195 | 1000 |
| crossed_controls_3split | Precision | grassp_rnafm_vs_graph_seq3d | -0.0250 | -0.0502 to +0.0018 | 0.07792 | 1 | 1000 |
| crossed_controls_3split | Recall | grassp_rnafm_vs_graph_seq3d | +0.2192 | +0.1829 to +0.2662 | 0.001998 | 0.05195 | 1000 |
| crossed_controls_3split | GMean | grassp_rnafm_vs_graph_seq3d | +0.0589 | +0.0306 to +0.0897 | 0.001998 | 0.05195 | 1000 |
| crossed_controls_3split | MCC | grassp_ernie_vs_rgcn_ernie | +0.0935 | +0.0499 to +0.1438 | 0.001998 | 0.05195 | 1000 |
| crossed_controls_3split | AUC | grassp_ernie_vs_rgcn_ernie | +0.0698 | +0.0437 to +0.1009 | 0.001998 | 0.05195 | 1000 |
| crossed_controls_3split | AUPRC | grassp_ernie_vs_rgcn_ernie | +0.0430 | +0.0162 to +0.0697 | 0.001998 | 0.05195 | 1000 |
| crossed_controls_3split | F1 | grassp_ernie_vs_rgcn_ernie | +0.1149 | +0.0826 to +0.1507 | 0.001998 | 0.05195 | 1000 |
| crossed_controls_3split | Precision | grassp_ernie_vs_rgcn_ernie | +0.0164 | -0.0149 to +0.0493 | 0.3277 | 1 | 1000 |
| crossed_controls_3split | Recall | grassp_ernie_vs_rgcn_ernie | +0.2276 | +0.1980 to +0.2603 | 0.001998 | 0.05195 | 1000 |
| crossed_controls_3split | GMean | grassp_ernie_vs_rgcn_ernie | +0.0862 | +0.0646 to +0.1110 | 0.001998 | 0.05195 | 1000 |
| external_independent_2split | MCC | duet_ensemble_vs_cobra | +0.1093 | +0.0699 to +0.1476 | 0.001998 | 0.00999 | 1000 |
| external_independent_2split | AUC | duet_ensemble_vs_cobra | +0.0729 | -0.0060 to +0.1359 | 0.1019 | 0.2877 | 1000 |
| external_independent_2split | AUPRC | duet_ensemble_vs_cobra | +0.0627 | -0.0017 to +0.1093 | 0.08392 | 0.1678 | 1000 |
| external_independent_2split | F1 | duet_ensemble_vs_cobra | +0.0801 | +0.0459 to +0.1078 | 0.001998 | 0.00999 | 1000 |
| external_independent_2split | Precision | duet_ensemble_vs_cobra | +0.0522 | +0.0198 to +0.0743 | 0.001998 | 0.00999 | 1000 |
| external_independent_2split | Recall | duet_ensemble_vs_cobra | +0.1349 | +0.0963 to +0.1826 | 0.001998 | 0.00999 | 1000 |
| external_independent_2split | GMean | duet_ensemble_vs_cobra | +0.1042 | +0.0432 to +0.1521 | 0.001998 | 0.00999 | 1000 |
| external_independent_2split | MCC | grassp_ernie_vs_duet_ensemble | -0.1195 | -0.2207 to -0.0082 | 0.04396 | 0.1319 | 1000 |
| external_independent_2split | AUC | grassp_ernie_vs_duet_ensemble | -0.1390 | -0.2121 to -0.0322 | 0.001998 | 0.00999 | 1000 |
| external_independent_2split | AUPRC | grassp_ernie_vs_duet_ensemble | -0.1179 | -0.1585 to -0.0386 | 0.001998 | 0.00999 | 1000 |
| external_independent_2split | F1 | grassp_ernie_vs_duet_ensemble | -0.0739 | -0.1252 to -0.0052 | 0.03996 | 0.1199 | 1000 |
| external_independent_2split | Precision | grassp_ernie_vs_duet_ensemble | -0.0636 | -0.0927 to -0.0190 | 0.003996 | 0.01598 | 1000 |
| external_independent_2split | Recall | grassp_ernie_vs_duet_ensemble | +0.0073 | -0.1526 to +0.1685 | 0.963 | 0.963 | 1000 |
| external_independent_2split | GMean | grassp_ernie_vs_duet_ensemble | -0.0688 | -0.1193 to +0.0179 | 0.1718 | 0.3437 | 1000 |
| external_independent_2split | MCC | grassp_rnafm_vs_duet_ensemble | -0.0027 | -0.0663 to +0.0790 | 0.9451 | 0.9451 | 1000 |
| external_independent_2split | AUC | grassp_rnafm_vs_duet_ensemble | +0.0101 | -0.0207 to +0.0751 | 0.3017 | 0.3017 | 1000 |
| external_independent_2split | AUPRC | grassp_rnafm_vs_duet_ensemble | +0.0394 | +0.0093 to +0.0839 | 0.01399 | 0.04196 | 1000 |
| external_independent_2split | F1 | grassp_rnafm_vs_duet_ensemble | -0.0091 | -0.0412 to +0.0373 | 0.8911 | 0.8911 | 1000 |
| external_independent_2split | Precision | grassp_rnafm_vs_duet_ensemble | -0.0233 | -0.0421 to +0.0116 | 0.2897 | 0.2897 | 1000 |
| external_independent_2split | Recall | grassp_rnafm_vs_duet_ensemble | +0.1316 | -0.0165 to +0.3006 | 0.1119 | 0.3357 | 1000 |
| external_independent_2split | GMean | grassp_rnafm_vs_duet_ensemble | +0.0171 | -0.0039 to +0.0810 | 0.0999 | 0.2997 | 1000 |
| external_independent_2split | MCC | grassp_rnafm_vs_cobra | +0.1066 | +0.0093 to +0.2080 | 0.02398 | 0.0959 | 1000 |
| external_independent_2split | AUC | grassp_rnafm_vs_cobra | +0.0830 | -0.0112 to +0.2053 | 0.08591 | 0.2877 | 1000 |
| external_independent_2split | AUPRC | grassp_rnafm_vs_cobra | +0.1021 | +0.0242 to +0.1829 | 0.001998 | 0.00999 | 1000 |
| external_independent_2split | F1 | grassp_rnafm_vs_cobra | +0.0710 | +0.0137 to +0.1343 | 0.005994 | 0.02398 | 1000 |
| external_independent_2split | Precision | grassp_rnafm_vs_cobra | +0.0290 | -0.0088 to +0.0758 | 0.1419 | 0.2837 | 1000 |
| external_independent_2split | Recall | grassp_rnafm_vs_cobra | +0.2665 | +0.1105 to +0.4552 | 0.001998 | 0.00999 | 1000 |
| external_independent_2split | GMean | grassp_rnafm_vs_cobra | +0.1213 | +0.0804 to +0.2213 | 0.001998 | 0.00999 | 1000 |
| external_independent_2split | MCC | graph_seqss3d_vs_graph_seq3d_external | -0.0124 | -0.0595 to +0.0071 | 0.2158 | 0.4316 | 1000 |
| external_independent_2split | AUC | graph_seqss3d_vs_graph_seq3d_external | -0.0184 | -0.0457 to +0.0018 | 0.07193 | 0.2877 | 1000 |
| external_independent_2split | AUPRC | graph_seqss3d_vs_graph_seq3d_external | -0.0026 | -0.0444 to +0.0452 | 0.6893 | 0.6893 | 1000 |
| external_independent_2split | F1 | graph_seqss3d_vs_graph_seq3d_external | -0.0104 | -0.0302 to +0.0042 | 0.1459 | 0.2917 | 1000 |
| external_independent_2split | Precision | graph_seqss3d_vs_graph_seq3d_external | -0.0149 | -0.0372 to -0.0014 | 0.03197 | 0.0959 | 1000 |
| external_independent_2split | Recall | graph_seqss3d_vs_graph_seq3d_external | +0.0229 | -0.0054 to +0.0475 | 0.1299 | 0.3357 | 1000 |
| external_independent_2split | GMean | graph_seqss3d_vs_graph_seq3d_external | +0.0010 | -0.0300 to +0.0181 | 0.7692 | 0.7692 | 1000 |
