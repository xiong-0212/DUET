# GRASSP hyperparameter sensitivity

The full 24-point author grid was evaluated on the fixed large-data fold 0 for each encoder with seed 42. The top three configurations were then confirmed on folds 1-4. Selection used pooled five-fold OOF MCC; the selected configuration was retrained with ten paired seeds.

| Encoder | Selected configuration | OOF MCC | OOF AUC | OOF AUPRC |
|:--|:--|--:|--:|--:|
| ernie | lr0.0001_bs1_h128_d0.3 | 0.2510 | 0.6412 | 0.4720 |
| rnafm | lr0.001_bs16_h128_d0.3 | 0.2433 | 0.6475 | 0.4599 |

## Final ten-seed summaries

| Model | Metric | Mean | SD |
|:--|:--|--:|--:|
| DUET ensemble | MCC | 0.2274 | 0.0157 |
| DUET ensemble | AUC | 0.6722 | 0.0085 |
| DUET ensemble | AUPRC | 0.4939 | 0.0131 |
| DUET ensemble | F1 | 0.4734 | 0.0316 |
| DUET ensemble | Precision | 0.5028 | 0.0181 |
| DUET ensemble | Recall | 0.4581 | 0.0669 |
| DUET ensemble | GMean | 0.5867 | 0.0259 |
| ernie | MCC | 0.2148 | 0.0176 |
| ernie | AUC | 0.6494 | 0.0111 |
| ernie | AUPRC | 0.4833 | 0.0147 |
| ernie | F1 | 0.4729 | 0.0129 |
| ernie | Precision | 0.4870 | 0.0142 |
| ernie | Recall | 0.4614 | 0.0212 |
| ernie | GMean | 0.5873 | 0.0105 |
| rnafm | MCC | 0.2083 | 0.0088 |
| rnafm | AUC | 0.6605 | 0.0053 |
| rnafm | AUPRC | 0.4724 | 0.0156 |
| rnafm | F1 | 0.4667 | 0.0222 |
| rnafm | Precision | 0.4852 | 0.0161 |
| rnafm | Recall | 0.4547 | 0.0523 |
| rnafm | GMean | 0.5815 | 0.0178 |

## Final ten-seed comparisons

| Metric | Comparison | Delta | 95% CI | raw p | Holm p |
|:--|:--|--:|:--|--:|--:|
| MCC | grassp_ernie_vs_duet | -0.0127 | -0.0339 to +0.0085 | 0.2092 | 0.4185 |
| MCC | grassp_rnafm_vs_duet | -0.0192 | -0.0321 to -0.0062 | 0.008481 | 0.02544 |
| MCC | grassp_ernie_vs_rnafm | +0.0065 | -0.0066 to +0.0195 | 0.2898 | 0.4185 |
| AUC | grassp_ernie_vs_duet | -0.0228 | -0.0350 to -0.0106 | 0.002267 | 0.006801 |
| AUC | grassp_rnafm_vs_duet | -0.0118 | -0.0181 to -0.0054 | 0.002285 | 0.006801 |
| AUC | grassp_ernie_vs_rnafm | -0.0110 | -0.0202 to -0.0019 | 0.02356 | 0.02356 |
| AUPRC | grassp_ernie_vs_duet | -0.0106 | -0.0270 to +0.0057 | 0.1751 | 0.1858 |
| AUPRC | grassp_rnafm_vs_duet | -0.0215 | -0.0339 to -0.0091 | 0.003488 | 0.01046 |
| AUPRC | grassp_ernie_vs_rnafm | +0.0109 | -0.0022 to +0.0240 | 0.09291 | 0.1858 |
| F1 | grassp_ernie_vs_duet | -0.0005 | -0.0267 to +0.0256 | 0.965 | 1 |
| F1 | grassp_rnafm_vs_duet | -0.0066 | -0.0410 to +0.0277 | 0.6723 | 1 |
| F1 | grassp_ernie_vs_rnafm | +0.0061 | -0.0051 to +0.0174 | 0.2505 | 0.7514 |
| Precision | grassp_ernie_vs_duet | -0.0158 | -0.0330 to +0.0015 | 0.06859 | 0.1484 |
| Precision | grassp_rnafm_vs_duet | -0.0175 | -0.0350 to -0.0001 | 0.04945 | 0.1484 |
| Precision | grassp_ernie_vs_rnafm | +0.0018 | -0.0157 to +0.0192 | 0.8246 | 0.8246 |
| Recall | grassp_ernie_vs_duet | +0.0033 | -0.0475 to +0.0541 | 0.8863 | 1 |
| Recall | grassp_rnafm_vs_duet | -0.0033 | -0.0781 to +0.0714 | 0.922 | 1 |
| Recall | grassp_ernie_vs_rnafm | +0.0066 | -0.0231 to +0.0363 | 0.6262 | 1 |
| GMean | grassp_ernie_vs_duet | +0.0007 | -0.0207 to +0.0220 | 0.9459 | 1 |
| GMean | grassp_rnafm_vs_duet | -0.0052 | -0.0329 to +0.0225 | 0.6819 | 1 |
| GMean | grassp_ernie_vs_rnafm | +0.0058 | -0.0033 to +0.0150 | 0.1828 | 0.5483 |
