# Main DUET weights

The three towers are trained independently. Prediction averages their
probabilities with equal weights.

| Tower | Development tags | Confirmation tags | Parameters |
|:--|:--|:--|--:|
| Graph R-GCN | clean_seq_ss_seed42/0/1/2/3 | ns_graph_seed7/11/13/17/23 | 386,433 |
| TCN | slim_tcn_seed42/0/1/2/3 | ns_tcn_seed7/11/13/17/23 | 524,897 |
| DenseNet | slim_dense_seed42/0/1/2/3 | ns_dense_seed7/11/13/17/23 | 652,577 |

Each checkpoint stores `state_dict`, validation-selected `threshold`, model
configuration, graph metadata and validation/test metrics.
