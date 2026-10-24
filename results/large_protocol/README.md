# Large-protocol result layout

- `large_p80_cv5_audit.json`: pool, cluster and fold audit.
- `large_p80_cv5_clusters.csv`: component membership, fold and sampling weight.
- `large_p80_cv5_leakage.json`: exhaustive cross-fold identity audit.
- `graphs_audit.json`: node, edge and relation counts for every graph variant.
- `smartpocket_audit.json`: SMARTPocket de-leaking against the large pool.
- `smartpocket_graphs_audit.json`: external feature and 3D-coordinate audit.
- `cv/<method>/seed42_fold<f>.json`: held-out fold prediction and metrics.
- `cv/<method>/selected.json`: CV-selected epoch and threshold.
- `final/<method>/seed<seed>.json`: full-pool retrained model result.
- `external/<method>_seed<seed>.json`: SMARTPocket predictions.
- `analysis.json`: summaries, paired tests, Holm corrections and bootstrap.
- `LARGE_PROTOCOL_RESULTS.md`: generated human-readable report.

All final runs use the ten paired seeds
`42, 0, 1, 2, 3, 7, 11, 13, 17, 23`. A final run is only valid when its
`fixed_epoch` equals the corresponding `cv/<method>/selected.json` value and
its `threshold` equals the CV-selected threshold.
