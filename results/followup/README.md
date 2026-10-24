# Follow-up experiment outputs

- `all_metric_stats/all_metrics.json`: all seven metrics, paired tests, Holm
  correction and chain bootstrap for the pre-existing comparison families.
- `shuffled3d/`: degree-matched sequence+3D control and comparison with the
  real graph, sequence+SS graph, sequence+SS+3D graph and CoBRA.
- `size_curve/`: nested 80/160/290 training-size manifests, results,
  seven-metric curves and source data.
- `repeated_cv/`: three independent cluster-stratified five-fold assignments.
- `grasp_grid/`: full 24-point fold-0 screen, top-three five-fold confirmation,
  and ten-seed final models for ERNIE-RNA and RNA-FM.
- `duplicate_weighting/`: grouped, uniform and exact-deduplicated training
  comparisons for DUET graph and GRASSP RNA-FM.
- `protocol/manifest_audit.json`: subset, repeated-CV and exact-dedup checks.
- `../large_protocol/smartpocket_audit.json`: canonical-test and pool de-leaked
  SMARTPocket external evaluation (Jiang 2 chains; time-dependent 11 chains;
  source-matched HARIBOSS control excluded from the independent mean).
- `acceptance.json`: final machine-checkable acceptance gate.

All reported comparisons use paired training seeds. Chain bootstrap keeps
seed pairing and resamples benchmark chains. `hariboss_test` is a
source-matched control and is never included in independent external means.
