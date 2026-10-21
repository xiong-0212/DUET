# Experiments

Controlled experiments behind the paper's negative results and ablations. Everything here
shares one principle: **the loss, optimiser, schedule, seeds and the de-leaked split are held
fixed**, and only the factor under study changes, so the comparison is unconfounded.

All scripts resolve the repository root from their own location, write to
`results/controlled/` and `checkpoints/controlled/`, and are safe to re-run (existing runs
are skipped).

| Folder | Question | Runs | Conclusion |
|:--|:--|--:|:--|
| `probability_encoding/` | Do base-pairing probabilities carry usable information beyond binary edges? | 100 | No — all seven usage families fail to beat binary `seq`+`ss`; real weights ≈ shuffled weights |
| `gating/` | Is a learned gate better than equal-weight averaging? | 20 | The gate is informative and load-bearing, but does not outperform equal weights |
| `mamba/` | Does the official Mamba block beat the transformer in the sequence tower? | 10 | No — +0.007 MCC (p = 0.33) |
| `backbones/` | Is RNA-FM the reason for the gain? How do other RNA language models compare? | 5+ | RNA-FM 0.248 / RiNALMo 0.300 / ERNIE-RNA 0.273; ERNIE dilutes the ensemble |
| `ensemble/` | Which combination rule? Is the ensemble real? | — | Equal-weight averaging is optimal |
| `leakage/` | How much of the reported performance is memorisation? | — | 16/49 test chains are identical to training chains and 22/49 are ≥ 80 % homologous |
| `hyperparameters/` | Can tuning beat the baseline on TE18? | 40 | No configuration improved the benchmarks over the defaults |
| `capacity/` | Is the ensemble gain just more parameters? | 15 | Matched and enlarged single models remain below the ensemble |

## Drivers

The reproducible entry points are shell drivers (set `PYTHON` to your interpreter):

```bash
PYTHON=python bash experiments/ablation/run_ablation.sh
PYTHON=python bash experiments/hyperparameters/run_tuning.sh
```

and for the larger grids:

```bash
python experiments/probability_encoding/build_graphs_b.py
python experiments/probability_encoding/run_grid_b.py
python experiments/gating/gate_analysis.py
python experiments/ensemble/ensemble_probe.py
```

`probability_encoding/verify_weighted_graph.py` checks the custom weighted R-GCN convolution
against `torch_geometric`'s (13 assertions) before any of the weighted experiments are trusted.
