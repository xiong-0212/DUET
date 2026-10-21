#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1
PY="${PYTHON:-python}"

G=data/processed/graphs_seq_ss_clean.pt
mkdir -p logs/controlled_b

for S in 7 11 13 17 23; do
  out="results/controlled/train_ns_mlp_rnafm_seed${S}.json"
  if [ -f "$out" ]; then echo "[skip] ns_mlp_rnafm $S"; continue; fi
  "$PY" src/train.py --config configs/duet.yaml --graphs "$G" \
    --tag ns_mlp_rnafm --seed "$S" --arch mlp --loss tcl_focal \
    --drop_val_leaked --epochs 120 \
    > "logs/controlled_b/ns_mlp_rnafm_seed${S}.log" 2>&1 &
  while [ "$(jobs -r | wc -l)" -ge 3 ]; do sleep 5; done
done
wait
echo GRAPH_CONTROL_CONFIRM_DONE
