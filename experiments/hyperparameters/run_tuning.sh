#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1
PY="${PYTHON:-python}"

G=data/processed/graphs_seq_ss_clean.pt
M=data/processed/graphs_dual_clean.pt
mkdir -p logs/controlled_b

run() {
  local tag=$1; shift
  for S in 42 0 1 2 3; do
    local out="results/controlled/train_${tag}_seed${S}.json"
    if [ -f "$out" ]; then echo "[skip] $tag $S"; continue; fi
    "$PY" src/train.py --config configs/duet.yaml --tag "$tag" --seed "$S" \
      --loss tcl_focal --drop_val_leaked --epochs 120 "$@" \
      > "logs/controlled_b/${tag}_seed${S}.log" 2>&1 &
    while [ "$(jobs -r | wc -l)" -ge 3 ]; do sleep 5; done
  done
}

run tune_g_d03_lr3e4 --graphs "$G" --dropout 0.3 --lr 3e-4
run tune_g_d03_lr1e3 --graphs "$G" --dropout 0.3 --lr 1e-3
run tune_g_d05_lr1e3 --graphs "$G" --dropout 0.5 --lr 1e-3
run tune_g_h256      --graphs "$G" --hidden 256

run tune_m640_d005   --graphs "$M" --arch dual2_mlp --dropout 0.05
run tune_m640_d02    --graphs "$M" --arch dual2_mlp --dropout 0.2
run tune_m640_lr3e4  --graphs "$M" --arch dual2_mlp --lr 3e-4
run tune_m640_lr1e3  --graphs "$M" --arch dual2_mlp --lr 1e-3

wait
echo TUNING_DONE
