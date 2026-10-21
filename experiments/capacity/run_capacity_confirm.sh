#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1
PY="${PYTHON:-python}"

G=data/processed/graphs_dual_clean.pt
SEEDS="7 11 13 17 23"
mkdir -p logs/controlled_b

run() {
  local tag=$1; shift
  for S in $SEEDS; do
    local out="results/controlled/train_${tag}_seed${S}.json"
    if [ -f "$out" ]; then echo "[skip] $tag $S"; continue; fi
    "$PY" src/train.py --config configs/duet.yaml --graphs "$G" \
      --tag "$tag" --seed "$S" --arch dual2_mlp --loss tcl_focal \
      --drop_val_leaked --epochs 120 "$@" \
      > "logs/controlled_b/${tag}_seed${S}.log" 2>&1 &
    while [ "$(jobs -r | wc -l)" -ge 3 ]; do sleep 5; done
  done
}

run ns_mlp
run ns_cap_m1275 --mlp_widths 1275,256,128
run ns_cap_m2048 --mlp_widths 2048,256,128
run ns_cap_m1920 --split_dim 0

wait
echo CAPACITY_CONFIRM_DONE
