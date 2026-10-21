#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1
PY="${PYTHON:-python}"

G=data/processed/graphs_seq_ss_clean.pt
SEEDS="7 11 13 17 23"
mkdir -p logs/controlled_b

run() {
  local tag=$1; shift
  for S in $SEEDS; do
    local out="results/controlled/train_${tag}_seed${S}.json"
    if [ -f "$out" ]; then echo "[skip] $tag $S"; continue; fi
    "$PY" src/train.py --config configs/duet.yaml --tag "$tag" --seed "$S" \
      --loss tcl_focal --drop_val_leaked --epochs 120 "$@" \
      > "logs/controlled_b/${tag}_seed${S}.log" 2>&1 &
    while [ "$(jobs -r | wc -l)" -ge 3 ]; do sleep 5; done
  done
}

run clean_abl_seqonly   --graphs data/processed/graphs_seq_only_clean.pt
run clean_abl_nores     --graphs "$G" --no_residual
run clean_abl_layernorm --graphs "$G" --norm layer
run clean_abl_nodrop    --graphs "$G" --drop_edge 0.0
run clean_abl_bce       --graphs "$G" --loss bce

wait
echo ABLATION_CONFIRM_DONE
