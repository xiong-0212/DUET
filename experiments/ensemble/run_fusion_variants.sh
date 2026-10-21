#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1
PY="${PYTHON:-python}"

M=data/processed/graphs_dual_clean.pt
G=data/processed/graphs_seq_ss_clean.pt
mkdir -p logs/controlled_b

run_one() {
  local tag=$1 arch=$2 graphs=$3 seed=$4
  local out="results/controlled/train_${tag}_seed${seed}.json"
  if [ -f "$out" ]; then echo "[skip] $tag $seed"; return; fi
  "$PY" src/train.py --config configs/duet.yaml --graphs "$graphs" \
    --tag "$tag" --seed "$seed" --arch "$arch" --loss tcl_focal \
    --drop_val_leaked --epochs 120 > "logs/controlled_b/${tag}_seed${seed}.log" 2>&1 &
  while [ "$(jobs -r | wc -l)" -ge 3 ]; do sleep 5; done
}

for S in 42 0 1 2 3; do
  run_one clean_dual2       dual2       "$M" "$S"
  run_one clean_dual2_avg   dual2_avg   "$M" "$S"
  run_one clean_dual2_logit dual2_logit "$M" "$S"
  run_one clean_dual2_graph dual2_graph "$M" "$S"
  run_one clean_dualview_mambaoff dualview_mamba_official "$G" "$S"
  run_one clean_fusion_mambaoff   fusion_mamba_official   "$G" "$S"
done

wait
echo FUSION_VARIANTS_DONE
