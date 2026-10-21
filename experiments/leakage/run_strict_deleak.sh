#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1
PY="${PYTHON:-python}"
GRASSP_ROOT="${GRASSP_ROOT:-../grassp/GRASSP-main}"
SEEDS="42 0 1 2 3 7 11 13 17 23"
mkdir -p logs/strict

throttle() { while [ "$(jobs -r | wc -l)" -ge "$1" ]; do sleep 5; done; }

for THR in 80 70 50; do
  SS="data/processed/graphs_seq_ss_clean_p${THR}.pt"
  DUAL="data/processed/graphs_dual_clean_p${THR}.pt"
  for S in $SEEDS; do
    for NAME in graph tcn dense; do
      TAG="p${THR}_${NAME}"
      [ -f "checkpoints/controlled/${TAG}_seed${S}.pt" ] && { echo "[skip] $TAG $S"; continue; }
      case "$NAME" in
        graph) ARGS="--arch rgcn --graphs $SS --loss tcl_focal" ;;
        tcn)   ARGS="--arch tcn --graphs $DUAL --loss tcl_focal --model_kv ch=160" ;;
        dense) ARGS="--arch dense --graphs $DUAL --loss tcl_focal --model_kv stem=160,growth=40" ;;
      esac
      "$PY" src/train.py --config configs/duet.yaml --tag "$TAG" --seed "$S" --epochs 120 $ARGS \
        > "logs/strict/${TAG}_seed${S}.log" 2>&1 &
      throttle 3
    done
  done
done
wait
echo "==== towers done"

for THR in 80 70 50; do
  OUT="cobra_m3_equiv_p${THR}.json"
  [ -f "results/${OUT}" ] && { echo "[skip] baseline p$THR"; continue; }
  LEAKED_PAIRS="leaked_pairs_p${THR}.json" BASELINE_OUT="$OUT" BASELINE_MODE=clean \
    SEEDS="$SEEDS" "$PY" scripts/06_train_baselines.py clean \
    > "logs/strict/baseline_p${THR}.log" 2>&1 &
  throttle 3
done
wait
echo "==== baseline done"

for THR in 80 70 50; do
  [ -f "results/grassp_deleaked_p${THR}.json" ] && { echo "[skip] grassp p$THR"; continue; }
  GRASSP_LEAKED="leaked_pairs_p${THR}.json" GRASSP_OUT="grassp_deleaked_p${THR}.json" \
    GRASSP_ROOT="$GRASSP_ROOT" "$PY" baselines/grassp/retrain_deleaked.py --seeds $SEEDS \
    > "logs/strict/grassp_p${THR}.log" 2>&1 &
  throttle 3
done
wait
echo STRICT_DELEAK_DONE
