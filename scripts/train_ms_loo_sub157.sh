#!/bin/bash
# ==============================================================================
# train_ms_loo_sub157.sh
# ==============================================================================
# Multi-subject FactFlow (shared trunk + per-subject adapters).
# Leave-one-out: TRAIN on subjects {1, 5, 7}; subject 2 held out.
#
# Few-shot adaptation to the held-out subject is now a real training loop,
# driven separately by scripts/train_fewshot_1h_all.sh
# (train_factflow_multisubject.py --fewshot_held_out 2).
# ==============================================================================
set -e
cd "$(dirname "$0")/.."

CONFIG=src/configs/factflow/multisubject/factflow_ms_sub157.yaml

echo "=== Multi-subject training: trunk on {1, 5, 7}, hold out 2 ==="
python src/train_factflow_multisubject.py --config "$CONFIG" "$@"
