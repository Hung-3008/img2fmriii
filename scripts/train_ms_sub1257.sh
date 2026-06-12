#!/bin/bash
# ==============================================================================
# train_ms_sub1257.sh
# ==============================================================================
# Multi-subject FactFlow (shared trunk + per-subject adapters).
# Joint Training: TRAIN on all subjects {1, 2, 5, 7}; no held-out subject.
# ==============================================================================
set -e
cd "$(dirname "$0")/.."

CONFIG=src/configs/factflow/multisubject/factflow_ms_sub1257.yaml

echo "=== Multi-subject training: trunk and adapters on {1,2,5,7} ==="
python src/train_factflow_multisubject.py --config "$CONFIG" "$@"
