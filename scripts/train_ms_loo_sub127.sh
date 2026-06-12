#!/bin/bash
# ==============================================================================
# train_ms_loo_sub127.sh
# ==============================================================================
# Multi-subject FactFlow (shared trunk + per-subject adapters).
# Leave-one-out: TRAIN on subjects {1, 2, 7}; subject 5 held out for few-shot.
#
# After training, run few-shot adaptation to subject 5:
#   python src/eval_factflow_fewshot.py \
#       --config src/configs/factflow/multisubject/factflow_ms_sub127.yaml \
#       --ckpt exps/factflow_ms_sub127/checkpoints/best.pt \
#       --held_out 5 --k_list 10 50 200 -1
# ==============================================================================
set -e
cd "$(dirname "$0")/.."

CONFIG=src/configs/factflow/multisubject/factflow_ms_sub127.yaml

echo "=== Multi-subject training: trunk on {1,2,7}, hold out 5 ==="
python src/train_factflow_multisubject.py --config "$CONFIG" "$@"

echo "=== Few-shot adaptation to held-out subject 5 ==="
python src/eval_factflow_fewshot.py \
    --config "$CONFIG" \
    --ckpt exps/factflow_ms_sub127/checkpoints/best.pt \
    --held_out 5 --k_list 10 50 200 -1 --adapt_steps 300
