#!/bin/bash
# ==============================================================================
# eval_nosrc_dino4_gabor.sh
# ==============================================================================
# Evaluate trained no-source + DINO4 + Gabor models (subjects 1, 2, 5, 7).
#
# Source x₀ is scaled Gaussian noise (eval_noise_scale from the config, ≈0 →
# near-deterministic) integrated with an ODE solver. Metrics on rep-averaged GT.
#
# Usage:
#   bash scripts/eval_nosrc_dino4_gabor.sh                 # all subjects, K=1
#   bash scripts/eval_nosrc_dino4_gabor.sh --device cuda
# ==============================================================================

set -e

cd "$(dirname "$0")/.."

DEVICE_ARG=""
[[ "${1:-}" == "--device" ]] && DEVICE_ARG="--device $2"

CSV_OUT="results/eval_nosrc_dino4_gabor.csv"

echo "=== Evaluating No-Source + DINO4 + Gabor models ==="
for sub in 1 2 5 7; do
    echo ""
    echo ">>> Subject $sub ..."
    python src/eval_factflow_fmri.py \
        --config src/configs/factflow/ablation/factflow_fmri_ablation_nosrc_dino4_gabor_sub${sub}.yaml \
        --ckpt exps/ablations/nosrc_dino4_gabor_sub${sub}/checkpoints/best.pt \
        --output results/eval_nosrc_dino4_gabor_sub${sub} \
        --csv_out "$CSV_OUT" \
        $DEVICE_ARG
done

echo ""
echo "=== Evaluation complete. Results: $CSV_OUT ==="
