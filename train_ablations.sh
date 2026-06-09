#!/bin/bash
# ==============================================================================
# train_ablations.sh
# ==============================================================================
# Script to run CLIP-only and DINO-only ablation training for all 4 subjects
# (1, 2, 5, 7) sequentially.
#
# All experiments will be stored under exps/ablations/
# ==============================================================================

set -e

# Make sure we are in the project root directory
cd "$(dirname "$0")"

echo "=== Starting CLIP-only and DINO-only Ablation Training ==="
echo "Experiments directory: exps/ablations/"
echo "========================================================="

# --- CLIP-only Ablation Training ---
for sub in 1 2 5 7; do
    echo ""
    echo ">>> Running CLIP-only training for Subject $sub..."
    python src/train_factflow_fmri.py \
        --config src/configs/factflow/ablation/factflow_fmri_ablation_clip_only_sub${sub}.yaml \
        --exps_dir exps/ablations \
        --exp_name clip_only_sub${sub}
done

# --- DINO-only Ablation Training ---
for sub in 1 2 5 7; do
    echo ""
    echo ">>> Running DINO-only training for Subject $sub..."
    python src/train_factflow_fmri.py \
        --config src/configs/factflow/ablation/factflow_fmri_ablation_dino_only_sub${sub}.yaml \
        --exps_dir exps/ablations \
        --exp_name dino_only_sub${sub}
done

echo ""
echo "=== All Ablation Training runs completed successfully! ==="
