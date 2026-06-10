#!/bin/bash
# ==============================================================================
# train_nosrc_dino4_gabor.sh
# ==============================================================================
# Ablation: No source encoder (pure Gaussian x₀) + DINO4 + Gabor cross-attn.
#
# Training uses x₀ ~ N(0, I) (standard flow matching).
# Eval uses x₀ ~ 0.01·N(0, I) for near-deterministic model selection
# (analogous to eval_use_mean in the source-encoder variant).
#
# Runs subjects 1, 2, 5, 7 sequentially.
# All experiments stored under exps/ablations/
# ==============================================================================

set -e

# Make sure we are in the project root directory
cd "$(dirname "$0")/.."

echo "=== Starting No-Source-Encoder + DINO4 + Gabor Ablation Training ==="
echo "Experiments directory: exps/ablations/"
echo "  - x₀ ~ N(0, I)   at training"
echo "  - x₀ ~ 0.01·N(0,I) at eval (near-deterministic)"
echo "=================================================================="

for sub in 1 2 5 7; do
    echo ""
    echo ">>> Running no-source + DINO4 + Gabor training for Subject $sub..."
    python src/train_factflow_fmri.py \
        --config src/configs/factflow/ablation/factflow_fmri_ablation_nosrc_dino4_gabor_sub${sub}.yaml \
        --exps_dir exps/ablations \
        --exp_name nosrc_dino4_gabor_sub${sub} \
        --resume_last
done

echo ""
echo "=== All No-Source Ablation Training runs completed successfully! ==="
