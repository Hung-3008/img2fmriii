#!/bin/bash
set -e

# Train all subjects sequentially: Stage 1 (VAE) → Stage 2 (Residual Flow)
SUBJECTS=(01 02 05 07)

echo "=========================================="
echo "Starting full training pipeline"
echo "Subjects: ${SUBJECTS[*]}"
echo "=========================================="

for sub in "${SUBJECTS[@]}"; do
    echo ""
    echo "=========================================="
    echo " Subject ${sub} — Stage 1: VAE"
    echo "=========================================="
    python -m src.train_stage1_mlp_vae \
        --config src/configs/fmri_mlp_vae_768_subj${sub}.yaml
    if [ $? -ne 0 ]; then
        echo "ERROR: Stage 1 failed for subj${sub}!"
        exit 1
    fi

    echo ""
    echo "=========================================="
    echo " Subject ${sub} — Stage 2: Residual Flow"
    echo "=========================================="
    python -m src.train_stage2_residual_flow \
        --config src/configs/stage2_residual_flow_v5_subj${sub}.yaml
    if [ $? -ne 0 ]; then
        echo "ERROR: Stage 2 failed for subj${sub}!"
        exit 1
    fi

    echo ""
    echo "✅ Subject ${sub} completed!"
done

echo ""
echo "=========================================="
echo "All subjects completed successfully!"
echo "=========================================="
