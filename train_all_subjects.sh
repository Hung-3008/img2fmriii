#!/bin/bash
set -e

# Train all subjects: Stage 1 (MoE VAE) → Stage 2 (Informed AdaLN Flow)
# Models: MoE VAE (4 experts, top-2) + Informed Prior + SimpleAdaLNFlow
#
# Usage:
#   bash train_all_subjects.sh          # train all
#   bash train_all_subjects.sh 01       # train single subject
#   bash train_all_subjects.sh 01 02    # train specific subjects

if [ $# -gt 0 ]; then
    SUBJECTS=("$@")
else
    SUBJECTS=(01 02 05 07)
fi

CONFIG_DIR="src/configs"

echo "=========================================="
echo "Training Pipeline: MoE VAE + Informed AdaLN"
echo "Subjects: ${SUBJECTS[*]}"
echo "=========================================="

for sub in "${SUBJECTS[@]}"; do
    echo ""
    echo "=========================================="
    echo " Subject ${sub} — Stage 1: MoE VAE"
    echo "=========================================="
    python -m src.train_stage1_mlp_vae \
        --config ${CONFIG_DIR}/subj${sub}/stage1_moe_vae.yaml
    if [ $? -ne 0 ]; then
        echo "ERROR: Stage 1 failed for subj${sub}!"
        exit 1
    fi

    echo ""
    echo "=========================================="
    echo " Subject ${sub} — Stage 2: Informed AdaLN Flow"
    echo "=========================================="
    python -m src.train_stage2_informed_flow \
        --config ${CONFIG_DIR}/subj${sub}/stage2_informed_adaln.yaml
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
