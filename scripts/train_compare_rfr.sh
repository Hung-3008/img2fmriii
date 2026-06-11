#!/bin/bash
# ==============================================================================
# train_compare_rfr.sh
# ==============================================================================
# So sánh trực tiếp hai config chỉ khác nhau ở use_roi_routing:
#
#   BASELINE : nosrc_dino4_gabor_sub1   (use_roi_routing: false)
#   RFR      : rfr_dino4_gabor_sub1     (use_roi_routing: true)
#
# Cả hai dùng cùng data, depth, hidden_size, seed → kết quả so sánh sạch.
# Val metric: profile_r với noise_scale=0 (deterministic ceiling)
#
# Usage:
#   bash scripts/train_compare_rfr.sh              # train cả hai
#   bash scripts/train_compare_rfr.sh baseline     # chỉ baseline
#   bash scripts/train_compare_rfr.sh rfr          # chỉ RFR
# ==============================================================================

set -e
cd "$(dirname "$0")/.."

MODE=${1:-"both"}   # both | baseline | rfr

echo "============================================================"
echo "  FactFlow ROI-Routing Comparison"
echo "  Mode: $MODE"
echo "  Val: profile_r @ noise_scale=0 (deterministic ceiling)"
echo "============================================================"

# ── Baseline: No ROI routing ─────────────────────────────────────────────────
if [[ "$MODE" == "both" || "$MODE" == "baseline" ]]; then
    echo ""
    echo ">>> [1/2] BASELINE — nosrc_dino4_gabor_sub1 (use_roi_routing: false)"
    echo "    Config : exps/ablations/nosrc_dino4_gabor_sub1/config.yaml"
    echo "    Logs   : exps/ablations/nosrc_dino4_gabor_sub1/factflow_trainer.log"
    echo "    History: exps/ablations/nosrc_dino4_gabor_sub1/history.csv"
    echo ""
    uv run python src/train_factflow_fmri.py \
        --config  exps/ablations/nosrc_dino4_gabor_sub1/config.yaml \
        --exps_dir exps/ablations \
        --exp_name nosrc_dino4_gabor_sub1 \
        --resume_last
fi

# ── RFR: ROI-Stratified Feature Routing ──────────────────────────────────────
if [[ "$MODE" == "both" || "$MODE" == "rfr" ]]; then
    echo ""
    echo ">>> [2/2] RFR — rfr_dino4_gabor_sub1 (use_roi_routing: true)"
    echo "    Config : exps/ablations/rfr_dino4_gabor_sub1/config.yaml"
    echo "    Logs   : exps/ablations/rfr_dino4_gabor_sub1/factflow_trainer.log"
    echo "    History: exps/ablations/rfr_dino4_gabor_sub1/history.csv"
    echo ""
    uv run python src/train_factflow_fmri.py \
        --config  exps/ablations/rfr_dino4_gabor_sub1/config.yaml \
        --exps_dir exps/ablations \
        --exp_name rfr_dino4_gabor_sub1 \
        --resume_last
fi

echo ""
echo "============================================================"
echo "  Training complete. Compare results:"
echo ""
echo "  python -c \""
echo "import pandas as pd"
echo "b = pd.read_csv('exps/ablations/nosrc_dino4_gabor_sub1/history.csv')"
echo "r = pd.read_csv('exps/ablations/rfr_dino4_gabor_sub1/history.csv')"
echo "print('BASELINE best profile_r:', b['val_profile_r'].max())"
echo "print('RFR      best profile_r:', r['val_profile_r'].max())"
echo "print('BASELINE best voxel_r  :', b['val_voxel_r'].max())"
echo "print('RFR      best voxel_r  :', r['val_voxel_r'].max())"
echo "\""
echo "============================================================"
