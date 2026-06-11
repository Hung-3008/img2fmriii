#!/bin/bash
# ==============================================================================
# train_rfr_all_subjects.sh
# ==============================================================================
# Train ROI-Stratified Feature Routing (RFR) model on all 4 subjects
# sequentially: sub1 → sub2 → sub5 → sub7.
#
# Architecture: DiT1D + use_roi_routing=true
#   - CLIP pool → AdaLN conditioning
#   - DINOv2 (multilayer4) + Gabor → per-stream cross-attn with ROI-level gates
#   - 3 visual buckets: early (V1/V2/V3), mid (V3A-PHC), high (LO/FFA/OPA...)
#
# Val metric: profile_r @ noise_scale=0 (deterministic ceiling)
# Best checkpoint: best.pt (selected by profile_r)
#
# Usage:
#   bash scripts/train_rfr_all_subjects.sh           # train sub 1,2,5,7
#   bash scripts/train_rfr_all_subjects.sh 1         # train only sub1
#   bash scripts/train_rfr_all_subjects.sh 1 2       # train sub1 and sub2
# ==============================================================================

set -e
cd "$(dirname "$0")/.."

# Subjects to train — defaults to all 4, or pass as args
SUBJECTS=("${@:-1 2 5 7}")
if [ "$#" -gt 0 ]; then
    SUBJECTS=("$@")
else
    SUBJECTS=(1 2 5 7)
fi

echo "========================================================"
echo "  FactFlow RFR — Multi-Subject Training"
echo "  Subjects : ${SUBJECTS[*]}"
echo "  Val      : profile_r @ noise_scale=0"
echo "========================================================"

for sub in "${SUBJECTS[@]}"; do
    exp_name="rfr_dino4_gabor_sub${sub}"
    cfg_path="exps/ablations/${exp_name}/config.yaml"

    if [ ! -f "$cfg_path" ]; then
        echo "ERROR: Config not found: $cfg_path" >&2
        exit 1
    fi

    echo ""
    echo ">>> Subject $sub — $exp_name"
    echo "    Config : $cfg_path"
    echo "    Ckpts  : exps/ablations/${exp_name}/checkpoints/"
    echo "    History: exps/ablations/${exp_name}/history.csv"
    echo ""

    uv run python src/train_factflow_fmri.py \
        --config   "$cfg_path" \
        --exps_dir exps/ablations \
        --exp_name "$exp_name" \
        --resume_last

    echo ""
    echo "    [DONE] Subject $sub"
    echo "--------------------------------------------------------"
done

echo ""
echo "========================================================"
echo "  All subjects done. Summary:"
echo ""
python - <<'PYEOF'
import os, pandas as pd

subjects = [1, 2, 5, 7]
rows = []
for s in subjects:
    csv = f"exps/ablations/rfr_dino4_gabor_sub{s}/history.csv"
    if os.path.exists(csv):
        df = pd.read_csv(csv)
        best_pr = df["val_profile_r"].max() if "val_profile_r" in df.columns else float("nan")
        best_vr = df["val_voxel_r"].max()   if "val_voxel_r"   in df.columns else float("nan")
        rows.append({"sub": s, "best_profile_r": best_pr, "best_voxel_r": best_vr})
    else:
        rows.append({"sub": s, "best_profile_r": "N/A", "best_voxel_r": "N/A"})

print(f"  {'Sub':<6} {'profile_r':>12} {'voxel_r':>12}")
print(f"  {'-'*6} {'-'*12} {'-'*12}")
for r in rows:
    pr = f"{r['best_profile_r']:.4f}" if isinstance(r['best_profile_r'], float) else r['best_profile_r']
    vr = f"{r['best_voxel_r']:.4f}"   if isinstance(r['best_voxel_r'],   float) else r['best_voxel_r']
    print(f"  {r['sub']:<6} {pr:>12} {vr:>12}")
PYEOF

echo "========================================================"
