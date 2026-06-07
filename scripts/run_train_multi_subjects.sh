#!/usr/bin/env bash
# ==============================================================================
# run_train_multi_subjects.sh
# ==============================================================================
# Train FactFlow fMRI model for subjects 2, 5, 7 (sequentially on single GPU).
# Creates temporary per-subject configs from the base srcdist_v2 config
# with subject-specific overrides (subject ID, n_voxels, epochs).
#
# Before training, creates necessary symlinks so the dataset can find
# feature files at the expected paths (subj root, not in subfolders).
#
# Usage:
#   bash scripts/run_train_multi_subjects.sh
#
# Optional:
#   --epochs 50 --batch_size 32 --device cuda
# ==============================================================================

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────
EPOCHS=50
BATCH_SIZE=32
DEVICE=""
CONFIG="src/configs/factflow/srcdist_v2/factflow_fmri_cross_dino_srcdist_v2.yaml"
DATA_DIR="NSD/data/nsd"

# ── Parse arguments ──────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs)     EPOCHS="$2";     shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --device)     DEVICE="$2";     shift 2 ;;
        --config)     CONFIG="$2";     shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Setup ─────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"

# Subject → n_voxels mapping (from NSD ROI data)
# ref: MindEyeV2 subject_sizes = [0, 15724, 14278, 15226, 13153, 13039, 17907, 12682, 14386]
declare -A NVOXELS
NVOXELS[2]=14278
NVOXELS[5]=13039
NVOXELS[7]=12682

SUBJECTS=(2 5 7)

echo "=============================================================="
echo "FactFlow fMRI — Multi-Subject Training"
echo "=============================================================="
echo "  Base config: $CONFIG"
echo "  Subjects:    ${SUBJECTS[*]}"
echo "  Epochs:      $EPOCHS"
echo "  Batch size:  $BATCH_SIZE"
echo "  Device:      ${DEVICE:-auto}"
echo "=============================================================="

# ── Step 1: Create symlinks for data files ────────────────────────────
echo ""
echo "── Creating data symlinks ──"

for s in "${SUBJECTS[@]}"; do
    SUBJ_DIR="${DATA_DIR}/subj0${s}"

    if [[ ! -d "$SUBJ_DIR" ]]; then
        echo "  ERROR: $SUBJ_DIR does not exist!"
        exit 1
    fi

    # Symlink files from subdirs to root (dataset expects root-level .npy)
    for subdir in fmri clip dino; do
        if [[ -d "${SUBJ_DIR}/${subdir}" ]]; then
            for f in "${SUBJ_DIR}/${subdir}/"*.npy; do
                [[ -f "$f" ]] || continue
                base="$(basename "$f")"
                target="${SUBJ_DIR}/${base}"
                if [[ ! -e "$target" ]]; then
                    ln -s "${subdir}/${base}" "$target"
                    echo "  ✓ subj0${s}/${base} → ${subdir}/${base}"
                fi
            done
        fi
    done
done

echo "  Symlinks ready."

# ── Step 2: Verify all required files ─────────────────────────────────
echo ""
echo "── Verifying data files ──"

ALL_OK=true
for s in "${SUBJECTS[@]}"; do
    SUBJ_DIR="${DATA_DIR}/subj0${s}"
    MISSING=""

    for mode in train test; do
        [[ -e "${SUBJ_DIR}/nsd_${mode}_fmri_zscore_sub${s}.npy" ]]       || MISSING+=" fmri_${mode}"
        [[ -e "${SUBJ_DIR}/nsd_sdxl_clip_${mode}_sub${s}.npy" ]]         || MISSING+=" clip_${mode}"
        [[ -e "${SUBJ_DIR}/nsd_sdxl_clip_pool_${mode}_sub${s}.npy" ]]    || MISSING+=" clip_pool_${mode}"
        [[ -e "${SUBJ_DIR}/nsd_dinov2_vitl14_${mode}_sub${s}.npy" ]]     || MISSING+=" dino_${mode}"
    done

    if [[ -z "$MISSING" ]]; then
        echo "  ✓ subj0${s}: all files OK (n_voxels=${NVOXELS[$s]})"
    else
        echo "  ✗ subj0${s}: MISSING:${MISSING}"
        ALL_OK=false
    fi
done

if [[ "$ALL_OK" != "true" ]]; then
    echo ""
    echo "ERROR: Some data files are missing. Aborting."
    exit 1
fi

# ── Step 3: Generate per-subject configs & train ──────────────────────
CONFIG_DIR="src/configs/factflow/srcdist_v2"
mkdir -p "$CONFIG_DIR"

for s in "${SUBJECTS[@]}"; do
    N_VOXELS=${NVOXELS[$s]}
    EXP_NAME="srcdist_v2_sub${s}"
    SUB_CONFIG="${CONFIG_DIR}/factflow_fmri_cross_dino_srcdist_v2_sub${s}.yaml"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Training Subject ${s} (n_voxels=${N_VOXELS}, epochs=${EPOCHS})"
    echo "  Experiment: ${EXP_NAME}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Generate per-subject config by patching the base config.
    # Only subject/n_voxels/epochs/batch_size are overridden — pad_to, seq_len
    # and num_queries are sized to n_voxels at runtime by data.auto_pad (set in
    # the base config), so no per-subject pad_to is hardcoded here.
    $PYTHON -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('${CONFIG}')
cfg.data.subject = ${s}
cfg.data.n_voxels = ${N_VOXELS}
cfg.training.epochs = ${EPOCHS}
cfg.training.batch_size = ${BATCH_SIZE}
OmegaConf.save(cfg, '${SUB_CONFIG}')
print(f'  Generated config: ${SUB_CONFIG}')
print(f'    subject={cfg.data.subject}, n_voxels={cfg.data.n_voxels}, '
      f'auto_pad={cfg.data.get(\"auto_pad\", False)}, epochs={cfg.training.epochs}')
"

    DEVICE_ARG=""
    if [[ -n "$DEVICE" ]]; then
        DEVICE_ARG="--device $DEVICE"
    fi

    $PYTHON src/train_factflow_fmri.py \
        --config "$SUB_CONFIG" \
        --exp_name "$EXP_NAME" \
        $DEVICE_ARG

    echo "  ✓ Subject ${s} training complete"
    echo ""
done

echo "=============================================================="
echo "All subjects trained!"
echo "=============================================================="
echo ""
echo "Experiments:"
for s in "${SUBJECTS[@]}"; do
    echo "  exps/srcdist_v2_sub${s}/"
done
echo ""
echo "To evaluate, run:"
for s in "${SUBJECTS[@]}"; do
    echo "  bash scripts/run_eval_scenarios.sh \\"
    echo "    --config src/configs/factflow/srcdist_v2/factflow_fmri_cross_dino_srcdist_v2_sub${s}.yaml \\"
    echo "    --ckpt exps/srcdist_v2_sub${s}/checkpoints/best.pt \\"
    echo "    --out_dir results/eval_scenarios_sub${s}"
    echo ""
done
