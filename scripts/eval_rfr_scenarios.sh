#!/bin/bash
# ==============================================================================
# eval_rfr_scenarios.sh
# ==============================================================================
# Evaluate the trained ROI-Stratified Feature Routing (RFR) models on a grid of
# sampling scenarios, for subjects 1, 2, 5, 7.
#
# Scenarios (Cartesian product):
#   - noise_scale ∈ {1.0, 0.5, 0.2}   (Gaussian x₀ scale; higher = more stochastic)
#   - K           ∈ {1, 3, 5}          (number of trials averaged; max_trials = 5)
#
# For each subject × noise_scale we run a single evaluation with max_trials=5 and
# report metrics for K=1,3,5 (the evaluator averages the first K of the 5 passes).
# Uses the config.yaml + best.pt saved inside each experiment directory.
#
# Output:
#   - results/rfr_eval/rfr_eval_noise<scale>.csv   (one CSV per noise scale; rows
#     for every subject × K — the subject is identifiable from the `ckpt` column)
#   - results/rfr_eval/sub<S>_noise<scale>/        (per-pass .npy + averaged .npz)
#
# Usage:
#   bash scripts/eval_rfr_scenarios.sh                 # all subjects, full grid
#   bash scripts/eval_rfr_scenarios.sh --device cuda
#   BATCH_SIZE=256 bash scripts/eval_rfr_scenarios.sh  # override eval batch size
# ==============================================================================

set -e

cd "$(dirname "$0")/.."

DEVICE_ARG=""
[[ "${1:-}" == "--device" ]] && DEVICE_ARG="--device $2"

SUBJECTS=(1 2 5 7)
NOISE_SCALES=(1.0 0.5 0.2)
MAX_TRIALS=5
K_VALUES="1,3,5"
BATCH_SIZE=${BATCH_SIZE:-512}

OUT_ROOT="results/rfr_eval"
mkdir -p "$OUT_ROOT"

echo "=== Evaluating RFR models | K=${K_VALUES} (max_trials=${MAX_TRIALS}) | noise=${NOISE_SCALES[*]} ==="

for noise in "${NOISE_SCALES[@]}"; do
    CSV_OUT="${OUT_ROOT}/rfr_eval_noise${noise}.csv"
    echo ""
    echo "############ noise_scale = ${noise}  (CSV: ${CSV_OUT}) ############"
    for sub in "${SUBJECTS[@]}"; do
        exp_dir="exps/ablations/rfr_dino4_gabor_sub${sub}"
        ckpt="${exp_dir}/checkpoints/best.pt"
        config="${exp_dir}/config.yaml"

        if [[ ! -f "$ckpt" || ! -f "$config" ]]; then
            echo ">>> [skip] Subject $sub: missing ckpt or config ($ckpt)"
            continue
        fi

        echo ""
        echo ">>> Subject $sub | noise=${noise} ..."
        python src/eval_factflow_fmri.py \
            --config "$config" \
            --ckpt "$ckpt" \
            --eval_noise_scale "$noise" \
            --batch_size "$BATCH_SIZE" \
            --max_trials "$MAX_TRIALS" \
            --k_values "$K_VALUES" \
            --output "${OUT_ROOT}/sub${sub}_noise${noise}" \
            --csv_out "$CSV_OUT" \
            $DEVICE_ARG
    done
done

echo ""
echo "=== Evaluation complete. CSVs under ${OUT_ROOT}/ ==="
ls -1 "${OUT_ROOT}"/rfr_eval_noise*.csv 2>/dev/null || true
