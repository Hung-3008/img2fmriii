#!/usr/bin/env bash
# ==============================================================================
# run_eval_scenarios.sh
# ==============================================================================
# Run FactFlow fMRI evaluation across all 3 inference scenarios.
#
# Scenarios:
#   1. deterministic        — μ + ODE  (K=1, fully deterministic)
#   2. perceiver_stochastic — sample(μ,σ) + ODE  (K=1, 5, 10)
#   3. flow_stochastic      — μ + SDE  (K=1, 5, 10)
#
# Usage:
#   bash scripts/run_eval_scenarios.sh \
#       --config src/configs/factflow_fmri_cross_dino_srcdist_v2.yaml \
#       --ckpt exps/factflow_fmri_sub1/checkpoints/best.pt
#
# Optional:
#   --batch_size 32 --device cuda --csv_dir results/eval_scenarios
# ==============================================================================

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────
CONFIG=""
CKPT=""
BATCH_SIZE=32
DEVICE=""
CSV_DIR="results/eval_scenarios"
NUM_WORKERS=4

# ── Parse arguments ──────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)      CONFIG="$2";      shift 2 ;;
        --ckpt)        CKPT="$2";        shift 2 ;;
        --batch_size)  BATCH_SIZE="$2";  shift 2 ;;
        --device)      DEVICE="$2";      shift 2 ;;
        --csv_dir)     CSV_DIR="$2";     shift 2 ;;
        --num_workers) NUM_WORKERS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$CONFIG" || -z "$CKPT" ]]; then
    echo "ERROR: --config and --ckpt are required."
    echo "Usage: bash scripts/run_eval_scenarios.sh --config <path> --ckpt <path>"
    exit 1
fi

# ── Setup ─────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"

mkdir -p "$CSV_DIR"

# CSV output file — one file, all results appended
CSV_OUT="${CSV_DIR}/eval_results.csv"

# Derive checkpoint name for output filenames
CKPT_NAME="$(basename "$CKPT" .pt)"

DEVICE_ARG=""
if [[ -n "$DEVICE" ]]; then
    DEVICE_ARG="--device $DEVICE"
fi

echo "=============================================================="
echo "FactFlow fMRI Evaluation — All Scenarios"
echo "=============================================================="
echo "  Config:     $CONFIG"
echo "  Checkpoint: $CKPT"
echo "  Batch size: $BATCH_SIZE"
echo "  CSV output: $CSV_OUT"
echo "  Device:     ${DEVICE:-auto}"
echo "=============================================================="

# ── Helper function ───────────────────────────────────────────────────
run_eval() {
    local scenario="$1"
    local k_trials="$2"
    local extra_args="${3:-}"

    local label="${scenario}_k${k_trials}"
    local npz_out="${CSV_DIR}/${CKPT_NAME}_${label}.npz"

    echo ""
    echo "--------------------------------------------------------------"
    echo "  Running: scenario=${scenario}  k_trials=${k_trials}"
    echo "  Output:  ${npz_out}"
    echo "--------------------------------------------------------------"

    $PYTHON src/eval_factflow_fmri.py \
        --config "$CONFIG" \
        --ckpt "$CKPT" \
        --batch_size "$BATCH_SIZE" \
        --num_workers "$NUM_WORKERS" \
        $DEVICE_ARG \
        --scenario "$scenario" \
        --k_trials "$k_trials" \
        --output "$npz_out" \
        --csv_out "$CSV_OUT" \
        $extra_args

    echo "  ✓ Done: ${label}"
}

# ── Scenario 1: Deterministic (μ + ODE, K=1) ────────────────────────
run_eval "deterministic" 1

# ── Scenario 2: Perceiver stochastic (sample + ODE) ─────────────────
for K in 1 5 10; do
    run_eval "perceiver_stochastic" "$K"
done

# ── Scenario 3: Flow stochastic (μ + SDE) ───────────────────────────
for K in 1 5 10; do
    run_eval "flow_stochastic" "$K" "--sde_num_steps 250"
done

# ── Summary ───────────────────────────────────────────────────────────
echo ""
echo "=============================================================="
echo "All evaluations complete!"
echo "Results saved to: ${CSV_OUT}"
echo "=============================================================="
echo ""

if [[ -f "$CSV_OUT" ]]; then
    echo "Summary:"
    column -t -s',' "$CSV_OUT"
fi
