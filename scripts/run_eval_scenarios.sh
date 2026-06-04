#!/usr/bin/env bash
# ==============================================================================
# run_eval_scenarios.sh
# ==============================================================================
# Run FactFlow fMRI evaluation across all 3 inference scenarios.
#
# Strategy:
#   1. deterministic         — 1 pass  (μ + ODE, fully deterministic)
#   2. perceiver_stochastic  — 10 passes → metrics for K=1,5,10
#   3. flow_stochastic       — 10 passes → metrics for K=1,5,10
#
# Total: 1 + 10 + 10 = 21 forward passes (not 1+16+16=33).
# Individual per-pass predictions are saved; K-averaged metrics computed
# post-hoc from the same set of passes.
#
# Usage:
#   bash scripts/run_eval_scenarios.sh \
#       --config src/configs/factflow_fmri_cross_dino_srcdist_v2.yaml \
#       --ckpt exps/srcdist_v2/checkpoints/best.pt
#
# Optional:
#   --batch_size 32 --device cuda --out_dir results/eval_scenarios
# ==============================================================================

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────
CONFIG=""
CKPT=""
BATCH_SIZE=32
DEVICE=""
OUT_DIR="results/eval_scenarios"
NUM_WORKERS=4

# ── Parse arguments ──────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)      CONFIG="$2";      shift 2 ;;
        --ckpt)        CKPT="$2";        shift 2 ;;
        --batch_size)  BATCH_SIZE="$2";  shift 2 ;;
        --device)      DEVICE="$2";      shift 2 ;;
        --out_dir)     OUT_DIR="$2";     shift 2 ;;
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

mkdir -p "$OUT_DIR"

# CSV output — all results in one file
CSV_OUT="${OUT_DIR}/eval_results.csv"

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
echo "  Output dir: $OUT_DIR"
echo "  CSV output: $CSV_OUT"
echo "  Device:     ${DEVICE:-auto}"
echo "=============================================================="

# ── Scenario 1: Deterministic (μ + ODE, 1 pass) ─────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Scenario 1: deterministic (1 pass)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

$PYTHON src/eval_factflow_fmri.py \
    --config "$CONFIG" \
    --ckpt "$CKPT" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    $DEVICE_ARG \
    --scenario deterministic \
    --output "${OUT_DIR}/deterministic" \
    --csv_out "$CSV_OUT"

echo "  ✓ Scenario 1 complete"

# ── Scenario 2: Perceiver stochastic (10 passes → K=1,5,10) ─────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Scenario 2: perceiver_stochastic (10 passes → K=1,5,10)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

$PYTHON src/eval_factflow_fmri.py \
    --config "$CONFIG" \
    --ckpt "$CKPT" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    $DEVICE_ARG \
    --scenario perceiver_stochastic \
    --max_trials 10 \
    --k_values "1,5,10" \
    --output "${OUT_DIR}/perceiver_stochastic" \
    --csv_out "$CSV_OUT"

echo "  ✓ Scenario 2 complete"

# ── Scenario 3: Flow stochastic (10 passes → K=1,5,10) ──────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Scenario 3: flow_stochastic (10 passes → K=1,5,10)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

$PYTHON src/eval_factflow_fmri.py \
    --config "$CONFIG" \
    --ckpt "$CKPT" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    $DEVICE_ARG \
    --scenario flow_stochastic \
    --max_trials 10 \
    --k_values "1,5,10" \
    --sde_num_steps 250 \
    --output "${OUT_DIR}/flow_stochastic" \
    --csv_out "$CSV_OUT"

echo "  ✓ Scenario 3 complete"

# ── Summary ───────────────────────────────────────────────────────────
echo ""
echo "=============================================================="
echo "All evaluations complete!"
echo "=============================================================="
echo ""
echo "Output structure:"
echo "  ${OUT_DIR}/"
echo "  ├── eval_results.csv              ← all metrics (7 rows)"
echo "  ├── deterministic/"
echo "  │   └── avg_k01.npz"
echo "  ├── perceiver_stochastic/"
echo "  │   ├── passes/pass_00..09.npy    ← individual passes"
echo "  │   ├── avg_k01.npz"
echo "  │   ├── avg_k05.npz"
echo "  │   └── avg_k10.npz"
echo "  └── flow_stochastic/"
echo "      ├── passes/pass_00..09.npy"
echo "      ├── avg_k01.npz"
echo "      ├── avg_k05.npz"
echo "      └── avg_k10.npz"
echo ""

if [[ -f "$CSV_OUT" ]]; then
    echo "Results:"
    column -t -s',' "$CSV_OUT"
fi
