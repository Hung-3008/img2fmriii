#!/usr/bin/env bash
# Few-shot (1 hour) adaptation to each held-out subject, run as a REAL training
# loop with per-epoch validation + best-checkpoint selection (integrated into the
# training code: src/train_factflow_multisubject.py --fewshot_held_out).
#   - 750 adaptation trials (1h, single-rep) + 250 disjoint val trials
#   - small #epochs, eval EVERY epoch on val, keep best by profile_r
#   - final TEST eval on best adapter: noise_scale=0.2, Trials in {1,5}
# Run sequentially (designed for tmux). One subject failing does not stop the rest.
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true

EPOCHS=${EPOCHS:-80}
LR=${LR:-3e-3}
OUT=results/fewshot_1h_trained
mkdir -p "$OUT"

# checkpoint dir tag -> held-out subject
TAGS=(sub125 sub127 sub157 sub257)
HELD=(7 5 2 1)

for i in "${!TAGS[@]}"; do
  tag="${TAGS[$i]}"
  held="${HELD[$i]}"
  ck="exps/multi_subject/factflow_ms_${tag}"
  echo "=================================================================="
  echo " [$((i+1))/4] ${tag}  ->  held-out subject ${held}   ($(date '+%F %T'))"
  echo "=================================================================="
  python src/train_factflow_multisubject.py \
    --config "${ck}/config.yaml" \
    --fewshot_held_out "${held}" \
    --fewshot_pretrained "${ck}/checkpoints/best.pt" \
    --fewshot_hours 1.0 --fewshot_val_trials 250 \
    --fewshot_epochs "${EPOCHS}" --adapt_lr "${LR}" \
    --noise_scale 0.2 --trials 1 5 \
    --exp_name "fewshot_sub${held}_from${tag#sub}" \
    2>&1 | tee "${OUT}/${tag}_held${held}.log"
  echo "[done] ${tag} -> subject ${held}  ($(date '+%F %T'))"
done

echo "ALL DONE -> ${OUT}/  (per-epoch history in exps/fewshot_sub*/history.csv)"
