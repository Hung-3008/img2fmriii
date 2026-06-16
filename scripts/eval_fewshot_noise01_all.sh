#!/usr/bin/env bash
# Re-evaluate the 4 trained few-shot best.pt checkpoints at noise_scale=0.1
# (Trials in {1,5}) WITHOUT re-adapting. Appends to a single CSV.
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true

OUT=exps/cross_subj/results_noise0.1.csv
rm -f "$OUT"

# held-out subject -> trunk tag
HELD=(1 2 5 7)
TAGS=(sub257 sub157 sub127 sub125)

for i in "${!HELD[@]}"; do
  held="${HELD[$i]}"
  tag="${TAGS[$i]}"
  trunk="exps/multi_subject/factflow_ms_${tag}"
  fs="exps/cross_subj/fewshot_sub${held}_from${tag#sub}"
  echo "=================================================================="
  echo " [$((i+1))/4] held-out ${held}  (trunk ${tag})   ($(date '+%F %T'))"
  echo "=================================================================="
  python src/eval_fewshot_noise.py \
    --config "${trunk}/config.yaml" \
    --trunk_ckpt "${trunk}/checkpoints/best.pt" \
    --fewshot_ckpt "${fs}/checkpoints/best.pt" \
    --held_out "${held}" \
    --noise_scales 0.1 --trials 1 5 \
    --out "${OUT}"
done

echo "ALL DONE -> ${OUT}"
