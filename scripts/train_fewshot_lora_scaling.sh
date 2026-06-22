#!/usr/bin/env bash
# LoRA few-shot data-scaling: adapt each held-out subject at HOURS in {1,2,3,4}
# with LoRA on the last trunk block(s) (readout + low-rank trunk residual).
#   - 50 epochs, LR=3e-3, noise_scale=0.2, Trials {1,5}
#   - LoRA: rank 8, last 1 block, alpha 16 (env-overridable)
# After each (subject,hours) run, parse its FINAL TEST and append to a master CSV.
# Hours-major loop so each data level completes across ALL subjects first
# (useful partial curve if interrupted).
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true

EPOCHS=${EPOCHS:-50}
LR=${LR:-3e-3}
HOURS_LIST=(${HOURS_LIST:-1 2 3 4})
LORA_BLOCKS=${LORA_BLOCKS:-1}
LORA_RANK=${LORA_RANK:-8}
LORA_ALPHA=${LORA_ALPHA:-16}
OUT_CSV=exps/cross_subj/results_lora_noise0.2.csv
LOGDIR=results/fewshot_lora_scaling
mkdir -p "$LOGDIR"

# held-out subject -> trunk tag
HELD=(1 2 5 7)
TAGS=(sub257 sub157 sub127 sub125)

for h in "${HOURS_LIST[@]}"; do
  for i in "${!HELD[@]}"; do
    held="${HELD[$i]}"
    tag="${TAGS[$i]}"
    trunk="exps/multi_subject/factflow_ms_${tag}"
    exp_name="fewshot_sub${held}_from${tag#sub}_${h}h_lora"
    exp_dir="exps/cross_subj/${exp_name}"
    echo "=================================================================="
    echo " held-out ${held} (trunk ${tag})  HOURS=${h}  LoRA r${LORA_RANK}/b${LORA_BLOCKS}   ($(date '+%F %T'))"
    echo "=================================================================="
    python src/train_factflow_multisubject.py \
      --config "${trunk}/config.yaml" \
      --fewshot_held_out "${held}" \
      --fewshot_pretrained "${trunk}/checkpoints/best.pt" \
      --fewshot_hours "${h}" --fewshot_val_trials 250 \
      --fewshot_epochs "${EPOCHS}" --adapt_lr "${LR}" \
      --noise_scale 0.2 --trials 1 5 \
      --lora --lora_blocks "${LORA_BLOCKS}" \
      --lora_rank "${LORA_RANK}" --lora_alpha "${LORA_ALPHA}" \
      --exps_dir exps/cross_subj --exp_name "${exp_name}" \
      2>&1 | tee "${LOGDIR}/${exp_name}.log"

    # Append FINAL TEST rows for this run to the master CSV.
    python src/parse_fewshot_test.py "${exp_dir}/train.log" \
      "${OUT_CSV}" "${held}" "${tag#sub}" "${h}" \
      || echo "[warn] parse failed for ${exp_name}"
    echo "[done] ${exp_name}  ($(date '+%F %T'))"
  done
done

echo "ALL DONE -> ${OUT_CSV}"
