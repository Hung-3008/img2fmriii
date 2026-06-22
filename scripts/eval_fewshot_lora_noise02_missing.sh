#!/usr/bin/env bash
# Evaluate MISSING few-shot LoRA checkpoints at noise_scale=0.2
# Missing from results_lora_noise0.2.csv: Sub7 3h, all subjects 4h
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true

OUT=exps/cross_subj/results_lora_noise0.2.csv

# Sub7 3h LoRA
echo "=================================================================="
echo " [1/5] Sub7 3h LoRA  ($(date '+%F %T'))"
echo "=================================================================="
python src/eval_fewshot_noise.py \
  --config exps/multi_subject/factflow_ms_sub125/config.yaml \
  --trunk_ckpt exps/multi_subject/factflow_ms_sub125/checkpoints/best.pt \
  --fewshot_ckpt exps/cross_subj/fewshot_sub7_from125_3h_lora/checkpoints/best.pt \
  --held_out 7 --noise_scales 0.2 --trials 1 5 \
  --out "${OUT}"

# Sub1 4h LoRA
echo "=================================================================="
echo " [2/5] Sub1 4h LoRA  ($(date '+%F %T'))"
echo "=================================================================="
python src/eval_fewshot_noise.py \
  --config exps/multi_subject/factflow_ms_sub257/config.yaml \
  --trunk_ckpt exps/multi_subject/factflow_ms_sub257/checkpoints/best.pt \
  --fewshot_ckpt exps/cross_subj/fewshot_sub1_from257_4h_lora/checkpoints/best.pt \
  --held_out 1 --noise_scales 0.2 --trials 1 5 \
  --out "${OUT}"

# Sub2 4h LoRA
echo "=================================================================="
echo " [3/5] Sub2 4h LoRA  ($(date '+%F %T'))"
echo "=================================================================="
python src/eval_fewshot_noise.py \
  --config exps/multi_subject/factflow_ms_sub157/config.yaml \
  --trunk_ckpt exps/multi_subject/factflow_ms_sub157/checkpoints/best.pt \
  --fewshot_ckpt exps/cross_subj/fewshot_sub2_from157_4h_lora/checkpoints/best.pt \
  --held_out 2 --noise_scales 0.2 --trials 1 5 \
  --out "${OUT}"

# Sub5 4h LoRA
echo "=================================================================="
echo " [4/5] Sub5 4h LoRA  ($(date '+%F %T'))"
echo "=================================================================="
python src/eval_fewshot_noise.py \
  --config exps/multi_subject/factflow_ms_sub127/config.yaml \
  --trunk_ckpt exps/multi_subject/factflow_ms_sub127/checkpoints/best.pt \
  --fewshot_ckpt exps/cross_subj/fewshot_sub5_from127_4h_lora/checkpoints/best.pt \
  --held_out 5 --noise_scales 0.2 --trials 1 5 \
  --out "${OUT}"

# Sub7 4h LoRA
echo "=================================================================="
echo " [5/5] Sub7 4h LoRA  ($(date '+%F %T'))"
echo "=================================================================="
python src/eval_fewshot_noise.py \
  --config exps/multi_subject/factflow_ms_sub125/config.yaml \
  --trunk_ckpt exps/multi_subject/factflow_ms_sub125/checkpoints/best.pt \
  --fewshot_ckpt exps/cross_subj/fewshot_sub7_from125_4h_lora/checkpoints/best.pt \
  --held_out 7 --noise_scales 0.2 --trials 1 5 \
  --out "${OUT}"

echo "ALL DONE -> ${OUT}"
