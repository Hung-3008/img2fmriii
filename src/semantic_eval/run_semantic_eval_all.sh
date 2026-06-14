#!/usr/bin/env bash
# Semantic-level evaluation (MindEye2 decode) for StimFlow synthesized fMRI.
# Runs all 4 NSD subjects (1,2,5,7); for each subject loads the MindEye2 + SDXL
# unCLIP decoders ONCE and decodes:
#   - gt  : real renorm betas (upper bound / sanity)
#   - T1  : StimFlow single-sample  (avg_k01.npz)
#   - T5  : StimFlow 5-sample mean  (avg_k05.npz)
# Saves reconstructed PNGs (+ GT PNGs) and a *_metrics.json per run, then prints
# a summary table.
#
# Usage:
#   bash reproduces/MindEyeV2/src/run_semantic_eval_all.sh [device] [max_images]
#   e.g.  bash reproduces/MindEyeV2/src/run_semantic_eval_all.sh cuda:0
#   (run inside tmux; full 1000-image sweep takes several hours)

set -u
cd /media/hung/data1/codes/img2fmri_02/img2fmriii || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=.venv/bin/python
SCRIPT=src/semantic_eval/stimflow_semantic_eval.py
SYNTH_DIR=results/ms_sub1257_rfr_eval
OUT_ROOT=results/semantic_eval
LOG_DIR="$OUT_ROOT/logs"
SUBJECTS=(1 2 5 7)
DEVICE="${1:-cuda:0}"
MAX_IMAGES="${2:--1}"          # -1 = all 1000 test images
mkdir -p "$LOG_DIR"

echo "=== StimFlow semantic eval: subjects ${SUBJECTS[*]} | device $DEVICE | max_images $MAX_IMAGES ==="
for S in "${SUBJECTS[@]}"; do
  echo ""
  echo "############################################################"
  echo "# Subject $S   ($(date '+%F %T'))"
  echo "############################################################"
  $PY "$SCRIPT" --batch --subject "$S" \
      --synth_dir "$SYNTH_DIR" --out_root "$OUT_ROOT" \
      --do_gt --save_pngs --device "$DEVICE" --max_images "$MAX_IMAGES" \
      2>&1 | tee "$LOG_DIR/sub${S}.log"
done

echo ""
echo "=== ALL SUBJECTS DONE ($(date '+%F %T')) ==="

# ---- summary table over all *_metrics.json ----
$PY - <<'PYEOF'
import json, glob, os
rows = []
for d in sorted(glob.glob("results/semantic_eval/sub*_*")):
    js = glob.glob(os.path.join(d, "*_metrics.json"))
    if not js:
        continue
    m = json.load(open(js[0]))
    rows.append((os.path.basename(d), m))
if rows:
    print("\n==================== SUMMARY ====================")
    print(f"{'run':<12}{'PixCorr':>8}{'SSIM':>7}{'Alex2':>7}{'Alex5':>7}"
          f"{'Incep':>7}{'CLIP':>7}{'Eff':>7}{'SwAV':>7}")
    for name, m in rows:
        print(f"{name:<12}{m['PixCorr']:>8.3f}{m['SSIM']:>7.3f}{m['Alex_2']:>7.3f}"
              f"{m['Alex_5']:>7.3f}{m['Incep']:>7.3f}{m['CLIP']:>7.3f}"
              f"{m['Eff']:>7.3f}{m['SwAV']:>7.3f}")
    # averages across subjects per tag (T1/T5/gt)
    import collections
    agg = collections.defaultdict(list)
    for name, m in rows:
        tag = name.split("_")[-1]
        agg[tag].append(m)
    print("\n---- average over subjects ----")
    for tag, ms in sorted(agg.items()):
        avg = {k: sum(x[k] for x in ms) / len(ms) for k in ms[0]}
        print(f"{tag:<12}{avg['PixCorr']:>8.3f}{avg['SSIM']:>7.3f}{avg['Alex_2']:>7.3f}"
              f"{avg['Alex_5']:>7.3f}{avg['Incep']:>7.3f}{avg['CLIP']:>7.3f}"
              f"{avg['Eff']:>7.3f}{avg['SwAV']:>7.3f}")
PYEOF
