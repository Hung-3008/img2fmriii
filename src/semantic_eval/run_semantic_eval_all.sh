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
      --do_gt --enhanced --save_pngs --device "$DEVICE" --max_images "$MAX_IMAGES" \
      2>&1 | tee "$LOG_DIR/sub${S}.log"
done

echo ""
echo "=== ALL SUBJECTS DONE ($(date '+%F %T')) ==="

# ---- summary table: base + enhanced, per run + per-tag average ----
$PY - <<'PYEOF'
import json, glob, os, collections
KS = ["PixCorr", "SSIM", "Alex_2", "Alex_5", "Incep", "CLIP", "Eff", "SwAV"]
def line(name, m):
    return (f"{name:<20}" + "".join(f"{m[k]:>7.3f}" for k in KS))
for kind in ("base", "enhanced"):
    rows = []
    for d in sorted(glob.glob("results/semantic_eval/sub*_*")):
        jf = os.path.join(d, f"{os.path.basename(d).split('_')[-1]}_{kind}_metrics.json")
        if os.path.exists(jf):
            rows.append((f"{os.path.basename(d)}", json.load(open(jf))))
    if not rows:
        continue
    print(f"\n================ SUMMARY ({kind}) ================")
    print(f"{'run':<20}" + "".join(f"{k:>7}" for k in ["PixC","SSIM","Alex2","Alex5","Incep","CLIP","Eff","SwAV"]))
    for name, m in rows:
        print(line(name, m))
    agg = collections.defaultdict(list)
    for name, m in rows:
        agg[name.split("_")[-1]].append(m)
    print(f"---- average over subjects ({kind}) ----")
    for tag, ms in sorted(agg.items()):
        avg = {k: sum(x[k] for x in ms) / len(ms) for k in KS}
        print(line(f"avg_{tag}", avg))
PYEOF
