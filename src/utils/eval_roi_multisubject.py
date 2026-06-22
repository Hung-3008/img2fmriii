"""
eval_roi_multisubject.py — Per-ROI fidelity of the multi-subject StimFlow model.
================================================================================
Evaluates a single multi-subject checkpoint (shared trunk + per-subject adapters,
optionally with ROI-stratified feature routing) over all of its subjects and
breaks the per-voxel encoding accuracy down per NSD streams-atlas ROI.

Runs T=1 (one ODE pass per image) over the full 1000-image test set with
``--eval_noise_scale`` (default 0.2). The underlying ``FactFlowEvaluator`` loops
over every subject in the config and writes ``sub<S>/avg_k01.npz``; we then
aggregate per ROI for each subject.

Usage::

    .venv/bin/python src/utils/eval_roi_multisubject.py \\
        --config exps/multi_subject/factflow_ms_sub1257_rfr/config.yaml \\
        --ckpt   exps/multi_subject/factflow_ms_sub1257_rfr/checkpoints/best.pt \\
        --eval_noise_scale 0.2 --output results/roi_multisubject
"""
from __future__ import annotations

import argparse
import os
import sys
from argparse import Namespace
from typing import List

# Ensure src/ is on sys.path (this file lives in src/utils/).
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from omegaconf import OmegaConf  # noqa: E402

from trainer.factflow_evaluator import FactFlowEvaluator  # noqa: E402
from utils.eval_roi_ablation import _per_roi_rows, _write_per_roi_csv  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-subject per-ROI fidelity eval")
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--subjects", default=None,
                    help="comma-separated subset; default = all in config")
    ap.add_argument("--data_dir", default="NSD/data/nsd")
    ap.add_argument("--eval_noise_scale", type=float, default=0.2)
    ap.add_argument("--n_reps", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="")
    ap.add_argument("--output", default="results/roi_multisubject")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)

    eval_args = Namespace(
        config=args.config, ckpt=args.ckpt, batch_size=args.batch_size,
        subject=args.subjects, eval_noise_scale=args.eval_noise_scale,
        num_workers=args.num_workers, device=args.device, output=args.output,
        max_trials=1, k_values="1",
        csv_out=os.path.join(args.output, "eval_global.csv"),
    )
    evaluator = FactFlowEvaluator(eval_args)
    evaluator.evaluate()  # writes <output>/sub<S>/avg_k01.npz for every subject

    # Which subjects were evaluated.
    cfg = OmegaConf.load(args.config)
    if args.subjects:
        subjects = [int(s) for s in args.subjects.split(",")]
    else:
        subjects = [int(s) for s in cfg.data.subjects]

    roi_args = Namespace(data_dir=args.data_dir, n_reps=args.n_reps)
    all_rows: List[dict] = []
    for subject in subjects:
        npz_path = os.path.join(args.output, f"sub{subject}", "avg_k01.npz")
        all_rows.extend(_per_roi_rows(roi_args, "ms_rfr", subject, npz_path))

    per_roi_csv = os.path.join(args.output, "per_roi.csv")
    _write_per_roi_csv(per_roi_csv, all_rows)
    print(f"\n→ {per_roi_csv}")


if __name__ == "__main__":
    main()
