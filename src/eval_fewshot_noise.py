"""
eval_fewshot_noise.py
=====================
Re-evaluate an ALREADY-adapted few-shot checkpoint (the best.pt produced by
``factflow_fewshot_trainer.py``) at one or more inference noise scales — WITHOUT
re-adapting. Reuses FactFlowFewShotTrainer for identical model / transport /
sampler / metric machinery, then overwrites the held-out adapter with the
trained best.pt and scores the real test set (rep-averaged GT).

Appends one row per (noise_scale, trials) to a CSV.

Usage::

    python src/eval_fewshot_noise.py \
        --config exps/multi_subject/factflow_ms_sub125/config.yaml \
        --trunk_ckpt exps/multi_subject/factflow_ms_sub125/checkpoints/best.pt \
        --fewshot_ckpt exps/cross_subj/fewshot_sub7_from125/checkpoints/best.pt \
        --held_out 7 --noise_scales 0.1 --trials 1 5 \
        --out exps/cross_subj/fewshot_noise0.1.csv
"""

import argparse
import csv
import os
import sys

import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from argparse import Namespace

from trainer.factflow_fewshot_trainer import FactFlowFewShotTrainer


def main() -> None:
    ap = argparse.ArgumentParser(description="Eval adapted few-shot ckpt at noise scales")
    ap.add_argument("--config", required=True)
    ap.add_argument("--trunk_ckpt", required=True,
                    help="Shared-trunk multi-subject best.pt (used to build/seed the model)")
    ap.add_argument("--fewshot_ckpt", required=True,
                    help="The trained few-shot best.pt to evaluate")
    ap.add_argument("--held_out", type=int, required=True)
    ap.add_argument("--noise_scales", type=float, nargs="+", default=[0.1])
    ap.add_argument("--trials", type=int, nargs="+", default=[1, 5])
    ap.add_argument("--out", required=True, help="CSV path (appended to)")
    ap.add_argument("--device", default="")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Build the trainer purely to reuse its model/data/eval machinery.
    targs = Namespace(
        config=args.config,
        fewshot_held_out=args.held_out,
        fewshot_pretrained=args.trunk_ckpt,
        fewshot_hours=1.0,
        fewshot_val_trials=250,
        fewshot_epochs=1,
        adapt_lr=3e-3,
        adapt_bs=32,
        adapt_wd=0.0,
        noise_scale=0.2,
        trials=args.trials,
        no_warm_start=False,
        exp_name="_eval_noise_tmp",
        exps_dir="/tmp",
        device=args.device,
        seed=args.seed,
    )
    trainer = FactFlowFewShotTrainer(targs)

    # Overwrite with the ALREADY-adapted few-shot checkpoint (full wrapper state,
    # incl. the trained held-out adapter at the last index).
    ckpt = torch.load(args.fewshot_ckpt, map_location=trainer.device)
    state = ckpt["model"] if "model" in ckpt else ckpt
    missing, unexpected = trainer.wrapper.load_state_dict(state, strict=False)
    trainer.logger.info("[eval-load] %s  missing=%d unexpected=%d",
                        args.fewshot_ckpt, len(missing), len(unexpected))

    trunk_tag = "".join(str(s) for s in trainer.trunk_subjects)
    new_file = not os.path.exists(args.out)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["held_out", "trunk", "noise_scale", "trials",
                        "mse", "profile_r", "voxel_r", "cosine"])
        for ns in args.noise_scales:
            for t in args.trials:
                m = trainer._evaluate(trainer.test_ds, n_inf_trials=t, noise_scale=ns)
                row = [args.held_out, trunk_tag, f"{ns:.2f}", t,
                       f"{m['mse']:.4f}", f"{m['profile_r']:.4f}",
                       f"{m['voxel_r']:.4f}", f"{m['cosine']:.4f}"]
                w.writerow(row)
                f.flush()
                trainer.logger.info(
                    "held=%d noise=%.2f trials=%d -> mse=%.4f profile_r=%.4f "
                    "voxel_r=%.4f cosine=%.4f", args.held_out, ns, t,
                    m["mse"], m["profile_r"], m["voxel_r"], m["cosine"])
    print(f"[done] appended to {args.out}")


if __name__ == "__main__":
    main()
