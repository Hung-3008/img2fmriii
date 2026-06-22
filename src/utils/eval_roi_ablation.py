"""
eval_roi_ablation.py — Per-ROI backbone ablation (DINOv2-only vs CLIP-only).
============================================================================
Decisive experiment for the "CLIP-only conditioning is a systematic semantic
bias" claim: evaluate the trained single-backbone ablation checkpoints and break
the encoding voxel_r down per NSD streams-atlas ROI. If the thesis holds, the
DINOv2-only advantage should concentrate in *early* visual cortex and shrink
toward higher (ventral/lateral) regions where CLIP's semantics suffice.

For each (model, subject) it runs the standard FactFlow ODE sampler over the
first ``--n_images`` test images (default 100), a single pass each
(``--eval_noise_scale`` default 0.2), then computes per-voxel Pearson r and
aggregates per ROI (plus the noise-ceiling correlation for context).

Outputs (under ``--output``, default results/roi_ablation):
  * <model>/sub<S>/avg_k<K>.npz   — preds/targets/voxel_r (from the evaluator)
  * per_roi.csv                   — one row per (model, subject, roi)
  * roi_gap_summary.csv           — per ROI: dino vs clip voxel_r + gap, averaged
                                    over subjects, ordered along the visual
                                    hierarchy (early → ventral → lateral →
                                    parietal)

Usage::

    .venv/bin/python src/utils/eval_roi_ablation.py \\
        --eval_noise_scale 0.2 --max_trials 100 \\
        --subjects 1,2,5,7 --output results/roi_ablation
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from argparse import Namespace
from typing import Dict, List

import numpy as np
from torch.utils.data import DataLoader, Subset

# Ensure src/ is on sys.path (this file lives in src/utils/).
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from trainer.factflow_evaluator import FactFlowEvaluator  # noqa: E402
from utils.metrics import compute_voxel_reliability  # noqa: E402


class SubsetEvaluator(FactFlowEvaluator):
    """FactFlowEvaluator restricted to the first ``n_images`` test images."""

    def __init__(self, args: Namespace) -> None:
        self.n_images_eval = int(getattr(args, "n_images", 0) or 0)
        super().__init__(args)

    def _set_subject(self, subject: int) -> None:
        super()._set_subject(subject)
        n = self.n_images_eval
        if n and n < len(self.test_ds):
            subset = Subset(self.test_ds, list(range(n)))
            self.test_loader = DataLoader(
                subset, batch_size=self.args.batch_size, shuffle=False,
                num_workers=self.args.num_workers, pin_memory=True,
            )
            # Metrics infer n_images = N // n_reps; with avg_reps n_reps=1.
            self.test_ds.n_images = n

# Streams-atlas ROIs ordered along the visual hierarchy (early → higher).
# Any ROI not listed here (e.g. "other") is appended afterwards.
HIERARCHY_ORDER = [
    "early", "midventral", "midlateral", "midparietal",
    "ventral", "lateral", "parietal",
]


def _data_root(args) -> str:
    return args.data_dir


def _ckpt_path(args, model: str, subject: int) -> str:
    return os.path.join(
        args.exp_root, f"{model}_sub{subject}", "checkpoints", args.ckpt_name
    )


def _config_path(args, model: str, subject: int) -> str:
    return os.path.join(args.exp_root, f"{model}_sub{subject}", "config.yaml")


def _ceiling_r(args, subject: int) -> np.ndarray:
    """Per-voxel correlation noise ceiling (anatomical order), cached per subject."""
    fmri = os.path.join(
        _data_root(args), f"subj0{subject}", "fmri",
        f"nsd_test_fmri_zscore_sub{subject}.npy",
    )
    test = np.load(fmri)  # (N, reps, V)
    nc_var = compute_voxel_reliability(test, args.n_reps)  # (V,) in [0, 1]
    return np.sqrt(np.clip(nc_var, 0.0, 1.0))


def _roi_meta(args, subject: int):
    meta = np.load(
        os.path.join(_data_root(args), f"subj0{subject}",
                     f"roi_meta_sub{subject}.npz"),
        allow_pickle=True,
    )
    labels = meta["roi_labels"].astype(int)
    roi_ids = meta["roi_ids"].astype(int)
    roi_names = [str(x) for x in meta["roi_names"]]
    return labels, roi_ids, roi_names


def _run_one(args, model: str, subject: int) -> str:
    """Run the evaluator for one (model, subject); return the avg_kK.npz path."""
    out_dir = os.path.join(args.output, model)
    eval_args = Namespace(
        config=_config_path(args, model, subject),
        ckpt=_ckpt_path(args, model, subject),
        batch_size=args.batch_size,
        subject=None,                       # single-subject configs
        eval_noise_scale=args.eval_noise_scale,
        num_workers=args.num_workers,
        device=args.device,
        output=out_dir,
        max_trials=1,                       # one pass per image
        k_values="1",
        n_images=args.n_images,
        csv_out=os.path.join(args.output, "eval_global.csv"),
    )
    evaluator = SubsetEvaluator(eval_args)
    evaluator.evaluate()
    return os.path.join(out_dir, f"sub{subject}", "avg_k01.npz")


def _per_roi_rows(args, model: str, subject: int, npz_path: str) -> List[dict]:
    voxel_r = np.asarray(np.load(npz_path)["voxel_r"], dtype=np.float64)  # (V,)
    labels, roi_ids, roi_names = _roi_meta(args, subject)
    ceil_r = _ceiling_r(args, subject)
    assert labels.shape[0] == voxel_r.shape[0] == ceil_r.shape[0]

    rows = []
    for rid, name in zip(roi_ids, roi_names):
        m = labels == rid
        n = int(m.sum())
        if n == 0:
            continue
        vr = float(voxel_r[m].mean())
        cr = float(ceil_r[m].mean())
        rows.append({
            "model": model, "subject": subject, "roi": name, "nvox": n,
            "voxel_r": vr, "ceil_r": cr,
            "captured": (vr / cr if cr > 1e-6 else float("nan")),
        })
    rows.append({
        "model": model, "subject": subject, "roi": "ALL",
        "nvox": int(voxel_r.shape[0]),
        "voxel_r": float(voxel_r.mean()), "ceil_r": float(ceil_r.mean()),
        "captured": float(voxel_r.mean() / ceil_r.mean()),
    })
    return rows


def _write_per_roi_csv(path: str, rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["model", "subject", "roi", "nvox", "voxel_r",
                           "ceil_r", "captured"]
        )
        w.writeheader()
        for r in rows:
            w.writerow({
                **r,
                "voxel_r": f"{r['voxel_r']:.6f}",
                "ceil_r": f"{r['ceil_r']:.6f}",
                "captured": f"{r['captured']:.6f}",
            })


def _write_gap_summary(path: str, rows: List[dict], models: List[str]) -> None:
    """Per ROI: mean voxel_r for each model + (dino − clip) gap, over subjects."""
    if not ("dino_only" in models and "clip_only" in models):
        return
    rois = sorted({r["roi"] for r in rows})
    ordered = [r for r in HIERARCHY_ORDER if r in rois]
    ordered += [r for r in rois if r not in ordered and r != "ALL"]
    if "ALL" in rois:
        ordered.append("ALL")

    def mean_for(model: str, roi: str) -> float:
        vals = [r["voxel_r"] for r in rows
                if r["model"] == model and r["roi"] == roi]
        return float(np.mean(vals)) if vals else float("nan")

    print(f"\n{'ROI':<13}{'dino_only':>11}{'clip_only':>11}{'gap(d-c)':>11}")
    print("-" * 46)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["roi", "dino_only", "clip_only", "gap_dino_minus_clip"])
        for roi in ordered:
            d = mean_for("dino_only", roi)
            c = mean_for("clip_only", roi)
            gap = d - c
            w.writerow([roi, f"{d:.6f}", f"{c:.6f}", f"{gap:.6f}"])
            print(f"{roi:<13}{d:>11.4f}{c:>11.4f}{gap:>+11.4f}")
    print(f"\n→ {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-ROI backbone ablation eval")
    ap.add_argument("--exp_root", default="exps/ablations")
    ap.add_argument("--models", default="dino_only,clip_only",
                    help="comma-separated ablation model tags")
    ap.add_argument("--subjects", default="1,2,5,7")
    ap.add_argument("--data_dir", default="NSD/data/nsd")
    ap.add_argument("--ckpt_name", default="best.pt")
    ap.add_argument("--eval_noise_scale", type=float, default=0.2)
    ap.add_argument("--n_images", type=int, default=100,
                    help="number of test images per subject (single pass each)")
    ap.add_argument("--n_reps", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="")
    ap.add_argument("--output", default="results/roi_ablation")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    subjects = [int(s) for s in args.subjects.split(",") if s.strip()]
    os.makedirs(args.output, exist_ok=True)

    all_rows: List[dict] = []
    for model in models:
        for subject in subjects:
            print("=" * 60)
            print(f"EVAL  model={model}  subject={subject}  "
                  f"noise={args.eval_noise_scale}  n_images={args.n_images}")
            print("=" * 60)
            npz_path = _run_one(args, model, subject)
            all_rows.extend(_per_roi_rows(args, model, subject, npz_path))

    per_roi_csv = os.path.join(args.output, "per_roi.csv")
    _write_per_roi_csv(per_roi_csv, all_rows)
    print(f"\n→ {per_roi_csv}")
    _write_gap_summary(
        os.path.join(args.output, "roi_gap_summary.csv"), all_rows, models
    )


if __name__ == "__main__":
    main()
