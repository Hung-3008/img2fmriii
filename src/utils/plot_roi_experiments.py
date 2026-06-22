"""
plot_roi_experiments.py — Figures for the per-ROI experiments.
==============================================================
Produces two publication figures from the per-ROI CSVs:

  1. Backbone complementarity (``--gap_csv`` = roi_gap_summary.csv):
     a grouped bar of DINOv2-only vs CLIP-only per-voxel encoding Pearson per
     NSD stream, beside a diverging bar of the gap (DINOv2 - CLIP). Shows the
     structural-feature advantage concentrated in early-to-mid visual cortex and
     vanishing in higher ventral/lateral streams.

  2. Multi-subject per-ROI fidelity (``--ms_csv`` = per_roi.csv):
     per-stream encoding Pearson of the multi-subject StimFlow model (mean +/-
     std across subjects), overlaid on the per-stream noise ceiling so the
     captured fraction is visible.

Usage::

    .venv/bin/python src/utils/plot_roi_experiments.py \\
        --gap_csv results/roi_ablation_full/roi_gap_summary.csv \\
        --ms_csv  results/roi_multisubject/per_roi.csv \\
        --outdir  ACCV_2026_template/images
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Streams ordered early -> higher; pretty labels for the axis.
# Ordered by hierarchy tier: early -> mid-tier streams -> high-tier streams.
HIERARCHY = ["early", "midventral", "midlateral", "midparietal",
             "ventral", "lateral", "parietal"]
PRETTY = {
    "early": "Early", "midventral": "Mid\nventral", "midlateral": "Mid\nlateral",
    "midparietal": "Mid\nparietal", "ventral": "Ventral", "lateral": "Lateral",
    "parietal": "Parietal",
}
C_DINO, C_CLIP = "#2a7fb8", "#e07b39"   # structural / semantic
C_POS, C_NEG = "#2a9d8f", "#e76f51"     # gap favouring dino / clip


def _read_csv(path: str) -> List[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def plot_backbone_gap(gap_csv: str, out: str) -> None:
    rows = {r["roi"]: r for r in _read_csv(gap_csv)}
    rois = [r for r in HIERARCHY if r in rows]
    dino = np.array([float(rows[r]["dino_only"]) for r in rois])
    clip = np.array([float(rows[r]["clip_only"]) for r in rois])
    gap = np.array([float(rows[r]["gap_dino_minus_clip"]) for r in rois])
    labels = [PRETTY[r] for r in rois]
    x = np.arange(len(rois))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.6), width_ratios=[1.45, 1])

    w = 0.4
    ax1.bar(x - w / 2, dino, w, label="DINOv2 only (structural)", color=C_DINO)
    ax1.bar(x + w / 2, clip, w, label="CLIP only (semantic)", color=C_CLIP)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("Encoding Pearson $r$")
    ax1.set_title("(a) Per-stream encoding accuracy")
    ax1.legend(fontsize=8, frameon=False, loc="upper right")
    ax1.set_ylim(0, max(dino.max(), clip.max()) * 1.18)
    ax1.grid(axis="y", ls=":", alpha=0.4)

    colors = [C_POS if g >= 0 else C_NEG for g in gap]
    ax2.bar(x, gap, 0.6, color=colors)
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel(r"Gap  $r_{\mathrm{DINOv2}}-r_{\mathrm{CLIP}}$")
    ax2.set_title("(b) Structural$-$semantic crossover")
    ax2.grid(axis="y", ls=":", alpha=0.4)
    pad = (gap.max() - gap.min()) * 0.12 + 1e-3
    ax2.set_ylim(gap.min() - pad - 0.004, gap.max() + pad + 0.004)
    # crossover annotations
    ax2.text(0.0, gap.max() + pad * 0.3, "structural\nfavoured", fontsize=7,
             ha="center", va="bottom", color=C_POS)
    ax2.text(len(rois) - 1.0, gap.min() - pad * 0.3, "semantic\nfavoured", fontsize=7,
             ha="center", va="top", color=C_NEG)

    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"→ {out}")


def plot_ms_fidelity(ms_csv: str, out: str) -> None:
    rows = _read_csv(ms_csv)
    # roi -> list of per-subject voxel_r / ceil_r
    vr: Dict[str, List[float]] = defaultdict(list)
    cr: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        if r["roi"] in ("ALL", "other"):
            continue
        vr[r["roi"]].append(float(r["voxel_r"]))
        cr[r["roi"]].append(float(r["ceil_r"]))

    rois = [r for r in HIERARCHY if r in vr]
    labels = [PRETTY[r] for r in rois]
    x = np.arange(len(rois))
    vr_mean = np.array([np.mean(vr[r]) for r in rois])
    vr_std = np.array([np.std(vr[r]) for r in rois])
    cr_mean = np.array([np.mean(cr[r]) for r in rois])

    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(x, cr_mean, 0.62, color="#d9d9d9", label="Noise ceiling")
    ax.bar(x, vr_mean, 0.62, yerr=vr_std, capsize=3, color="#3a6ea5",
           error_kw=dict(lw=1, ecolor="#222"),
           label="StimFlow (multi-subject) encoding $r$")
    for xi, v, c in zip(x, vr_mean, cr_mean):
        frac = 100 * v / c if c > 1e-6 else 0
        ax.text(xi, v + 0.012, f"{frac:.0f}%", ha="center", fontsize=7, color="#222")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Encoding Pearson $r$")
    ax.set_title("Per-stream fidelity of multi-subject StimFlow "
                 "(mean $\\pm$ std over subjects 1/2/5/7)")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.set_ylim(0, cr_mean.max() * 1.2)
    ax.grid(axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"→ {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap_csv", default="results/roi_ablation_full/roi_gap_summary.csv")
    ap.add_argument("--ms_csv", default="results/roi_multisubject/per_roi.csv")
    ap.add_argument("--outdir", default="ACCV_2026_template/images")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    if os.path.exists(args.gap_csv):
        plot_backbone_gap(args.gap_csv, os.path.join(args.outdir, "roi_backbone_gap.png"))
    if os.path.exists(args.ms_csv):
        plot_ms_fidelity(args.ms_csv, os.path.join(args.outdir, "roi_ms_fidelity.png"))


if __name__ == "__main__":
    main()
