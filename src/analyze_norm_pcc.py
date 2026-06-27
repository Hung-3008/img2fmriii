#!/usr/bin/env python3
"""Normalization-dependence analysis for the SynBrain baseline (supplement S5).

Shows that voxel-level per-image Pearson is inflated when the per-voxel mean is
*retained* (SynBrain's released preprocessing divides betas by a constant 2000
without removing the per-voxel mean), and that a stimulus-agnostic "mean-only"
predictor already scores high under that regime but collapses under z-scoring.

The decisive bar -- mean-only | mean-retained -- is computed here from the real
NSD betas (subject 1, the only subject for which the mean-retained `*_scale_*`
file is on disk). SynBrain's own bars are taken from our reproduction:
mean-retained = originally reported 0.687, z-scored = 0.340 (main-paper number).
SynBrain's synthesized betas are not cached on disk, so its bars are documented
constants rather than recomputed here; the mean-only control is what makes the
argument, and it is real.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

V = 15724
FMRI = "NSD/data/nsd/subj01/fmri/"
OUT = "ACCV_2026_template/images/supp_norm_pcc.pdf"

# --- SynBrain reproduction numbers (documented; betas not cached on disk) ---
SYN_MR, SYN_ZS = 0.687, 0.340          # mean-retained (reported) vs z-scored (ours)
MINDSIM, STIMFLOW = 0.346, 0.438       # reference lines (Table 1, T=1)


def per_image_pcc(pred, gt):
    """Per-image, across-voxel Pearson, averaged over images. pred,gt: [N,V]."""
    p = pred - pred.mean(1, keepdims=True)
    g = gt - gt.mean(1, keepdims=True)
    num = (p * g).sum(1)
    den = np.sqrt((p ** 2).sum(1) * (g ** 2).sum(1)) + 1e-8
    return float((num / den).mean())


def mean_only_bars():
    """Compute the mean-only control on real sub-1 betas."""
    tr = np.load(FMRI + "nsd_train_fmri_scale_sub1.npy").reshape(-1, V)   # mean-retained
    te = np.load(FMRI + "nsd_test_fmri_scale_sub1.npy").mean(1)           # avg 3 trials
    mu = tr.mean(0, keepdims=True)                                        # per-voxel train mean
    pred = np.broadcast_to(mu, te.shape)
    mr = per_image_pcc(pred, te)
    # z-scored mean-only: zscore(mu) with the same train stats is the all-zero
    # vector -> correlation is undefined / chance. Report 0.
    zs = 0.0
    return mr, zs


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    mo_mr, mo_zs = mean_only_bars()
    print(f"[real] mean-only | mean-retained PCC = {mo_mr:.4f}")
    print(f"[real] mean-only | z-scored      PCC = {mo_zs:.4f}")
    print(f"[repro] SynBrain  | mean-retained PCC = {SYN_MR:.3f}")
    print(f"[repro] SynBrain  | z-scored      PCC = {SYN_ZS:.3f}")

    labels = ["SynBrain", "mean-only\n(no stimulus)"]
    mr = [SYN_MR, mo_mr]
    zs = [SYN_ZS, mo_zs]
    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    b1 = ax.bar(x - w / 2, mr, w, label="mean-retained (÷2000, mean kept)",
                color="#d1495b")
    b2 = ax.bar(x + w / 2, zs, w, label="z-scored (standard)", color="#2e7d9e")
    ax.axhline(MINDSIM, ls="--", lw=1, c="gray", label="MindSimulator (pub., across-voxel)")
    ax.axhline(STIMFLOW, ls=":", lw=1.2, c="k", label="StimFlow (across-voxel)")
    for bars in (b1, b2):
        for r in bars:
            h = r.get_height()
            ax.annotate(f"{h:.2f}", (r.get_x() + r.get_width() / 2, h),
                        ha="center", va="bottom", fontsize=7,
                        xytext=(0, 1), textcoords="offset points")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("per-image across-voxel Pearson")
    ax.set_ylim(0, 1.06)
    ax.legend(fontsize=6.8, frameon=False, loc="upper center", ncol=2,
              columnspacing=1.0, handlelength=1.4)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print("saved", OUT)


if __name__ == "__main__":
    main()
