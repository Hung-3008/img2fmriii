"""
analyze_roi.py — Per-ROI breakdown of encoding voxel_r vs noise ceiling.
=========================================================================
Loads an eval ``avg_k*.npz`` (per-voxel ``voxel_r`` + ``targets``, anatomical
order) and ``roi_meta_sub{S}.npz`` (per-voxel ROI labels, anatomical order),
then reports, per streams-atlas ROI:

  * nvox      — number of voxels in the ROI
  * voxel_r   — mean per-voxel encoding Pearson r (how well we predict it)
  * ceil_r    — mean per-voxel correlation noise ceiling = sqrt(noise-ceiling
                variance fraction); the best r any model could reach there
  * captured% — voxel_r / ceil_r; fraction of the *achievable* correlation we
                actually capture

A ROI with a HIGH ceil_r (lots of explainable signal) but a LOW captured%
is being under-served by the current features → prime candidate for a new
orthogonal feature (low-level / retinotopy / color / depth ...).

Usage::

    .venv/bin/python src/analyze_roi.py \\
        --npz results/roi_dino4p_gabor/avg_k01.npz \\
        --roi_meta NSD/data/nsd/subj01/roi_meta_sub1.npz \\
        --test_fmri NSD/data/nsd/subj01/fmri/nsd_test_fmri_zscore_sub1.npy
"""
import argparse
import os
import sys

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from utils.metrics import compute_voxel_reliability


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-ROI voxel_r vs noise ceiling")
    ap.add_argument("--npz", required=True,
                    help="eval avg_k*.npz containing per-voxel voxel_r (anatomical order)")
    ap.add_argument("--roi_meta", required=True,
                    help="roi_meta_sub{S}.npz with roi_labels/roi_ids/roi_names")
    ap.add_argument("--test_fmri", required=True,
                    help="raw test betas (N_images, n_reps, V) for the noise ceiling")
    ap.add_argument("--n_reps", type=int, default=3)
    ap.add_argument("--nc_thresh", type=float, default=0.1,
                    help="report count of reliable voxels with noise-ceiling var > this")
    args = ap.parse_args()

    d = np.load(args.npz)
    voxel_r = np.asarray(d["voxel_r"], dtype=np.float64)        # (V,) anatomical
    V = voxel_r.shape[0]

    meta = np.load(args.roi_meta, allow_pickle=True)
    labels = meta["roi_labels"].astype(int)                     # (V,) anatomical
    roi_ids = meta["roi_ids"].astype(int)
    roi_names = [str(x) for x in meta["roi_names"]]
    assert labels.shape[0] == V, f"roi_labels {labels.shape} != voxel_r {V}"

    # Per-voxel noise ceiling (variance fraction) → correlation ceiling sqrt(.)
    test = np.load(args.test_fmri)                              # (N, reps, V)
    nc_var = compute_voxel_reliability(test, args.n_reps)       # (V,) in [0,1]
    ceil_r = np.sqrt(np.clip(nc_var, 0.0, 1.0))                 # (V,) correlation ceiling

    hdr = f"{'ROI':<13}{'nvox':>7}{'voxel_r':>10}{'ceil_r':>9}{'captured':>11}{'reliable':>10}"
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for rid, name in zip(roi_ids, roi_names):
        m = labels == rid
        n = int(m.sum())
        if n == 0:
            continue
        vr = float(voxel_r[m].mean())
        cr = float(ceil_r[m].mean())
        cap = vr / cr if cr > 1e-6 else float("nan")
        n_rel = int((nc_var[m] > args.nc_thresh).sum())
        rows.append((name, n, vr, cr, cap, n_rel))

    # Print sorted by captured fraction ascending (most under-served first)
    for name, n, vr, cr, cap, n_rel in sorted(rows, key=lambda r: r[4]):
        print(f"{name:<13}{n:>7}{vr:>10.4f}{cr:>9.4f}{cap*100:>10.1f}%{n_rel:>10}")

    vr_all = float(voxel_r.mean())
    cr_all = float(ceil_r.mean())
    n_rel_all = int((nc_var > args.nc_thresh).sum())
    print("-" * len(hdr))
    print(f"{'ALL':<13}{V:>7}{vr_all:>10.4f}{cr_all:>9.4f}"
          f"{vr_all / cr_all * 100:>10.1f}%{n_rel_all:>10}")


if __name__ == "__main__":
    main()
