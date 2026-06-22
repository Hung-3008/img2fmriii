#!/usr/bin/env python3
"""Concept-selective cortical FLATMAP from StimFlow-synthesized fMRI (pycortex).

Reproduces MindSimulator Fig.8 style: a Pycortex butterfly flatmap (dark-navy
cortex on white page) with concept-selective voxels in red and white ROI outlines
(V1/V2/V3/hV4, FFA/OFA, PPA/OPA/RSC, EBA, VWFA), beside a montage of the concept's
top example stimuli.

Selectivity(voxel) = mean synthesized response over the concept's images
                     minus the mean response over all test images (baseline).

Pipeline:
  * group the 1000 test images by concept via COCO caption keywords
    (shared with plot_concept_selectivity.py)
  * average synthesized fMRI within each group, subtract the global baseline
  * sample the volume onto each hemi's vertices (white->pial, multi-depth NN)
  * render with pycortex make_flatmap_image (data layer) over a navy brain mask,
    then overlay ROI boundaries + labels computed from NSD freesurfer label files.

Requires the pycortex subject built by src/pycortex_import_nsd.py (default
cx name 'nsd_subj01'). sub1 only (only subject with a pycortex flat surface).

Usage:
  .venv/bin/python src/utils/plot_concept_flatmap.py --sub sub1 \
      --concepts food person animal vehicle
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import matplotlib.patheffects as pe
from matplotlib import gridspec
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from PIL import Image

from plot_concept_selectivity import (
    CONCEPT_KEYWORDS, DEFAULT_CONCEPTS, group_by_caption, maybe_unsort,
)

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "results" / "rfr_eval"
NSD = ROOT / "NSD" / "data" / "nsd"
ROIDIR = ROOT / "NSD" / "data" / "nsddata" / "ppdata" / "{subjxx}" / "func1pt8mm" / "roi"
FSDIR = ROOT / "NSD" / "data" / "nsddata" / "freesurfer"
SUBJ = {"sub1": "subj01", "sub2": "subj02", "sub5": "subj05", "sub7": "subj07"}
CX = {"sub1": "nsd_subj01", "sub2": "nsd_subj02", "sub5": "nsd_subj05", "sub7": "nsd_subj07"}

# ROI groups to outline: name -> (label-file stem, {merged sub-region ids})
ROI_GROUPS = {
    "prf-visualrois": {"V1": {1, 2}, "V2": {3, 4}, "V3": {5, 6}, "hV4": {7}},
    "floc-faces":     {"OFA": {1}, "FFA": {2, 3}},
    "floc-places":    {"OPA": {1}, "PPA": {2}, "RSC": {3}},
    "floc-bodies":    {"EBA": {1}, "FBA": {2, 3}},
    "floc-words":     {"VWFA": {2, 3}},
}
NAVY = (0.078, 0.122, 0.290)  # dark navy like the reference figure
# only these get a text label (others keep outlines but no text, to reduce clutter)
LABEL_ROIS = {"V1", "V2", "V3", "hV4", "FFA", "PPA", "OPA", "EBA"}


def roi_vol_path(sub, name):
    return Path(str(ROIDIR).format(subjxx=SUBJ[sub])) / name


def montage(stim, idx, s=130):
    ims = [np.asarray(Image.fromarray(np.asarray(stim[i])).convert("RGB").resize((s, s)))
           for i in idx[:4]]
    while len(ims) < 4:
        ims.append(np.full((s, s, 3), 255, np.uint8))
    return np.concatenate([np.concatenate(ims[:2], 1), np.concatenate(ims[2:4], 1)], 0)


def load_roi_membership(sub, merged_n, nL):
    """For each outlined ROI -> boolean over merged [lh; rh] vertices."""
    groups = {}
    for stem, subregions in ROI_GROUPS.items():
        per_hemi = {}
        for h in ("lh", "rh"):
            f = FSDIR / SUBJ[sub] / "label" / f"{h}.{stem}.mgz"
            per_hemi[h] = np.asarray(nib.load(str(f)).dataobj).squeeze().astype(int)
        for roi, ids in subregions.items():
            m = np.zeros(merged_n, bool)
            m[:nL] = np.isin(per_hemi["lh"], list(ids))
            m[nL:] = np.isin(per_hemi["rh"], list(ids))
            groups[roi] = m
    return groups


def roi_boundary_segments(pts2d, polys, member):
    edges = np.vstack([polys[:, [0, 1]], polys[:, [1, 2]], polys[:, [2, 0]]])
    on = member[edges[:, 0]] != member[edges[:, 1]]
    bnd = edges[on]
    return pts2d[bnd]                                     # (n_edge, 2, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default="sub1", choices=["sub1"],
                    help="Only sub1 has a pycortex flat surface + captions here.")
    ap.add_argument("--noise", default="0.2")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--concepts", nargs="*", default=DEFAULT_CONCEPTS,
                    help=f"Subset of {list(CONCEPT_KEYWORDS)}")
    ap.add_argument("--min-n", type=int, default=20,
                    help="Skip concepts with fewer than this many matched images.")
    ap.add_argument("--pct", type=float, default=82.0, help="threshold percentile of positive selectivity")
    ap.add_argument("--no-labels", action="store_true", help="draw ROI outlines but no text labels")
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--outdir", default=str(ROOT / "results" / "viz"))
    args = ap.parse_args()
    sub = args.sub
    cx = CX[sub]
    for c in args.concepts:
        if c not in CONCEPT_KEYWORDS:
            ap.error(f"unknown concept '{c}'; choose from {list(CONCEPT_KEYWORDS)}")

    import cortex
    from cortex.quickflat import utils as qu
    from nibabel.freesurfer import read_geometry
    from nilearn.surface import vol_to_surf

    # --- data (anatomical nsdgeneral order; guarded unsort) ----------------
    # Two columns per concept: synthesized vs ground-truth selectivity.
    ev = np.load(EVAL / f"{sub}_noise{args.noise}" / f"avg_k{args.k:02d}.npz")
    syn = maybe_unsort(ev["preds"], ev["targets"], sub)
    gt = ev["targets"]                       # already anatomical
    sources = [("synth", syn, syn.mean(0)), ("GT", gt, gt.mean(0))]
    stim = np.load(NSD / SUBJ[sub] / f"nsd_test_stim_{sub}.npy", mmap_mode="r")
    caps = np.load(NSD / SUBJ[sub] / f"nsd_test_cap_{sub}.npy", allow_pickle=True)
    mask_img = nib.load(str(roi_vol_path(sub, "nsdgeneral.nii.gz")))
    mask = np.asarray(mask_img.get_fdata()) > 0

    # --- caption-keyword grouping ------------------------------------------
    groups, kept = [], []
    for c in args.concepts:
        idx = group_by_caption(caps, CONCEPT_KEYWORDS[c])
        print(f"[group] {c:9s}: {len(idx):4d} images")
        if len(idx) >= args.min_n:
            groups.append(idx); kept.append(c)
        else:
            print(f"        -> skipped (< {args.min_n})")
    if not kept:
        raise SystemExit("No concept group large enough; lower --min-n.")

    # --- learn coordinate transform and setup projection -------------------
    fsdir = FSDIR / SUBJ[sub]
    roi_dir = Path(str(ROIDIR).format(subjxx=SUBJ[sub]))

    func_affine = mask_img.affine
    w_lh, _ = read_geometry(str(fsdir / "surf" / "lh.white"))
    p_lh, _ = read_geometry(str(fsdir / "surf" / "lh.pial"))
    mid = 0.5 * (w_lh + p_lh)
    pts_s, pts_v = [], []
    for atlas in ("floc-faces", "floc-places", "floc-bodies",
                  "prf-visualrois", "Kastner2015", "streams"):
        try:
            avol = np.asarray(nib.load(str(roi_dir / f"lh.{atlas}.nii.gz")).get_fdata())
            asrf = np.asarray(nib.load(str(fsdir / "label" / f"lh.{atlas}.mgz")
                                       ).dataobj).squeeze().astype(int)
        except FileNotFoundError:
            continue
        for lid in np.unique(asrf):
            if lid <= 0:
                continue
            sm, vm = asrf == lid, avol == lid
            if sm.sum() < 10 or vm.sum() < 10:
                continue
            pts_s.append(mid[sm].mean(0))
            vol_ijk = np.array(np.where(vm)).T.astype(float)
            pts_v.append(func_affine[:3, :3] @ vol_ijk.mean(0) + func_affine[:3, 3])

    S = np.column_stack([np.array(pts_s), np.ones(len(pts_s))])
    V = np.array(pts_v)
    xfm_coeffs, *_ = np.linalg.lstsq(S, V, rcond=None)
    tkr_to_scanner = np.eye(4)
    tkr_to_scanner[:3, :3] = xfm_coeffs[:3].T
    tkr_to_scanner[:3,  3] = xfm_coeffs[3]

    surf = {}
    for h in ["lh", "rh"]:
        cw, polys = read_geometry(str(fsdir / "surf" / f"{h}.white"))
        cp, _ = read_geometry(str(fsdir / "surf" / f"{h}.pial"))
        
        # Transform surface coordinates to scanner RAS space
        cw_scanner = (tkr_to_scanner[:3, :3] @ cw.T + tkr_to_scanner[:3, 3:4]).T
        cp_scanner = (tkr_to_scanner[:3, :3] @ cp.T + tkr_to_scanner[:3, 3:4]).T
        
        surf[h] = {
            "white_scanner": cw_scanner,
            "pial_scanner": cp_scanner,
            "polys": polys,
            "nv": len(cw)
        }

    nL = surf["lh"]["nv"]
    merged_n = nL + surf["rh"]["nv"]

    def to_surface(sel_vol, h):
        vol_img = nib.Nifti1Image(sel_vol, func_affine)
        projected = vol_to_surf(
            vol_img,
            surf_mesh=(surf[h]["pial_scanner"], surf[h]["polys"]),
            inner_mesh=(surf[h]["white_scanner"], surf[h]["polys"]),
            interpolation="linear"
        )
        return projected

    # --- flat geometry (shared) -------------------------------------------
    pts, polys = cortex.db.get_surf(cx, "flat", merge=True, nudge=True)[:2]
    pts2d = pts[:, :2]
    brainmask, extents = qu.get_flatmask(cx, height=args.height)
    brainmask = brainmask.T                                # (H, W) to match data image
    roi_members = load_roi_membership(sub, merged_n, nL)

    # --- per-concept selectivity -> vertex vectors (synth + GT) ------------
    # vverts[ci][si] = vertex selectivity for concept ci under source si.
    # Each concept is scaled to its OWN positive distribution (thr_c/vmax_c) so
    # weak concepts aren't washed out by a global scale set by food/person;
    # synth & GT share a concept's scale, keeping them directly comparable.
    vverts, thr_c, vmax_c = [], [], []
    for idx in groups:
        per_src, pos = [], []
        for _, data, base in sources:
            sel = data[idx].mean(0) - base
            vol = np.zeros(mask.shape, np.float32)
            vol[mask] = sel
            vv = np.concatenate([to_surface(vol, "lh"), to_surface(vol, "rh")])
            per_src.append(vv)
            pos.append(vv[np.isfinite(vv) & (vv > 0)])
        vverts.append(per_src)
        pos = np.concatenate(pos)
        thr_c.append(float(np.percentile(pos, args.pct)))
        vmax_c.append(float(np.percentile(pos, 99.5)))
    for c, t, vm in zip(kept, thr_c, vmax_c):
        print(f"[scale] {c:9s} thr={t:.3f} vmax={vm:.3f}")

    # --- render (2x2 grid of concepts; synth | GT flatmap per concept) -----
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad(alpha=0.0)
    halo = [pe.withStroke(linewidth=1.4, foreground=NAVY)]
    navy = np.zeros((*brainmask.shape, 4), np.float32)
    navy[brainmask] = (*NAVY, 1.0)

    def draw_flat(ax, vv, thr, vmax, header=None):
        norm = Normalize(thr, vmax)
        ax.set_facecolor("white")
        ax.imshow(navy, extent=extents, origin="lower", interpolation="nearest", zorder=1)
        v = cortex.Vertex(vv, cx, vmin=thr, vmax=vmax)
        img, _ = qu.make_flatmap_image(v, height=args.height)
        img = np.ma.masked_invalid(np.ma.masked_less(img, thr))
        # make_flatmap_image returns a display-oriented raster (row 0 = top), so
        # it must be drawn origin="upper" to align with the pts2d ROI outlines /
        # navy mask; origin="lower" vertically flips the data off the outlines.
        ax.imshow(img, extent=extents, origin="upper", cmap=cmap, norm=norm,
                  interpolation="nearest", zorder=2)
        for roi, member in roi_members.items():
            segs = roi_boundary_segments(pts2d, polys, member)
            if len(segs):
                ax.add_collection(LineCollection(segs, colors="white", linewidths=0.6,
                                                 alpha=0.9, zorder=3))
            if args.no_labels or roi not in LABEL_ROIS:
                continue
            for lo, hi in [(0, nL), (nL, merged_n)]:
                ridx = np.where(member[lo:hi])[0] + lo
                ridx = ridx[(pts2d[ridx] != 0).any(1)]   # drop cleaned (0,0) verts
                if len(ridx) > 15:
                    cx0, cy0 = np.median(pts2d[ridx], axis=0)
                    ax.text(cx0, cy0, roi, color="white", fontsize=6.5, ha="center",
                            va="center", zorder=4, fontweight="bold", path_effects=halo)
        ax.set_xlim(extents[0], extents[1])
        ax.set_ylim(extents[2], extents[3])
        ax.set_aspect("equal"); ax.axis("off")
        if header:
            ax.set_title(header, fontsize=11, color="0.25")

    n = len(kept)
    ncol = 1 if n == 1 else 2                  # concepts per row (2x2 like the reference)
    nrow = int(np.ceil(n / ncol))
    cols = 3                                   # montage | synth | GT, per concept
    fig = plt.figure(figsize=(11.5 * ncol, 3.7 * nrow), facecolor="white")
    gs = gridspec.GridSpec(nrow, cols * ncol,
                           width_ratios=[0.34, 1.0, 1.0] * ncol, wspace=0.02, hspace=0.14)

    for ci, concept in enumerate(kept):
        r, c = divmod(ci, ncol)
        base_col = cols * c
        # montage
        axm = fig.add_subplot(gs[r, base_col])
        axm.imshow(montage(stim, groups[ci]))
        axm.set_xticks([]); axm.set_yticks([])
        axm.set_title(f"Concept: {concept}", fontsize=12, loc="left", fontweight="bold")
        # synth + GT flatmaps (header only on first row to avoid clutter)
        for si, (name, _, _) in enumerate(sources):
            ax = fig.add_subplot(gs[r, base_col + 1 + si])
            draw_flat(ax, vverts[ci][si], thr_c[ci], vmax_c[ci],
                      header=(name if r == 0 else None))

    # colour scale is per-concept (each row normalised to its own range), so the
    # legend shows relative low->high rather than absolute values.
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(0, 1))
    cax = fig.add_axes([0.93, 0.30, 0.011, 0.40])
    cb = fig.colorbar(sm, cax=cax, ticks=[0, 1],
                      label="concept selectivity (per-concept: low $\\to$ high)")
    cb.ax.set_yticklabels(["low", "high"])
    out = Path(args.outdir) / f"14_concept_flatmap_{sub}_syn_vs_gt.png"
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
