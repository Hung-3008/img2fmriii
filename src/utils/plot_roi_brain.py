#!/usr/bin/env python3
"""ROI-level fidelity visualizations for StimFlow (cortical surface + bar charts).

Modes
-----
streams  : NSD functional streams painted on the surface + per-stream accuracy bars
           (the categorical figure).
heatmap  : fine-grained accuracy *heatmap* -- the cortical surface colored continuously
           by per-region voxel Pearson r, using the HCP-MMP1 atlas (full visual-cortex
           coverage). Shows the spatial gradient of synthesis accuracy.  [viz 1]
fine     : finer functional breakdown -- retinotopic areas (V1-hV4 from prf-visualrois)
           and category-selective regions (FFA/OFA/EBA/OPA/PPA/VWFA from floc) painted
           on the surface + a color-matched accuracy bar chart.                [viz 2]

Per-ROI accuracy = per-voxel Pearson r between synthesized and GT responses across the
1000 test images, grouped by the func-space volume atlas (same C-order as the eval
tensors) and rendered with the matching native surface labels. Error bars = std across
the 5 stochastic synthesis passes.

Usage:
    .venv/bin/python src/plot_roi_brain.py --mode heatmap --sub sub1
    .venv/bin/python src/plot_roi_brain.py --mode fine    --sub sub1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from matplotlib import gridspec
from matplotlib.colors import ListedColormap
from nilearn import plotting

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "results" / "rfr_eval"
PP = ROOT / "NSD" / "data" / "nsddata" / "ppdata" / "{subjxx}" / "func1pt8mm" / "roi"
FS = ROOT / "NSD" / "data" / "nsddata" / "freesurfer" / "{subjxx}"
SUBJ = {"sub1": "subj01", "sub2": "subj02", "sub5": "subj05", "sub7": "subj07"}

STREAM_COLORS = {1: "#E15759", 2: "#F28E2B", 3: "#59A14F", 4: "#B07AA1",
                 5: "#4E79A7", 6: "#9C755F", 7: "#EDC948"}

# fine visual atlas (viz 2): name, atlas file, source labels, color. Listed early -> high;
# later (category-selective) entries override earlier (retinotopic) on overlap.
FINE = [
    ("V1",  "prf-visualrois", (1, 2), "#08519c"),
    ("V2",  "prf-visualrois", (3, 4), "#3182bd"),
    ("V3",  "prf-visualrois", (5, 6), "#6baed6"),
    ("hV4", "prf-visualrois", (7,),   "#9ecae1"),
    ("OFA", "floc-faces",     (1,),   "#fdae6b"),
    ("FFA", "floc-faces",     (2, 3), "#e6550d"),
    ("EBA", "floc-bodies",    (1,),   "#31a354"),
    ("OPA", "floc-places",    (1,),   "#9e9ac8"),
    ("PPA", "floc-places",    (2,),   "#756bb1"),
    ("VWFA", "floc-words",    (2, 3), "#d62728"),
]


def fs_path(sub, *parts):
    return Path(str(FS).format(subjxx=SUBJ[sub])).joinpath(*parts)


def roi_path(sub, name):
    return Path(str(PP).format(subjxx=SUBJ[sub])) / name


def load_mask(sub):
    return np.asarray(nib.load(str(roi_path(sub, "nsdgeneral.nii.gz"))).get_fdata()) > 0


def stream_names(sub):
    f = fs_path(sub, "label", "streams.mgz.ctab")
    return {int(l.split()[0]): l.split()[1] for l in open(f) if l.split()}


def per_pass_voxel_r(sub, noise, k):
    """voxel_r per stochastic pass (n_pass, n_vox); falls back to avg if no passes."""
    ev = np.load(EVAL / f"{sub}_noise{noise}" / f"avg_k{k:02d}.npz")
    t = ev["targets"]; tz = (t - t.mean(0)) / (t.std(0) + 1e-8)
    pdir = EVAL / f"{sub}_noise{noise}" / "passes"
    passes = sorted(pdir.glob("pass_*.npy")) if pdir.exists() else []
    if not passes:
        return ev["voxel_r"][None]
    out = []
    for f in passes:
        p = np.load(f); pz = (p - p.mean(0)) / (p.std(0) + 1e-8)
        out.append((pz * tz).mean(0))
    return np.stack(out)


def load_surf(sub, h):
    mesh = str(fs_path(sub, "surf", f"{h}.inflated"))
    curv = nib.freesurfer.read_morph_data(str(fs_path(sub, "surf", f"{h}.curv")))
    return mesh, curv


def surf_label(sub, h, atlas):
    return np.asarray(nib.load(str(fs_path(sub, "label", f"{h}.{atlas}.mgz")
                                       )).get_fdata()).squeeze().astype(int)


# --------------------------------------------------------------------------- #
def render_streams(sub, noise, k, outdir):
    mask = load_mask(sub)
    svol = np.asarray(nib.load(str(roi_path(sub, "streams.nii.gz"))).get_fdata())[mask].astype(int)
    names = stream_names(sub)
    vr = per_pass_voxel_r(sub, noise, k)
    streams = [v for v in sorted(STREAM_COLORS) if (svol == v).sum() > 0]
    mean_r = np.array([vr[:, svol == v].mean() for v in streams])
    err = np.array([vr[:, svol == v].mean(1).std() for v in streams])
    order = np.argsort(mean_r)

    h = "lh"
    mesh, curv = load_surf(sub, h)
    roi = surf_label(sub, h, "streams").astype(float); roi[roi == 0] = np.nan
    cmap = ListedColormap([STREAM_COLORS[v] for v in range(1, 8)])

    fig = plt.figure(figsize=(13, 5))
    gs = gridspec.GridSpec(2, 2, width_ratios=[0.85, 1.4], wspace=0.06, hspace=0.02)
    for i, view in enumerate(["lateral", "medial"]):
        ax = fig.add_subplot(gs[i, 0], projection="3d")
        plotting.plot_surf_roi(mesh, roi, hemi="left", view=view, bg_map=curv,
                               cmap=cmap, vmin=1, vmax=7, bg_on_data=True,
                               colorbar=False, avg_method="median", axes=ax, figure=fig)
        ax.set_title(view, fontsize=10, y=0.97)
    axb = fig.add_subplot(gs[:, 1])
    y = np.arange(len(streams))
    axb.barh(y, mean_r[order], xerr=err[order],
             color=[STREAM_COLORS[streams[i]] for i in order], edgecolor="white",
             error_kw=dict(ecolor="#444", lw=1.1, capsize=3))
    axb.set_yticks(y); axb.set_yticklabels([names[streams[i]].capitalize() for i in order])
    for tk, i in zip(axb.get_yticklabels(), order):
        tk.set_color(STREAM_COLORS[streams[i]]); tk.set_fontweight("bold")
    axb.set_xlabel("Pearson score")
    for s in ("top", "right"):
        axb.spines[s].set_visible(False)
    axb.set_xlim(0, max(mean_r + err) * 1.12)
    out = outdir / f"10_roi_brain_{sub}.png"
    fig.savefig(out, dpi=170, bbox_inches="tight"); plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
def render_heatmap(sub, noise, k, outdir):
    """viz 1: surface colored continuously by per-HCP-region voxel_r."""
    mask = load_mask(sub)
    vr = per_pass_voxel_r(sub, noise, k).mean(0)            # mean over passes
    views = ["lateral", "ventral", "medial"]

    # build per-vertex r for the left hemisphere via HCP regions
    h = "lh"
    volm = np.asarray(nib.load(str(roi_path(sub, f"{h}.HCP_MMP1.nii.gz"))).get_fdata())[mask].astype(int)
    reg_mean = {lab: vr[volm == lab].mean() for lab in np.unique(volm) if lab > 0}
    surf = surf_label(sub, h, "HCP_MMP1")
    vtx = np.full(surf.shape, np.nan)
    for lab, m in reg_mean.items():
        vtx[surf == lab] = m
    mesh, curv = load_surf(sub, h)
    vmax = float(np.nanpercentile(list(reg_mean.values()), 97))

    fig = plt.figure(figsize=(4.2 * len(views), 4.4))
    gs = gridspec.GridSpec(1, len(views) + 1, width_ratios=[1] * len(views) + [0.06],
                           wspace=0.02)
    for i, view in enumerate(views):
        ax = fig.add_subplot(gs[0, i], projection="3d")
        plotting.plot_surf_stat_map(mesh, vtx, hemi="left", view=view, bg_map=curv,
                                    cmap="inferno", vmin=0, vmax=vmax, threshold=1e-6,
                                    bg_on_data=True, colorbar=False, axes=ax, figure=fig)
        ax.set_title(view, fontsize=11, y=0.97)
    cax = fig.add_subplot(gs[0, -1])
    sm = plt.cm.ScalarMappable(cmap="inferno",
                               norm=plt.Normalize(vmin=0, vmax=vmax))
    fig.colorbar(sm, cax=cax, label="Pearson score")
    out = outdir / f"11_roi_heatmap_{sub}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight"); plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
def _combine(get_atlas):
    """combined fine-region id per element (1..len(FINE)); later entries override."""
    cache, out = {}, None
    for i, (_, atlas, labels, _) in enumerate(FINE):
        if atlas not in cache:
            cache[atlas] = get_atlas(atlas)
        a = cache[atlas]
        if out is None:
            out = np.zeros(a.shape, int)
        out[np.isin(a, labels)] = i + 1
    return out


def render_fine(sub, noise, k, outdir):
    """viz 2: fine retinotopic + category-selective ROIs (surface + bars)."""
    mask = load_mask(sub)
    vr = per_pass_voxel_r(sub, noise, k)
    vol_lab = _combine(lambda a: np.asarray(
        nib.load(str(roi_path(sub, f"{a}.nii.gz"))).get_fdata())[mask].astype(int))

    names = [r[0] for r in FINE]; colors = [r[3] for r in FINE]
    mean_r, err = [], []
    for i in range(1, len(FINE) + 1):
        m = vol_lab == i
        mean_r.append(vr[:, m].mean()); err.append(vr[:, m].mean(1).std())
    mean_r, err = np.array(mean_r), np.array(err)

    h = "lh"
    mesh, curv = load_surf(sub, h)
    surf_lab = _combine(lambda a: surf_label(sub, h, a)).astype(float)
    surf_lab[surf_lab == 0] = np.nan
    cmap = ListedColormap(colors)

    fig = plt.figure(figsize=(13, 5.2))
    gs = gridspec.GridSpec(2, 2, width_ratios=[0.85, 1.4], wspace=0.06, hspace=0.02)
    for i, view in enumerate(["lateral", "ventral"]):
        ax = fig.add_subplot(gs[i, 0], projection="3d")
        plotting.plot_surf_roi(mesh, surf_lab, hemi="left", view=view, bg_map=curv,
                               cmap=cmap, vmin=1, vmax=len(FINE), bg_on_data=True,
                               colorbar=False, avg_method="median", axes=ax, figure=fig)
        ax.set_title(view, fontsize=10, y=0.97)
    axb = fig.add_subplot(gs[:, 1])
    y = np.arange(len(FINE))[::-1]                          # V1 on top
    axb.barh(y, mean_r, xerr=err, color=colors, edgecolor="white",
             error_kw=dict(ecolor="#444", lw=1.0, capsize=3))
    axb.set_yticks(y); axb.set_yticklabels(names)
    for tk, c in zip(axb.get_yticklabels(), colors):
        tk.set_color(c); tk.set_fontweight("bold")
    axb.set_xlabel("Pearson score")
    for s in ("top", "right"):
        axb.spines[s].set_visible(False)
    axb.set_xlim(0, float(np.max(mean_r + err)) * 1.12)
    axb.text(0.99, 0.5, "retinotopic  $\\rightarrow$  category-selective",
             transform=axb.transAxes, rotation=90, va="center", ha="left",
             fontsize=8, color="#888")
    out = outdir / f"12_roi_fine_{sub}.png"
    fig.savefig(out, dpi=170, bbox_inches="tight"); plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="heatmap", choices=["streams", "heatmap", "fine"])
    ap.add_argument("--sub", default="sub1")
    ap.add_argument("--noise", default="0.2")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--outdir", default=str(ROOT / "results" / "viz"))
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    fn = {"streams": render_streams, "heatmap": render_heatmap, "fine": render_fine}[args.mode]
    out = fn(args.sub, args.noise, args.k, outdir)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
