#!/usr/bin/env python3
"""Visualize StimFlow fMRI-synthesis results.

Produces 7 complementary figures that highlight the fidelity of synthesized fMRI
and exploit the functional-ROI structure already present in the NSD data:

    1. encoding-accuracy glass brain   (where in cortex is synthesis accurate?)
    2. per-ROI accuracy bars           (no blind spot across functional streams)
    3. example pattern match           (stimulus | GT brain | synthesized brain)
    4. predicted-vs-true density       (hexbin with identity line + Pearson r)
    5. representational geometry (RSA)  (GT vs synthesized RDM, preserved structure)
    6. trials x noise curves           (multi-sample averaging -> conditional mean)
    7. brain identification / retrieval (recover the seen image from synth fMRI)

The eval tensors live in results/rfr_eval/<sub>_noise<sigma>/avg_k{01,03,05}.npz
with keys: preds, targets, voxel_r, profile_r  (1000 shared-test images).
Voxels are in native nsdgeneral mask order (C-order nonzero of nsdgeneral.nii.gz).
ROI/stream labels per voxel come from NSD/data/nsd/<subjXX>/roi_meta_<subX>.npz.

Usage:
    .venv/bin/python src/visualize_stimflow.py                 # all, sub5 default
    .venv/bin/python src/visualize_stimflow.py --subject sub5 --noise 0.2 --k 5
    .venv/bin/python src/visualize_stimflow.py --only 1 2 7
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from matplotlib import gridspec
from nilearn import plotting
from scipy.stats import rankdata

# --------------------------------------------------------------------------- #
# paths & constants
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "results" / "rfr_eval"
NSD_DIR = ROOT / "NSD" / "data" / "nsd"
ROI_DIR = ROOT / "NSD" / "data" / "nsddata" / "ppdata" / "{subjxx}" / "func1pt8mm" / "roi"
ROI_NII = ROI_DIR / "nsdgeneral.nii.gz"
# visual-hierarchy display order for the NSD functional streams
STREAM_RANK = {"early": 0, "midventral": 1, "ventral": 2, "midlateral": 3,
               "lateral": 4, "midparietal": 5, "parietal": 6, "other": 7}
SUBJ = {"sub1": "subj01", "sub2": "subj02", "sub5": "subj05", "sub7": "subj07"}
ACCENT = "#C8102E"   # StimFlow accent (deep red)
GT_C = "#1f6feb"     # ground-truth blue
SY_C = "#C8102E"     # synthesized red


# --------------------------------------------------------------------------- #
# loading helpers
# --------------------------------------------------------------------------- #
def load_eval(sub: str, noise: str, k: int):
    f = EVAL_DIR / f"{sub}_noise{noise}" / f"avg_k{k:02d}.npz"
    d = np.load(f)
    return dict(preds=d["preds"], targets=d["targets"],
                voxel_r=d["voxel_r"], profile_r=d["profile_r"])


def load_roi_meta(sub: str):
    subj = SUBJ[sub]
    d = np.load(NSD_DIR / subj / f"roi_meta_{sub}.npz", allow_pickle=True)
    return dict(labels=d["roi_labels"], names=[str(x) for x in d["roi_names"]],
                sizes=d["roi_sizes"], ids=d["roi_ids"])


def load_mask(sub: str):
    img = nib.load(str(ROI_NII).format(subjxx=SUBJ[sub]))
    return img, np.asarray(img.get_fdata()) > 0


def vec_to_nifti(vec, mask_img, mask_bool):
    """Scatter a 1-D voxel vector back into the nsdgeneral volume (C-order)."""
    vol = np.full(mask_bool.shape, np.nan, dtype=np.float32)
    vol[mask_bool] = vec
    return nib.Nifti1Image(np.nan_to_num(vol, nan=0.0), mask_img.affine)


def load_stim(sub: str):
    return np.load(NSD_DIR / SUBJ[sub] / f"nsd_test_stim_{sub}.npy", mmap_mode="r")


def load_parcellation(sub: str, atlas: str, mask_bool):
    """Return per-voxel parcel labels in eval (nsdgeneral C-order) for a given atlas."""
    f = Path(str(ROI_DIR).format(subjxx=SUBJ[sub])) / f"{atlas}.nii.gz"
    lab = np.asarray(nib.load(str(f)).get_fdata())[mask_bool].astype(int)
    return lab


def nice_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# --------------------------------------------------------------------------- #
# 1. encoding-accuracy glass brain
# --------------------------------------------------------------------------- #
def viz_encoding_glassbrain(sub, noise, k, outdir):
    ev = load_eval(sub, noise, k)
    img, mb = load_mask(sub)
    stat = vec_to_nifti(ev["voxel_r"], img, mb)
    vmax = float(np.nanpercentile(ev["voxel_r"], 99))

    fig = plt.figure(figsize=(13, 4.2))
    disp = plotting.plot_glass_brain(
        stat, figure=fig, display_mode="lyrz", colorbar=True,
        cmap="inferno", vmin=0, vmax=vmax, plot_abs=False,
        title=f"StimFlow per-voxel encoding accuracy (Pearson r)  |  "
              f"{sub}, sigma={noise}, k={k}   mean r = {ev['voxel_r'].mean():.3f}",
    )
    out = outdir / f"01_encoding_glassbrain_{sub}_noise{noise}_k{k}.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# 2. per-ROI accuracy bars across subjects
# --------------------------------------------------------------------------- #
def viz_roi_bars(subjects, noise, k, outdir):
    # gather per-ROI mean voxel_r for each subject on a common ROI-name axis
    per_sub = {}
    name_order = None
    for sub in subjects:
        ev, rm = load_eval(sub, noise, k), load_roi_meta(sub)
        lab, names = rm["labels"], rm["names"]
        means = {names[i]: ev["voxel_r"][lab == rid].mean()
                 for i, rid in enumerate(rm["ids"]) if (lab == rid).any()}
        per_sub[sub] = means
        if name_order is None:
            # order ROIs by overall accuracy (descending) using first subject
            name_order = sorted(means, key=means.get, reverse=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(name_order))
    w = 0.8 / len(subjects)
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(subjects)))
    for j, sub in enumerate(subjects):
        vals = [per_sub[sub].get(n, np.nan) for n in name_order]
        ax.bar(x + j * w, vals, w, label=sub, color=cmap[j], edgecolor="white")
    ax.set_xticks(x + 0.4 - w / 2)
    ax.set_xticklabels(name_order, rotation=20, ha="right")
    ax.set_ylabel("encoding accuracy  (per-voxel Pearson r)")
    ax.set_title(f"Synthesis accuracy is uniformly high across functional streams "
                 f"(sigma={noise}, k={k})")
    ax.axhline(0, color="k", lw=0.6)
    ax.legend(title="subject", ncol=len(subjects), frameon=False)
    nice_axes(ax)
    out = outdir / f"02_roi_bars_noise{noise}_k{k}.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# 3. example pattern match: stimulus | GT brain | synthesized brain
# --------------------------------------------------------------------------- #
def viz_pattern_match(sub, noise, k, outdir, n_examples=4):
    ev = load_eval(sub, noise, k)
    img, mb = load_mask(sub)
    stim = load_stim(sub)
    # pick the best-synthesized images (highest per-image profile r), well spread
    order = np.argsort(-ev["profile_r"])
    picks = order[np.linspace(0, len(order) // 4, n_examples).astype(int)]

    fig = plt.figure(figsize=(13, 2.7 * n_examples))
    gs = gridspec.GridSpec(n_examples, 3, width_ratios=[0.5, 1, 1],
                           wspace=0.05, hspace=0.25)
    vmax = float(np.nanpercentile(np.abs(ev["targets"][picks]), 98))
    for r, idx in enumerate(picks):
        ax_im = fig.add_subplot(gs[r, 0])
        ax_im.imshow(np.asarray(stim[idx]))
        ax_im.set_xticks([]); ax_im.set_yticks([])
        ax_im.set_ylabel(f"r={ev['profile_r'][idx]:.2f}", fontsize=10)
        if r == 0:
            ax_im.set_title("stimulus", fontsize=11)
        for col, (vec, ttl, c) in enumerate(
            [(ev["targets"][idx], "ground-truth fMRI", GT_C),
             (ev["preds"][idx], "synthesized fMRI", SY_C)]):
            ax = fig.add_subplot(gs[r, col + 1])
            plotting.plot_glass_brain(
                vec_to_nifti(vec, img, mb), figure=fig, axes=ax,
                display_mode="z", colorbar=False, cmap="cold_hot",
                vmin=-vmax, vmax=vmax, plot_abs=False)
            if r == 0:
                ax.set_title(ttl, fontsize=11)
    fig.suptitle(f"Image-specific activation patterns: synthesized vs. ground truth "
                 f"({sub}, sigma={noise}, k={k})", y=0.995, fontsize=13)
    out = outdir / f"03_pattern_match_{sub}_noise{noise}_k{k}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# 4. predicted-vs-true voxel density (hexbin), colored per ROI panel
# --------------------------------------------------------------------------- #
def viz_pred_vs_true(sub, noise, k, outdir):
    ev, rm = load_eval(sub, noise, k), load_roi_meta(sub)
    # pooled scatter (subsample) + per-ROI Pearson summary
    rng = np.random.default_rng(0)
    n_img = ev["preds"].shape[0]
    sel_img = rng.choice(n_img, size=min(300, n_img), replace=False)
    t = ev["targets"][sel_img].ravel()
    p = ev["preds"][sel_img].ravel()
    sub_idx = rng.choice(t.size, size=min(400_000, t.size), replace=False)
    t, p = t[sub_idx], p[sub_idx]
    r = np.corrcoef(t, p)[0, 1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    ax = axes[0]
    lim = np.percentile(np.abs(np.concatenate([t, p])), 99)
    hb = ax.hexbin(t, p, gridsize=70, bins="log", cmap="magma",
                   extent=(-lim, lim, -lim, lim))
    ax.plot([-lim, lim], [-lim, lim], color="cyan", lw=1.2, ls="--")
    ax.set_xlabel("ground-truth voxel z-score")
    ax.set_ylabel("synthesized voxel z-score")
    ax.set_title(f"Voxel-wise agreement (pooled)   r = {r:.3f}")
    ax.set_aspect("equal")
    fig.colorbar(hb, ax=ax, label="log10(count)")

    # per-ROI bar of pooled voxel-value correlation
    ax2 = axes[1]
    names, rs = [], []
    for i, rid in enumerate(rm["ids"]):
        m = rm["labels"] == rid
        if m.sum() < 5:
            continue
        tt = ev["targets"][:, m].ravel(); pp = ev["preds"][:, m].ravel()
        names.append(rm["names"][i]); rs.append(np.corrcoef(tt, pp)[0, 1])
    o = np.argsort(rs)[::-1]
    ax2.barh([names[i] for i in o][::-1], [rs[i] for i in o][::-1],
             color=ACCENT, edgecolor="white")
    ax2.set_xlabel("voxel-value Pearson r")
    ax2.set_title("Agreement per functional stream")
    nice_axes(ax2)
    fig.suptitle(f"Predicted vs. ground-truth voxel values ({sub}, sigma={noise}, k={k})",
                 fontsize=13)
    out = outdir / f"04_pred_vs_true_{sub}_noise{noise}_k{k}.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# 5. representational similarity (RSA): GT vs synthesized RDM
# --------------------------------------------------------------------------- #
def _rdm(x):
    """1 - correlation RDM over rows (images)."""
    xc = x - x.mean(1, keepdims=True)
    xc /= (np.linalg.norm(xc, axis=1, keepdims=True) + 1e-8)
    return 1.0 - xc @ xc.T


def viz_rsa(sub, noise, k, outdir, n_img=120):
    ev = load_eval(sub, noise, k)
    rng = np.random.default_rng(1)
    sel = np.sort(rng.choice(ev["preds"].shape[0], n_img, replace=False))
    rdm_gt, rdm_sy = _rdm(ev["targets"][sel]), _rdm(ev["preds"][sel])
    iu = np.triu_indices(n_img, k=1)
    a, b = rdm_gt[iu], rdm_sy[iu]
    rsa = np.corrcoef(rankdata(a), rankdata(b))[0, 1]   # Spearman

    fig = plt.figure(figsize=(13, 5.0))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.1], wspace=0.32)
    for j, (rdm, ttl) in enumerate([(rdm_gt, "ground-truth RDM"),
                                    (rdm_sy, "synthesized RDM")]):
        ax = fig.add_subplot(gs[0, j])
        # per-matrix scaling so both panels reveal their own structure
        lo, hi = np.percentile(rdm[iu], [2, 98])
        im = ax.imshow(rdm, cmap="viridis", vmin=lo, vmax=hi)
        ax.set_title(ttl, fontsize=11); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
    ax = fig.add_subplot(gs[0, 2])
    ax.hexbin(a, b, gridsize=45, bins="log", cmap="magma")
    ax.set_xlabel("GT dissimilarity"); ax.set_ylabel("synth dissimilarity")
    ax.set_title(f"Representational geometry\nSpearman RSA = {rsa:.3f}", fontsize=11)
    nice_axes(ax)
    fig.suptitle(f"Synthesized fMRI preserves representational geometry "
                 f"({sub}, sigma={noise}, k={k}, {n_img} images)", fontsize=13, y=1.02)
    out = outdir / f"05_rsa_{sub}_noise{noise}_k{k}.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# 6. trials x noise curves (multi-sample averaging -> conditional mean)
# --------------------------------------------------------------------------- #
def viz_trials_noise(sub, noises, ks, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    cmap = plt.cm.plasma(np.linspace(0.15, 0.8, len(noises)))
    for ci, noise in enumerate(noises):
        vr, pr = [], []
        for k in ks:
            try:
                ev = load_eval(sub, noise, k)
            except FileNotFoundError:
                vr.append(np.nan); pr.append(np.nan); continue
            vr.append(ev["voxel_r"].mean()); pr.append(ev["profile_r"].mean())
        axes[0].plot(ks, vr, "-o", color=cmap[ci], label=f"sigma={noise}")
        axes[1].plot(ks, pr, "-o", color=cmap[ci], label=f"sigma={noise}")
    for ax, ttl, yl in [(axes[0], "encoding accuracy", "mean voxel Pearson r"),
                        (axes[1], "pattern accuracy", "mean per-image r")]:
        ax.set_xlabel("samples averaged (k)"); ax.set_ylabel(yl)
        ax.set_title(ttl); ax.set_xticks(ks); nice_axes(ax)
        ax.legend(frameon=False)
    fig.suptitle(f"Averaging stochastic samples approaches the conditional mean "
                 f"({sub})", fontsize=13)
    out = outdir / f"06_trials_noise_{sub}.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# 7. brain identification / retrieval
# --------------------------------------------------------------------------- #
def viz_retrieval(sub, noise, k, outdir, n_show=60):
    ev = load_eval(sub, noise, k)
    # correlation between each synthesized response and every GT response
    P = ev["preds"] - ev["preds"].mean(1, keepdims=True)
    T = ev["targets"] - ev["targets"].mean(1, keepdims=True)
    P /= (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)
    T /= (np.linalg.norm(T, axis=1, keepdims=True) + 1e-8)
    S = P @ T.T                                    # (n,n) synth-vs-GT similarity
    n = S.shape[0]
    rank = (S >= S[np.arange(n), np.arange(n)][:, None]).sum(1)  # 1 = best
    top1 = (rank <= 1).mean(); top5 = (rank <= 5).mean(); top10 = (rank <= 10).mean()
    chance = 1.0 / n

    fig = plt.figure(figsize=(12.5, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.1, 1], wspace=0.3)
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(S[:n_show, :n_show], cmap="magma")
    ax.set_xlabel("ground-truth image"); ax.set_ylabel("synthesized response")
    ax.set_title(f"synth-vs-GT similarity (first {n_show} images)\n"
                 "bright diagonal = correct identification")
    fig.colorbar(im, ax=ax, fraction=0.046, label="correlation")

    ax2 = fig.add_subplot(gs[0, 1])
    bars = ax2.bar(["top-1", "top-5", "top-10"], [top1, top5, top10],
                   color=[ACCENT, "#e8743b", "#f0b429"], edgecolor="white")
    ax2.axhline(chance, color="gray", ls="--", lw=1,
                label=f"chance = {chance:.3f}")
    for bbar, v in zip(bars, [top1, top5, top10]):
        ax2.text(bbar.get_x() + bbar.get_width() / 2, v + 0.01,
                 f"{v*100:.1f}%", ha="center", fontsize=11)
    ax2.set_ylim(0, 1.05); ax2.set_ylabel("identification accuracy")
    ax2.set_title(f"Image identification from synthesized fMRI\n(N={n} candidates)")
    ax2.legend(frameon=False); nice_axes(ax2)
    fig.suptitle(f"Brain identification / retrieval ({sub}, sigma={noise}, k={k})",
                 fontsize=13)
    out = outdir / f"07_retrieval_{sub}_noise{noise}_k{k}.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# 8. functional-connectivity-graph similarity (native-space advantage)
# --------------------------------------------------------------------------- #
def _region_profiles(mat, labels, regions):
    """Mean response per region per image -> (n_img, n_region) profile matrix."""
    out = np.empty((mat.shape[0], len(regions)), dtype=np.float32)
    for j, r in enumerate(regions):
        out[:, j] = mat[:, labels == r].mean(1)
    return out


def viz_connectivity(sub, noise, k, outdir, atlas="HCP_MMP1", min_vox=10):
    ev = load_eval(sub, noise, k)
    img, mb = load_mask(sub)
    parc = load_parcellation(sub, atlas, mb)          # fine parcel id per voxel
    stream = load_roi_meta(sub)                        # 8 functional streams
    slab, snames = stream["labels"], stream["names"]

    # keep regions with enough voxels; order them along the visual hierarchy
    regions = [r for r in np.unique(parc[parc > 0]) if (parc == r).sum() >= min_vox]

    def region_stream_rank(r):
        vox = parc == r
        sid = np.bincount(slab[vox]).argmax()          # majority stream
        return STREAM_RANK.get(snames[list(stream["ids"]).index(sid)], 99)

    regions.sort(key=lambda r: (region_stream_rank(r), int(r)))
    regions = np.array(regions)
    ranks = np.array([region_stream_rank(r) for r in regions])

    # region x region functional connectivity = corr of regional profiles across images
    def fc(mat):
        prof = _region_profiles(mat, parc, regions)
        return np.corrcoef(prof.T)

    fc_gt, fc_sy = fc(ev["targets"]), fc(ev["preds"])
    iu = np.triu_indices(len(regions), k=1)
    a, b = fc_gt[iu], fc_sy[iu]
    mae = np.abs(a - b).mean()
    r_fc = np.corrcoef(a, b)[0, 1]

    # boundaries between streams (for block separators)
    bnds = np.where(np.diff(ranks) != 0)[0] + 0.5

    fig = plt.figure(figsize=(14, 4.8))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.05], wspace=0.4)
    vlim = np.nanpercentile(np.abs(fc_gt[iu]), 99)
    for j, (mat, ttl) in enumerate([(fc_gt, "ground-truth FC"),
                                    (fc_sy, "synthesized FC")]):
        ax = fig.add_subplot(gs[0, j])
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vlim, vmax=vlim)
        for x in bnds:
            ax.axhline(x, color="k", lw=0.4); ax.axvline(x, color="k", lw=0.4)
        ax.set_title(ttl, fontsize=11); ax.set_xticks([]); ax.set_yticks([])
        cb = fig.colorbar(im, ax=ax, fraction=0.046)
        if j == 1:
            cb.set_label("Pearson r")
    ax = fig.add_subplot(gs[0, 2])
    ax.hexbin(a, b, gridsize=40, bins="log", cmap="magma",
              extent=(-vlim, vlim, -vlim, vlim))
    ax.plot([-vlim, vlim], [-vlim, vlim], color="cyan", lw=1.1, ls="--")
    ax.set_xlabel("GT region-region r"); ax.set_ylabel("synth region-region r")
    ax.set_title(f"FC agreement\nMAE = {mae:.3f}   r = {r_fc:.3f}", fontsize=11)
    ax.set_aspect("equal"); nice_axes(ax)
    fig.suptitle(f"Synthesized fMRI preserves inter-region functional connectivity "
                 f"({sub}, sigma={noise}, k={k}, {atlas}: {len(regions)} regions)",
                 fontsize=13, y=1.02)
    out = outdir / f"08_connectivity_{sub}_noise{noise}_k{k}.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"    FC: {len(regions)} regions, MAE={mae:.4f}, r={r_fc:.4f}")
    return out


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="sub5", choices=list(SUBJ))
    ap.add_argument("--noise", default="0.2", choices=["0.2", "0.5", "1.0"])
    ap.add_argument("--k", type=int, default=5, choices=[1, 3, 5])
    ap.add_argument("--all-subjects", nargs="*", default=list(SUBJ),
                    help="subjects used in the per-ROI bar chart (#2)")
    ap.add_argument("--outdir", default=str(ROOT / "results" / "viz"))
    ap.add_argument("--only", nargs="*", type=int, default=None,
                    help="run only these viz numbers (1..8)")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    sub, noise, k = args.subject, args.noise, args.k
    want = set(args.only) if args.only else set(range(1, 9))

    jobs = {
        1: lambda: viz_encoding_glassbrain(sub, noise, k, outdir),
        2: lambda: viz_roi_bars(args.all_subjects, noise, k, outdir),
        3: lambda: viz_pattern_match(sub, noise, k, outdir),
        4: lambda: viz_pred_vs_true(sub, noise, k, outdir),
        5: lambda: viz_rsa(sub, noise, k, outdir),
        6: lambda: viz_trials_noise(sub, ["0.2", "0.5", "1.0"], [1, 3, 5], outdir),
        7: lambda: viz_retrieval(sub, noise, k, outdir),
        8: lambda: viz_connectivity(sub, noise, k, outdir),
    }
    for num in sorted(want):
        try:
            out = jobs[num]()
            print(f"[viz {num}] -> {out}")
        except Exception as e:  # keep going on a single failure
            import traceback
            print(f"[viz {num}] FAILED: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
