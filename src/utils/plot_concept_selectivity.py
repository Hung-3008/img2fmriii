#!/usr/bin/env python3
"""Concept-selective region localization from StimFlow-synthesized fMRI (sub1).

Reproduces the MindSimulator "concept localization" experiment, but groups the
1000 shared test images by **image caption keywords** (not CLIP zero-shot): for
each concept we collect every test image whose COCO captions mention one of the
concept's keywords, average the *synthesized* fMRI within that group, subtract
the global baseline, and render which cortical regions respond.

Output (per concept, one row):
  * a 2x2 montage of example stimuli for the group,
  * the selectivity map on the inflated cortical surface (lh lateral, lh/rh
    ventral), red = concept-selective.

Why sub1 only: caption files and freesurfer surfaces are available only for
subj01 in this checkout (the 1000 test images are identical across subjects, but
sub2/5/7 lack surfaces to render on). Run ROI-level analysis for the other
subjects instead.

Voxel order: the eval tensors in results/rfr_eval/<sub>_noise<s>/avg_kNN.npz are
saved by the evaluator already in anatomical nsdgeneral order (the ROI-sort used
during training is undone via unsort_idx before export). We assert this against
roi_meta and only unsort if a stale sorted file is detected.

Selectivity(voxel) = (mean_group - mean_all) / (std_group / sqrt(n))   [one-sample t]

Usage:
    .venv/bin/python src/utils/plot_concept_selectivity.py
    .venv/bin/python src/utils/plot_concept_selectivity.py \
        --concepts food person animal vehicle indoor --source syn
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from matplotlib import gridspec
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "results" / "rfr_eval"
NSD = ROOT / "NSD" / "data" / "nsd"
ROI = ROOT / "NSD" / "data" / "nsddata" / "ppdata" / "{subjxx}" / "func1pt8mm" / "roi"
FSDIR = ROOT / "NSD" / "data" / "nsddata" / "freesurfer"
SUBJ = {"sub1": "subj01", "sub2": "subj02", "sub5": "subj05", "sub7": "subj07"}

# Concept -> caption keywords. A test image joins a concept group if ANY of its
# (up to 5) COCO captions contains one of these as a whole word. Concepts map to
# well-studied selective regions: food (ventral food patches), person
# (bodies/faces, EBA/FFA), animal (lateral/ventral), vehicle, indoor/outdoor
# scenes (places, PPA/RSC/OPA).
CONCEPT_KEYWORDS = {
    "food": ["food", "pizza", "sandwich", "fruit", "fruits", "vegetable",
             "vegetables", "meal", "plate", "cake", "donut", "donuts", "banana",
             "bananas", "bread", "dish", "dishes", "dinner", "lunch",
             "breakfast", "broccoli", "salad", "cheese", "hotdog"],
    "person": ["person", "people", "man", "men", "woman", "women", "child",
               "children", "boy", "boys", "girl", "girls", "player", "players",
               "crowd", "guy", "lady", "kid", "kids", "baby"],
    "animal": ["animal", "animals", "dog", "dogs", "cat", "cats", "bird",
               "birds", "horse", "horses", "elephant", "elephants", "bear",
               "bears", "zebra", "zebras", "giraffe", "giraffes", "cow", "cows",
               "sheep"],
    "vehicle": ["car", "cars", "truck", "trucks", "bus", "buses", "train",
                "trains", "airplane", "airplanes", "plane", "planes",
                "motorcycle", "motorcycles", "bicycle", "bicycles", "bike",
                "bikes", "boat", "boats", "vehicle"],
    "sport": ["baseball", "tennis", "soccer", "skateboard", "skateboarding",
              "surfboard", "surfing", "surfer", "skiing", "skis", "ski",
              "snowboard", "frisbee", "court", "racket", "bat", "skis"],
    "indoor": ["room", "kitchen", "bathroom", "bedroom", "indoor", "office",
               "desk", "couch", "furniture", "table", "counter", "cabinet"],
    # places/scenes (PPA/RSC) — architecture & streets, distinct from food/person
    "place": ["building", "buildings", "house", "houses", "street", "city",
              "road", "bridge", "tower", "downtown", "store", "station",
              "skyline", "church", "skyscraper"],
    # words/text (VWFA) — the remaining canonical localizer category
    "words": ["sign", "signs", "text", "letters", "clock", "numbers", "number",
              "writing", "menu", "words", "board", "license plate"],
    "outdoor": ["beach", "mountain", "mountains", "field", "sky", "ocean",
                "street", "park", "snow", "forest", "grass", "lake", "river"],
}
DEFAULT_CONCEPTS = ["food", "person", "animal", "vehicle", "indoor"]


def roi_path(sub, name):
    return Path(str(ROI).format(subjxx=SUBJ[sub])) / name


def group_by_caption(caps, keywords):
    """Indices of images whose captions mention any keyword (whole-word match)."""
    pat = re.compile(r"\b(" + "|".join(re.escape(k) for k in keywords) + r")\b")
    idx = []
    for i in range(len(caps)):
        text = " ".join(str(c) for c in caps[i]).lower()
        if pat.search(text):
            idx.append(i)
    return np.asarray(idx, dtype=int)


def montage(stim, idx, s=120):
    """2x2 collage of the first 4 example stimuli."""
    ims = [np.asarray(Image.fromarray(np.asarray(stim[i])).convert("RGB").resize((s, s)))
           for i in idx[:4]]
    while len(ims) < 4:
        ims.append(np.full((s, s, 3), 255, np.uint8))
    top = np.concatenate(ims[:2], 1); bot = np.concatenate(ims[2:4], 1)
    return np.concatenate([top, bot], 0)


def maybe_unsort(fmri, targets, sub):
    """The eval npz is already anatomical; guard against a stale ROI-sorted file."""
    meta_path = NSD / SUBJ[sub] / f"roi_meta_{sub}.npz"
    if not meta_path.exists():
        return fmri
    unsort_idx = np.load(meta_path)["unsort_idx"]
    try:
        corr_naive = np.corrcoef(fmri[0], targets[0])[0, 1]
        corr_unsorted = np.corrcoef(fmri[0, unsort_idx], targets[0])[0, 1]
        if corr_unsorted > corr_naive + 0.1:
            print("[!] eval file appears ROI-sorted; applying unsort_idx -> anatomical")
            return fmri[:, unsort_idx]
    except Exception:
        pass
    return fmri


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default="sub1", choices=["sub1"],
                    help="Only sub1 has caption + surface data in this checkout.")
    ap.add_argument("--noise", default="0.2")
    ap.add_argument("--k", type=int, default=5, help="K-trial averaged eval file (avg_kNN).")
    ap.add_argument("--concepts", nargs="*", default=DEFAULT_CONCEPTS,
                    help=f"Subset of {list(CONCEPT_KEYWORDS)}")
    ap.add_argument("--source", default="syn", choices=["syn", "gt"])
    ap.add_argument("--min-n", type=int, default=20,
                    help="Skip concepts with fewer than this many matched images.")
    ap.add_argument("--outdir", default=str(ROOT / "results" / "viz"))
    args = ap.parse_args()
    sub = args.sub

    for c in args.concepts:
        if c not in CONCEPT_KEYWORDS:
            ap.error(f"unknown concept '{c}'; choose from {list(CONCEPT_KEYWORDS)}")

    # --- synthesized / GT fMRI (anatomical nsdgeneral order) ---------------
    ev = np.load(EVAL / f"{sub}_noise{args.noise}" / f"avg_k{args.k:02d}.npz")
    fmri = ev["preds"] if args.source == "syn" else ev["targets"]      # (1000, n_vox)
    fmri = maybe_unsort(fmri, ev["targets"], sub)
    base = fmri.mean(0)                                                # global baseline
    stim = np.load(NSD / SUBJ[sub] / f"nsd_test_stim_{sub}.npy", mmap_mode="r")
    caps = np.load(NSD / SUBJ[sub] / f"nsd_test_cap_{sub}.npy", allow_pickle=True)
    mask_img = nib.load(str(roi_path(sub, "nsdgeneral.nii.gz")))
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

    # --- surface samplers (data-driven tkr-RAS -> func-voxel transform) ----
    from nilearn import plotting
    from nibabel.freesurfer import read_geometry
    from nilearn.surface import vol_to_surf
    fsdir = FSDIR / SUBJ[sub]
    roi_dir = Path(str(ROI).format(subjxx=SUBJ[sub]))

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
        cw_scanner = (tkr_to_scanner[:3, :3] @ cw.T + tkr_to_scanner[:3, 3:4]).T
        cp_scanner = (tkr_to_scanner[:3, :3] @ cp.T + tkr_to_scanner[:3, 3:4]).T
        curv = nib.freesurfer.read_morph_data(str(fsdir / "surf" / f"{h}.curv"))
        bg = 0.6 - 0.25 * np.tanh(4.0 * (curv - curv.mean()) / (curv.std() + 1e-6))
        surf[h] = {
            "infl": str(fsdir / "surf" / f"{h}.inflated"),
            "bg": bg,
            "white_scanner": cw_scanner,
            "pial_scanner": cp_scanner,
            "polys": polys,
        }

    def to_surface(sel_vol, h):
        vol_img = nib.Nifti1Image(sel_vol, func_affine)
        return vol_to_surf(
            vol_img,
            surf_mesh=(surf[h]["pial_scanner"], surf[h]["polys"]),
            inner_mesh=(surf[h]["white_scanner"], surf[h]["polys"]),
            interpolation="linear",
        )

    views = [("lh", "lateral"), ("lh", "ventral"), ("rh", "ventral")]

    # pass 1: project all concepts, collect a shared threshold / vmax
    projs, allpos = [], []
    for idx in groups:
        m, s = fmri[idx].mean(0), fmri[idx].std(0)
        sel = (m - base) / (s / np.sqrt(len(idx)) + 1e-6)              # one-sample t
        vol = np.zeros(mask.shape, np.float32); vol[mask] = sel
        p = {h: to_surface(vol, h) for h in ["lh", "rh"]}
        projs.append(p)
        allpos.append(np.concatenate([p["lh"], p["rh"]]))
    allpos = np.concatenate(allpos)
    thr = float(np.percentile(allpos[allpos > 0], 55))
    vmax = float(np.percentile(allpos, 99.0))
    print(f"[*] threshold={thr:.3f} vmax={vmax:.3f}")

    # pass 2: render
    n = len(kept)
    fig = plt.figure(figsize=(3.0 + 3.2 * len(views), 2.9 * n), facecolor="white")
    gs = gridspec.GridSpec(n, 1 + len(views), width_ratios=[0.7] + [1.0] * len(views),
                           wspace=0.0, hspace=0.1)
    for ci, concept in enumerate(kept):
        axm = fig.add_subplot(gs[ci, 0])
        axm.imshow(montage(stim, groups[ci])); axm.set_xticks([]); axm.set_yticks([])
        axm.set_title(f"{concept}  (n={len(groups[ci])})", fontsize=12,
                      loc="left", fontweight="bold")
        for vi, (h, view) in enumerate(views):
            ax = fig.add_subplot(gs[ci, vi + 1], projection="3d")
            plotting.plot_surf_stat_map(
                surf[h]["infl"], projs[ci][h], hemi=("left" if h == "lh" else "right"),
                view=view, bg_map=surf[h]["bg"], cmap="YlOrRd", threshold=thr,
                vmin=thr, vmax=vmax, bg_on_data=True, colorbar=False, axes=ax, figure=fig)
            if ci == 0:
                ax.set_title(f"{h} {view}", fontsize=10, y=0.97)
    sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=plt.Normalize(thr, vmax))
    cax = fig.add_axes([0.93, 0.3, 0.011, 0.4])
    fig.colorbar(sm, cax=cax, label="concept selectivity ($t$ vs. baseline)")
    out = Path(args.outdir) / f"13_concept_selectivity_{sub}_{args.source}.png"
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
