#!/usr/bin/env python3
"""Quantitative concept-selectivity metrics: synth vs GT selectivity maps.

For each concept and each subject, computes:
  1. Per-voxel Pearson correlation between synth and GT selectivity maps
  2. Dice overlap on the top-k% most selective voxels
  3. (Optional) Cosine similarity

Selectivity is computed identically to plot_concept_selectivity.py:
    sel(voxel) = (mean_concept - mean_all) / (std_concept / sqrt(n) + eps)

The 1000 test images are shared across subjects, so sub1's captions are used
for all subjects.

Usage:
    .venv/bin/python src/utils/compute_concept_selectivity_metrics.py
    .venv/bin/python src/utils/compute_concept_selectivity_metrics.py \
        --concepts food person animal vehicle indoor --top-k 5 10
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "results" / "rfr_eval"
NSD = ROOT / "NSD" / "data" / "nsd"
SUBJ = {"sub1": "subj01", "sub2": "subj02", "sub5": "subj05", "sub7": "subj07"}

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
    "indoor": ["room", "kitchen", "bathroom", "bedroom", "indoor", "office",
               "desk", "couch", "furniture", "table", "counter", "cabinet"],
}
DEFAULT_CONCEPTS = ["food", "person", "animal", "vehicle"]


def group_by_caption(caps, keywords):
    """Indices of images whose captions mention any keyword (whole-word match)."""
    pat = re.compile(r"\b(" + "|".join(re.escape(k) for k in keywords) + r")\b")
    idx = []
    for i in range(len(caps)):
        text = " ".join(str(c) for c in caps[i]).lower()
        if pat.search(text):
            idx.append(i)
    return np.asarray(idx, dtype=int)


def selectivity_map(fmri, idx):
    """Raw mean difference: concept mean minus global mean (no SEM normalization).
    
    This matches the figure caption: 'mean response over the concept's images
    minus the average response across all test images'.
    """
    base = fmri.mean(0)  # global baseline (all 1000 images)
    m = fmri[idx].mean(0)
    return m - base


def pearson_r(a, b):
    """Pearson correlation between two 1D arrays."""
    a = a - a.mean()
    b = b - b.mean()
    num = np.dot(a, b)
    den = np.sqrt(np.dot(a, a) * np.dot(b, b))
    return num / (den + 1e-12)


def dice_topk(a, b, k_pct):
    """Dice overlap between the top-k% voxels of two selectivity maps."""
    n = int(len(a) * k_pct / 100)
    top_a = set(np.argsort(a)[-n:])
    top_b = set(np.argsort(b)[-n:])
    intersection = len(top_a & top_b)
    return 2 * intersection / (len(top_a) + len(top_b))


def cosine_sim(a, b):
    """Cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subs", nargs="*", default=["sub1", "sub2", "sub5", "sub7"])
    ap.add_argument("--noise", default="0.2")
    ap.add_argument("--k", type=int, default=5, help="K-trial averaged eval file")
    ap.add_argument("--concepts", nargs="*", default=DEFAULT_CONCEPTS)
    ap.add_argument("--top-k", nargs="*", type=float, default=[5, 10],
                    help="Top-k%% thresholds for Dice overlap")
    ap.add_argument("--min-n", type=int, default=20)
    ap.add_argument("--outdir", default=str(ROOT / "results" / "concept_selectivity_metrics"))
    args = ap.parse_args()

    # Captions from sub1 (shared test images across subjects)
    cap_path = NSD / "subj01" / "nsd_test_cap_sub1.npy"
    if not cap_path.exists():
        raise FileNotFoundError(f"Caption file not found: {cap_path}")
    caps = np.load(cap_path, allow_pickle=True)

    # Group images by concept
    concept_groups = {}
    for c in args.concepts:
        if c not in CONCEPT_KEYWORDS:
            ap.error(f"unknown concept '{c}'")
        idx = group_by_caption(caps, CONCEPT_KEYWORDS[c])
        if len(idx) >= args.min_n:
            concept_groups[c] = idx
            print(f"[group] {c:9s}: {len(idx):4d} images")
        else:
            print(f"[group] {c:9s}: {len(idx):4d} images -> SKIPPED (< {args.min_n})")

    if not concept_groups:
        raise SystemExit("No concept group large enough")

    # Header
    dice_headers = [f"Dice@{k:.0f}%" for k in args.top_k]
    print(f"\n{'Subject':>8s} {'Concept':>8s} {'n_img':>6s} {'Pearson':>8s} "
          f"{'Cosine':>8s} " + " ".join(f"{h:>9s}" for h in dice_headers))
    print("-" * (42 + 10 * len(args.top_k)))

    all_results = []

    for sub in args.subs:
        npz_path = EVAL / f"{sub}_noise{args.noise}" / f"avg_k{args.k:02d}.npz"
        if not npz_path.exists():
            print(f"[SKIP] {npz_path} not found")
            continue

        ev = np.load(npz_path)
        preds = ev["preds"]    # (1000, n_vox)
        targets = ev["targets"]  # (1000, n_vox)

        for concept, idx in concept_groups.items():
            sel_syn = selectivity_map(preds, idx)
            sel_gt = selectivity_map(targets, idx)

            r = pearson_r(sel_syn, sel_gt)
            cos = cosine_sim(sel_syn, sel_gt)
            dices = [dice_topk(sel_syn, sel_gt, k) for k in args.top_k]

            result = {
                "subject": sub, "concept": concept, "n_img": len(idx),
                "pearson": r, "cosine": cos,
            }
            for k, d in zip(args.top_k, dices):
                result[f"dice@{k:.0f}%"] = d
            all_results.append(result)

            dice_str = " ".join(f"{d:9.4f}" for d in dices)
            print(f"{sub:>8s} {concept:>8s} {len(idx):6d} {r:8.4f} {cos:8.4f} {dice_str}")

    # Per-concept averages across subjects
    print("-" * (42 + 10 * len(args.top_k)))
    concept_avgs = {}
    for concept in concept_groups:
        rows = [r for r in all_results if r["concept"] == concept]
        if not rows:
            continue
        avg_r = np.mean([r["pearson"] for r in rows])
        avg_cos = np.mean([r["cosine"] for r in rows])
        avg_dices = [np.mean([r[f"dice@{k:.0f}%"] for r in rows]) for k in args.top_k]
        concept_avgs[concept] = {"pearson": avg_r, "cosine": avg_cos, "dices": avg_dices}
        dice_str = " ".join(f"{d:9.4f}" for d in avg_dices)
        n = concept_groups[concept].shape[0]
        print(f"{'avg':>8s} {concept:>8s} {n:6d} {avg_r:8.4f} {avg_cos:8.4f} {dice_str}")

    # Grand average
    if concept_avgs:
        grand_r = np.mean([v["pearson"] for v in concept_avgs.values()])
        grand_cos = np.mean([v["cosine"] for v in concept_avgs.values()])
        grand_dices = [np.mean([v["dices"][i] for v in concept_avgs.values()])
                       for i in range(len(args.top_k))]
        dice_str = " ".join(f"{d:9.4f}" for d in grand_dices)
        print(f"{'GRAND':>8s} {'avg':>8s} {'--':>6s} {grand_r:8.4f} {grand_cos:8.4f} {dice_str}")

    # Save results
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_csv = outdir / "concept_selectivity_metrics.csv"

    with open(out_csv, "w") as f:
        cols = ["subject", "concept", "n_img", "pearson", "cosine"] + \
               [f"dice@{k:.0f}%" for k in args.top_k]
        f.write(",".join(cols) + "\n")
        for r in all_results:
            vals = [str(r[c]) for c in cols]
            f.write(",".join(vals) + "\n")

    print(f"\n-> Saved to {out_csv}")


if __name__ == "__main__":
    main()
