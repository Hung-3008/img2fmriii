#!/usr/bin/env python3
"""Import an NSD FreeSurfer subject into the pycortex database WITHOUT FreeSurfer.

`cortex.freesurfer.import_subj` shells out to `mri_convert` / `mris_convert`, which
require a FreeSurfer install. We only ever render *per-vertex* data on the flatmap
(`cortex.Vertex`), so the absolute surface coordinate frame is irrelevant -- we can
build the store directly from the FreeSurfer geometry/morph files with nibabel and
then call the (pure-python) `cortex.freesurfer.import_flat` for the flat patch.

NSD ships the flat patches as `surf/{lh,rh}.full.flat.patch.3d`, so patch="full".

Usage:
    .venv/bin/python src/pycortex_import_nsd.py --fs_subject subj01 --cx_subject nsd_subj01
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FSDIR = ROOT / "NSD" / "data" / "nsddata" / "freesurfer"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fs_subject", default="subj01")
    ap.add_argument("--cx_subject", default=None, help="pycortex name (default nsd_<fs>)")
    ap.add_argument("--patch", default="full", help="flat patch base name -> {hemi}.<patch>.flat.patch.3d")
    args = ap.parse_args()
    cx = args.cx_subject or f"nsd_{args.fs_subject}"

    import cortex
    from cortex import database, formats
    from cortex import freesurfer as fs

    fs_surf = FSDIR / args.fs_subject / "surf"
    fs_mri = FSDIR / args.fs_subject / "mri"

    print(f"[*] make_subj {cx}")
    database.db.make_subj(cx)
    store = Path(database.default_filestore) / cx
    (store / "anatomicals").mkdir(parents=True, exist_ok=True)
    (store / "surfaces").mkdir(parents=True, exist_ok=True)
    (store / "surface-info").mkdir(parents=True, exist_ok=True)

    # --- anatomical volume(s): .mgz -> .nii.gz via nibabel (T1 as 'raw') -----
    for mgz, name in [("T1", "raw"), ("wm", "raw_wm")]:
        src = fs_mri / f"{mgz}.mgz"
        if src.exists():
            img = nib.load(str(src))
            nib.save(nib.Nifti1Image(np.asarray(img.dataobj), img.affine),
                     str(store / "anatomicals" / f"{name}.nii.gz"))
            print(f"    anatomicals/{name}.nii.gz <- {mgz}.mgz")

    # --- surfaces: wm / pia / inflated / fiducial ----------------------------
    geom = {}
    for hemi in ("lh", "rh"):
        wm_pts, polys = nib.freesurfer.read_geometry(str(fs_surf / f"{hemi}.smoothwm"))
        pia_pts, _ = nib.freesurfer.read_geometry(str(fs_surf / f"{hemi}.pial"))
        inf_pts, _ = nib.freesurfer.read_geometry(str(fs_surf / f"{hemi}.inflated"))
        formats.write_gii(str(store / "surfaces" / f"wm_{hemi}.gii"), pts=wm_pts, polys=polys)
        formats.write_gii(str(store / "surfaces" / f"pia_{hemi}.gii"), pts=pia_pts, polys=polys)
        formats.write_gii(str(store / "surfaces" / f"inflated_{hemi}.gii"), pts=inf_pts, polys=polys)
        formats.write_gii(str(store / "surfaces" / f"fiducial_{hemi}.gii"),
                          pts=(wm_pts + pia_pts) / 2.0, polys=polys)
        geom[hemi] = polys
        print(f"    surfaces/{{wm,pia,inflated,fiducial}}_{hemi}.gii  ({len(wm_pts)} verts)")

    # --- surface-info: curvature / sulcaldepth / thickness -------------------
    def morph(name):
        return {h: nib.freesurfer.read_morph_data(str(fs_surf / f"{h}.{name}")) for h in ("lh", "rh")}

    for fsname, outname, sign in [("curv", "curvature", -1.0),
                                  ("sulc", "sulcaldepth", -1.0),
                                  ("thickness", "thickness", 1.0)]:
        try:
            m = morph(fsname)
            np.savez(str(store / "surface-info" / f"{outname}.npz"),
                     left=sign * m["lh"], right=sign * m["rh"])
            print(f"    surface-info/{outname}.npz")
        except FileNotFoundError:
            print(f"    (skip {fsname}: not found)")

    # re-init db so it sees the new subject, then import the flat patch
    database.db = database.Database()
    print(f"[*] import_flat patch={args.patch}")
    fs.import_flat(args.fs_subject, args.patch, cx_subject=cx,
                   freesurfer_subject_dir=str(FSDIR), auto_overwrite=True)
    database.db = database.Database()

    # sanity: load flat + fiducial
    for hemi in (0, 1):
        flat = cortex.db.get_surf(cx, "flat", hemisphere=("lh" if hemi == 0 else "rh"))
        print(f"    flat {'lh' if hemi==0 else 'rh'}: pts {flat[0].shape}, polys {flat[1].shape}")
    print(f"[done] pycortex subject '{cx}' ready in {store}")


if __name__ == "__main__":
    main()
