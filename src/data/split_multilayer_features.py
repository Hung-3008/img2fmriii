"""
split_multilayer_features.py
============================
Subsample the *layer* axis of a multilayer image-feature file
``(N, L, T, D)`` → ``(N, n, T, D)`` and write the result to a new ``.npy``.

Why: DINOv2/CLIP ``_multilayer_`` features can be huge (e.g. vitg14 16-layer =
113 GB) — too big for GPU memory and too big for the OS page cache, which forces
random reads off a spinning HDD during training (see the I/O bottleneck note).
Dropping to 8 layers roughly halves the size; writing the output straight to an
NVMe SSD then symlinking it in removes the disk bottleneck (mirrors the CLIP fix).

The copy is done in chunks through ``mmap`` / ``open_memmap`` so neither the full
input nor the full output is ever held in RAM.

Layer selection (which ``n`` of the stored ``L`` to keep):
  * default: ``np.linspace(0, L-1, n)`` rounded → evenly spaced, keeps the
    shallowest AND deepest layer (best hierarchy coverage). For L=16, n=8 →
    [0, 2, 4, 6, 9, 11, 13, 15].
  * ``--indices i j k …`` overrides with explicit positions.

Examples:
  python src/data/split_multilayer_features.py \
    --src NSD/data/nsd/subj01/dino/nsd_dinov2_vitg14_multilayer_train_sub1.npy \
    --dst /home/hung/nsd_feat/subj01/dino/nsd_dinov2_vitg14_multilayer8_train_sub1.npy \
    --n_layers 8
"""
import argparse
import os
import numpy as np


def select_layers(L: int, n_layers: int, indices) -> list:
    if indices:
        sel = [int(i) for i in indices]
    else:
        sel = np.unique(np.linspace(0, L - 1, n_layers).round().astype(int)).tolist()
    assert all(0 <= i < L for i in sel), f"layer indices out of range [0,{L}): {sel}"
    assert len(sel) == len(set(sel)), f"duplicate layer indices: {sel}"
    return sel


def main():
    ap = argparse.ArgumentParser(description="Subsample the layer axis of an (N,L,T,D) feature file.")
    ap.add_argument("--src", required=True, help="Input .npy (N, L, T, D)")
    ap.add_argument("--dst", required=True, help="Output .npy path (parent dirs created)")
    ap.add_argument("--n_layers", type=int, default=8, help="How many layers to keep (default 8)")
    ap.add_argument("--indices", type=int, nargs="+", default=None,
                    help="Explicit layer indices to keep (overrides --n_layers)")
    ap.add_argument("--chunk", type=int, default=128, help="Images per copy chunk (RAM control)")
    args = ap.parse_args()

    src = np.load(args.src, mmap_mode="r")
    assert src.ndim == 4, f"expected (N, L, T, D), got {src.shape}"
    N, L, T, D = src.shape
    sel = select_layers(L, args.n_layers, args.indices)
    sel_arr = np.asarray(sel)

    print(f"src: {args.src}")
    print(f"  shape {tuple(src.shape)} {src.dtype}")
    print(f"keep {len(sel)} of {L} layers: {sel}")

    out_shape = (N, len(sel), T, D)
    os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)
    dst = np.lib.format.open_memmap(args.dst, mode="w+", dtype=src.dtype, shape=out_shape)

    for i in range(0, N, args.chunk):
        j = min(i + args.chunk, N)
        dst[i:j] = np.asarray(src[i:j])[:, sel_arr]
        if (i // args.chunk) % 10 == 0:
            print(f"  {j}/{N}", flush=True)
    dst.flush()
    del dst

    sz = os.path.getsize(args.dst) / 1e9
    print(f"✅ wrote {args.dst}\n  shape {out_shape}  {sz:.1f} GB")


if __name__ == "__main__":
    main()
