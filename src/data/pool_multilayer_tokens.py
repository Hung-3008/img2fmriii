"""
pool_multilayer_tokens.py
=========================
Average-pool the *spatial token* axis of a ViT feature file, shrinking the
257-token grid (1 CLS + 16×16 patches) down to 65 (1 CLS + 8×8) — or any
``factor`` — and write the result to a new ``.npy``.

    (N, L, 257, D)  --2×2 avg-pool-->  (N, L, 65, D)
    (N,    257, D)  --2×2 avg-pool-->  (N,    65, D)

Why: the multilayer ``_multilayer*_`` files are huge because they keep the FULL
spatial grid × every layer (e.g. DINO-4L = 9000×4×257×1536 = 26.5 GiB in fp16).
The grid is highly redundant — adjacent ViT patches are near-duplicates and the
fMRI's own resolution can't exploit 16×16 anyway — so a 2×2 average pool divides
storage / RAM / cross-attn cost by ~4 (257→65) at a small accuracy cost
(spatial layout is preserved, only the finest retinotopic detail is coarsened).

How the token axis is handled (``T = cls + G²``):
  * the first ``--cls`` tokens (default 1 = the CLS) are passed through unchanged
  * the remaining ``G²`` patch tokens are reshaped to the ``G×G`` raster grid,
    average-pooled in non-overlapping ``factor×factor`` blocks, then flattened
    back. Pooling is done in float32 then cast back to the source dtype.

Examples:
  # DINO 4-layer 257 → 65 tokens
  python src/data/pool_multilayer_tokens.py \
    --src /home/hung/nsd_feat/subj01/dino/nsd_dinov2_vitg14_multilayer4_train_sub1.npy \
    --dst NSD/data/nsd/subj01/dino/nsd_dinov2_vitg14_multilayer4p_train_sub1.npy

  # explicit 4×4 pool (257 → 17), or keep no CLS (--cls 0)
  python src/data/pool_multilayer_tokens.py --src ... --dst ... --factor 4
"""
import argparse
import math
import os
import numpy as np


def pool_patches(patch: np.ndarray, g: int, f: int) -> np.ndarray:
    """Average-pool the (… , G², D) patch tokens in f×f blocks → (… , (G/f)², D).

    Computed in float32 for a stable mean, returned in float32.
    """
    lead = patch.shape[:-2]          # (chunk[, L])
    d = patch.shape[-1]
    x = patch.astype(np.float32).reshape(*lead, g, g, d)
    go = g // f
    # (…, go, f, go, f, D) → mean over the two block axes
    x = x.reshape(*lead, go, f, go, f, d).mean(axis=(-4, -2))
    return x.reshape(*lead, go * go, d)


def main():
    ap = argparse.ArgumentParser(
        description="Average-pool the spatial token grid of a ViT feature file (e.g. 257→65)."
    )
    ap.add_argument("--src", required=True, help="Input .npy (N, L, T, D) or (N, T, D)")
    ap.add_argument("--dst", required=True, help="Output .npy path (parent dirs created)")
    ap.add_argument("--factor", type=int, default=2,
                    help="Spatial pool factor per axis (default 2 → 16×16→8×8, 257→65)")
    ap.add_argument("--cls", type=int, default=1,
                    help="Number of leading non-spatial tokens passed through (default 1 = CLS)")
    ap.add_argument("--chunk", type=int, default=128, help="Images per chunk (RAM control)")
    args = ap.parse_args()

    src = np.load(args.src, mmap_mode="r")
    assert src.ndim in (3, 4), f"expected (N,L,T,D) or (N,T,D), got {src.shape}"
    has_layers = src.ndim == 4
    N, T, D = (src.shape[0], src.shape[-2], src.shape[-1])
    L = src.shape[1] if has_layers else None

    # T = cls + G²  → solve for the grid side G
    n_patch = T - args.cls
    g = int(round(math.sqrt(n_patch)))
    assert g * g == n_patch, (
        f"patch count {n_patch} (= T{T} - cls{args.cls}) is not a perfect square; "
        f"check --cls"
    )
    assert g % args.factor == 0, f"grid {g}×{g} not divisible by factor {args.factor}"
    go = g // args.factor
    T_out = args.cls + go * go

    out_shape = (N, L, T_out, D) if has_layers else (N, T_out, D)
    print(f"src: {args.src}")
    print(f"  shape {tuple(src.shape)} {src.dtype}")
    print(f"pool grid {g}×{g} → {go}×{go} (factor {args.factor}), cls={args.cls}: "
          f"{T} → {T_out} tokens")
    print(f"  out shape {out_shape}  (~{np.prod(out_shape) * src.dtype.itemsize / 1024**3:.1f} GiB)")

    os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)
    dst = np.lib.format.open_memmap(args.dst, mode="w+", dtype=src.dtype, shape=out_shape)

    for i in range(0, N, args.chunk):
        j = min(i + args.chunk, N)
        blk = np.asarray(src[i:j])                 # (chunk[, L], T, D)
        cls = blk[..., : args.cls, :]              # (chunk[, L], cls, D)
        patch = blk[..., args.cls:, :]             # (chunk[, L], G², D)
        pooled = pool_patches(patch, g, args.factor)
        merged = np.concatenate([cls.astype(np.float32), pooled], axis=-2)
        dst[i:j] = merged.astype(src.dtype)
        if (i // args.chunk) % 10 == 0:
            print(f"  {j}/{N}", flush=True)
    dst.flush()
    del dst

    sz = os.path.getsize(args.dst) / 1024**3
    print(f"✅ wrote {args.dst}\n  shape {out_shape}  {sz:.1f} GiB")


if __name__ == "__main__":
    main()
