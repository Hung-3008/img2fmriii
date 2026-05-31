"""
eval_ema_vs_raw.py
==================
So sánh PCC của EMA model vs raw model trên checkpoint hiện có.
Mục đích: xác nhận giả thuyết EMA chưa ấm là nguyên nhân PCC≈0.

Usage:
    python src/eval_ema_vs_raw.py \
        --config src/configs/factflow_fmri.yaml \
        --ckpt exps/factflow_1d_baseline/checkpoints/last-1000.pt \
        --n_samples 200
"""

import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import autocast
from torch.utils.data import DataLoader

from data.factflow_fmri_dataset import FactFlowfMRIDataset
from model.factflow_factory import build_models, build_transport, build_sampler
from utils.fmri_utils import create_pad_mask, get_latent_size
from utils.metrics import pearson_corr_per_sample, masked_mse


def run_eval(model, sample_fn, loader, latent_size, pad_mask, device,
             autocast_kwargs, use_source, label, n_samples):
    model.eval()
    all_corr = []
    all_mse = []
    seen = 0

    with torch.no_grad():
        for batch in loader:
            if seen >= n_samples:
                break
            fmri_gt = batch["fmri"].to(device)
            clip_tok = batch["clip_tokens"].to(device)
            clip_pool = batch["clip_pool"].to(device)
            B = fmri_gt.shape[0]
            remain = n_samples - seen
            if B > remain:
                fmri_gt = fmri_gt[:remain]
                clip_tok = clip_tok[:remain]
                clip_pool = clip_pool[:remain]
                B = remain

            if use_source:
                x0_tok, _, _ = model.encode_source(clip_tok)
                x0 = x0_tok.permute(0, 2, 1).contiguous().view(B, *latent_size)
            else:
                x0 = torch.randn(B, *latent_size, device=device)

            with autocast(**autocast_kwargs):
                traj = sample_fn(x0, model.dit.forward, y=clip_pool)
            pred = traj[-1]

            corr = pearson_corr_per_sample(pred, fmri_gt, pad_mask)
            mse  = masked_mse(pred, fmri_gt, pad_mask).item()
            all_corr.extend(corr.cpu().tolist())
            all_mse.append(mse)
            seen += B

    mean_pcc = float(np.mean(all_corr))
    mean_mse = float(np.mean(all_mse))
    std_pcc  = float(np.std(all_corr))
    print(f"[{label:>10s}]  n={seen:4d}  "
          f"PCC mean={mean_pcc:+.5f}  std={std_pcc:.5f}  MSE={mean_mse:.5f}")
    return mean_pcc, mean_mse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/configs/factflow_fmri.yaml")
    parser.add_argument("--ckpt",   default="exps/factflow_1d_baseline/checkpoints/last-1000.pt")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(args.config)
    data_cfg   = OmegaConf.to_container(cfg.data, resolve=True)
    train_cfg  = OmegaConf.to_container(cfg.training, resolve=True)

    # ── Dataset ──────────────────────────────────────────────────────────
    ds = FactFlowfMRIDataset(
        data_dir=data_cfg["data_dir"], subject=data_cfg["subject"],
        mode="test", fmri_mode=data_cfg["fmri_mode"],
        clip_feature=data_cfg["clip_feature"],
        n_voxels=data_cfg["n_voxels"], pad_to=data_cfg["pad_to"],
        fmri_channels=data_cfg.get("fmri_channels", 1),
        fmri_spatial=data_cfg.get("fmri_spatial", None),
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    n_eval = min(args.n_samples, len(ds))

    # ── Geometry ─────────────────────────────────────────────────────────
    latent_size = get_latent_size(data_cfg)
    pad_mask = create_pad_mask(data_cfg["n_voxels"], data_cfg["pad_to"], device)

    # ── Models ───────────────────────────────────────────────────────────
    wrapper, ema = build_models(cfg, device)

    # ── Load checkpoint ──────────────────────────────────────────────────
    ckpt = torch.load(args.ckpt, map_location=device)
    wrapper.load_state_dict(ckpt["model"])
    ema.load_state_dict(ckpt["ema"])
    train_steps = int(ckpt.get("train_steps", 0))
    epoch       = int(ckpt.get("epoch", 0))
    print(f"Loaded: {args.ckpt}")
    print(f"  train_steps={train_steps}  epoch={epoch}")
    print(f"  ema_decay={train_cfg['ema_decay']}  "
          f"effective_ema_weight={train_cfg['ema_decay']**train_steps:.4f}")
    print()

    # ── Sampler ──────────────────────────────────────────────────────────
    transport  = build_transport(cfg, latent_size)
    sample_fn  = build_sampler(transport, cfg.sampler)
    use_source = bool(cfg.get("use_source_encoder", False))

    use_bf16 = train_cfg.get("precision", "fp32") == "bf16"
    autocast_kwargs = dict(
        device_type=device.split(":")[0],
        dtype=torch.bfloat16,
        enabled=use_bf16,
    )

    # ── Eval EMA ─────────────────────────────────────────────────────────
    run_eval(ema, sample_fn, loader, latent_size, pad_mask, device,
             autocast_kwargs, use_source, "EMA", n_eval)

    # ── Eval raw model ────────────────────────────────────────────────────
    run_eval(wrapper, sample_fn, loader, latent_size, pad_mask, device,
             autocast_kwargs, use_source, "RAW", n_eval)

    # ── Bonus: naive baseline (mean fMRI prediction) ──────────────────────
    # Dự đoán bằng vector 0 (đầu ra của mạng chưa học) → PCC lý thuyết ≈ 0
    print()
    print("[THEORY] EMA weight fraction from init (should be close to 0 for good training):")
    print(f"  decay^steps = {train_cfg['ema_decay']}^{train_steps} "
          f"= {train_cfg['ema_decay']**train_steps:.6f}")
    expected_raw_fraction = 1 - train_cfg['ema_decay']**train_steps
    print(f"  => ~{expected_raw_fraction*100:.2f}% bắt nguồn từ update thực (không tính init)")


if __name__ == "__main__":
    main()
