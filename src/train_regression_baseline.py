"""
train_regression_baseline.py
============================
Direct CLIP+DINOv2 → fMRI regression baseline (no flow / source / ODE).

Bounds the conditional-mean ceiling of the visual features so we can decide
whether the FactFlow ~0.37 plateau is a feature limit or a decoder limit:

  * regression  >> 0.37  → the flow is leaving performance on the table.
  * regression  ≈  0.37  → 0.37 is the feature ceiling (decoder changes won't help).

Loss: masked, SNR-weighted MSE (the conditional-mean objective). Validation is
identical to the flow trainer (rep-averaged GT, voxel-wise Pearson r).

Usage::

    python src/train_regression_baseline.py --config src/configs/regression_baseline.yaml
"""

import argparse
import csv
import math
import os
import sys
from time import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.factflow_fmri_dataset import FactFlowfMRIDataset
from utils.checkpoint import save_checkpoint
from utils.config_utils import instantiate_from_config
from utils.fmri_utils import create_pad_mask
from utils.logging_utils import create_logger
from utils.metrics import masked_mse, voxel_pearson, compute_voxel_reliability
from utils.training_utils import build_optimizer_and_scheduler


def build_voxel_weight(train_ds, n_voxels, pad_to, device, logger):
    """Per-voxel noise-ceiling weight over pad_to (mean over real voxels = 1)."""
    nc = compute_voxel_reliability(train_ds.fmri_data, train_ds.n_reps)  # (V,)
    w = np.zeros(pad_to, dtype=np.float64)
    w[:n_voxels] = nc
    mean_w = w[:n_voxels].mean()
    if mean_w > 0:
        w = w / mean_w
    logger.info("SNR-weighted MSE ON: noise-ceiling mean=%.3f median=%.3f",
                float(nc.mean()), float(np.median(nc)))
    return torch.tensor(w, dtype=torch.float32, device=device)


@torch.no_grad()
def validate(model, loader, pad_mask, device, autocast_kwargs, use_dino):
    model.eval()
    preds, gts = [], []
    for batch in loader:
        fmri = batch["fmri"].to(device)
        clip_pool = batch["clip_pool"].to(device)
        clip_tok = batch["clip_tokens"].to(device)
        dino_tok = batch["dino_tokens"].to(device) if use_dino else None
        B = fmri.shape[0]
        with autocast(**autocast_kwargs):
            pred = model(clip_pool, clip_tok, dino_tok).float()
        preds.append(pred.reshape(B, -1)[:, pad_mask].cpu())
        gts.append(fmri.reshape(B, -1)[:, pad_mask].cpu())
    preds = torch.cat(preds, 0)
    gts = torch.cat(gts, 0)
    model.train()
    return voxel_pearson(preds, gts).mean().item()


def main() -> None:
    ap = argparse.ArgumentParser(description="CLIP+DINO→fMRI regression baseline")
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--exps_dir", type=str, default="exps")
    ap.add_argument("--exp_name", type=str, default=None)
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--max_steps", type=int, default=None)
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
    train_cfg = OmegaConf.to_container(cfg.training, resolve=True)
    loss_cfg = OmegaConf.to_container(cfg.get("losses", {}), resolve=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(train_cfg.get("global_seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── Native sizing: pad_to = ceil(n_voxels / patch_size) * patch_size ──
    n_voxels = int(data_cfg["n_voxels"])
    patch_size = int(cfg.model.params.patch_size)
    pad_to = math.ceil(n_voxels / patch_size) * patch_size
    data_cfg["pad_to"] = pad_to
    cfg.model.params.seq_len = pad_to

    exp_name = args.exp_name or f"regression_sub{data_cfg['subject']}"
    exp_dir = os.path.join(args.exps_dir, exp_name)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = create_logger(exp_dir, name="regression")
    logger.info("native sizing: n_voxels=%d → pad_to=%d (seq_len=%d), patch=%d",
                n_voxels, pad_to, pad_to, patch_size)
    if not os.path.exists(os.path.join(exp_dir, "config.yaml")):
        OmegaConf.save(cfg, os.path.join(exp_dir, "config.yaml"))

    # ── Data ──
    use_dino = data_cfg.get("dino_feature") is not None
    ds_kwargs = dict(
        data_dir=data_cfg["data_dir"], subject=data_cfg["subject"],
        fmri_mode=data_cfg["fmri_mode"], clip_feature=data_cfg["clip_feature"],
        n_voxels=n_voxels, pad_to=pad_to,
        dino_feature=data_cfg.get("dino_feature"),
    )
    train_ds = FactFlowfMRIDataset(mode="train", avg_reps=data_cfg.get("avg_reps", True), **ds_kwargs)
    val_ds = FactFlowfMRIDataset(mode="test", avg_reps=True, **ds_kwargs)
    bs = int(train_cfg.get("batch_size", 32))
    nw = int(train_cfg.get("num_workers", 4))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw,
                              pin_memory=True, drop_last=True, persistent_workers=nw > 0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    logger.info("Train: %d samples, Val(avg_reps): %d images", len(train_ds), len(val_ds))

    # ── Model / optim ──
    model = instantiate_from_config(cfg.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Regressor params: %.2fM", n_params / 1e6)

    grad_accum = int(train_cfg.get("grad_accum_steps", 1))
    steps_per_epoch = max(len(train_loader) // grad_accum, 1)
    epochs = int(train_cfg.get("epochs", 100))
    optimizer, scheduler, m1, m2 = build_optimizer_and_scheduler(
        model.parameters(), train_cfg, steps_per_epoch, epochs)
    logger.info(m1); logger.info(m2)

    pad_mask = create_pad_mask(n_voxels, pad_to, device)
    voxel_weight = (build_voxel_weight(train_ds, n_voxels, pad_to, device, logger)
                    if loss_cfg.get("use_snr_weight", True) else None)

    use_bf16 = train_cfg.get("precision", "fp32") == "bf16"
    autocast_kwargs = dict(device_type=device.split(":")[0], dtype=torch.bfloat16, enabled=use_bf16)
    clip_grad = float(train_cfg.get("clip_grad", 1.0))
    val_every = int(train_cfg.get("val_every", 5))
    log_every = int(train_cfg.get("log_every", 50))

    history = open(os.path.join(exp_dir, "history.csv"), "a", newline="")
    writer = csv.writer(history)
    if os.stat(os.path.join(exp_dir, "history.csv")).st_size == 0:
        writer.writerow(["epoch", "step", "train_mse", "val_voxel_r", "lr"]); history.flush()

    # ── Train loop ──
    model.train()
    train_steps, best_r = 0, -1.0
    logger.info("Starting regression training for %d epochs, grad_accum=%d", epochs, grad_accum)
    for epoch in range(epochs):
        accum, run_loss, run_n = 0, 0.0, 0
        ep_loss, ep_n = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", dynamic_ncols=True)
        for batch in pbar:
            fmri = batch["fmri"].to(device)
            clip_pool = batch["clip_pool"].to(device)
            clip_tok = batch["clip_tokens"].to(device)
            dino_tok = batch["dino_tokens"].to(device) if use_dino else None
            with autocast(**autocast_kwargs):
                pred = model(clip_pool, clip_tok, dino_tok)
                loss = masked_mse(pred, fmri, pad_mask, weight=voxel_weight)
            (loss / grad_accum).backward()
            accum += 1
            run_loss += loss.item(); run_n += 1
            ep_loss += loss.item(); ep_n += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
            if accum < grad_accum:
                continue
            accum = 0
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            train_steps += 1
            if train_steps % log_every == 0:
                logger.info("[step=%06d ep=%d] train_mse=%.5f lr=%.2e",
                            train_steps, epoch, run_loss / max(run_n, 1), scheduler.get_last_lr()[0])
                run_loss, run_n = 0.0, 0
            if args.max_steps and train_steps >= args.max_steps:
                break

        if (epoch + 1) % val_every == 0 or (epoch + 1) == epochs:
            val_r = validate(model, val_loader, pad_mask, device, autocast_kwargs, use_dino)
            lr = scheduler.get_last_lr()[0]
            logger.info("[Val ep=%d] voxel_r=%.4f  (train_mse=%.5f)", epoch + 1, val_r, ep_loss / max(ep_n, 1))
            writer.writerow([epoch + 1, train_steps, f"{ep_loss / max(ep_n, 1):.6f}",
                             f"{val_r:.6f}", f"{lr:.2e}"]); history.flush()
            if val_r > best_r:
                best_r = val_r
                save_checkpoint(os.path.join(ckpt_dir, "best.pt"), model, optimizer,
                                scheduler, train_steps, epoch, best_r)
                logger.info("New best voxel_r: %.4f", best_r)

        if args.max_steps and train_steps >= args.max_steps:
            logger.info("Reached max_steps=%d, stopping.", args.max_steps)
            break

    save_checkpoint(os.path.join(ckpt_dir, f"final-{train_steps}.pt"), model, optimizer,
                    scheduler, train_steps, epochs, best_r)
    history.close()
    logger.info("Done. Best val voxel_r=%.4f", best_r)


if __name__ == "__main__":
    main()
