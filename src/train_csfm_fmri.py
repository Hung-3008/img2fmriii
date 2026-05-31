"""
train_csfm_fmri.py
==================
Training script for CSFM-based fMRI synthesis.

CLIP image features → PerceiverVE (learned source x₀) → Flow Matching → fMRI

Adapted from reproduces/CSFM/src/train_t2i.py with:
  - No RAE/stage-1 encoder (operates directly on fMRI voxels)
  - No CLIP text encoder (uses pre-extracted CLIP visual features)
  - Masked loss on padded voxels
  - Single-GPU training (no DDP by default)
  - Pearson correlation evaluation
"""

import argparse
import csv
import logging
import math
import os
import sys
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Add CSFM source to path so we can import its modules directly
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
CSFM_SRC = os.path.join(PROJECT_ROOT, "reproduces", "CSFM", "src")
if CSFM_SRC not in sys.path:
    sys.path.insert(0, CSFM_SRC)

from stage2.transport import create_transport, Sampler, ModelType
from stage2.text_encoders.regularization_loss import kld_loss_factory
from stage2 import Wrapper
from utils.optim_utils import build_optimizer, build_scheduler

# Local imports
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
from data.csfm_fmri_dataset import CSFMfMRIDataset

# ==========================================================================
# Helpers
# ==========================================================================

def get_obj_from_str(string):
    """Resolve 'module.ClassName' → class object."""
    module_path, cls_name = string.rsplit(".", 1)
    return getattr(__import__(module_path, fromlist=[cls_name]), cls_name)


def instantiate_from_config(config):
    """Instantiate a class from an OmegaConf config section."""
    target = config["target"]
    params = OmegaConf.to_container(config.get("params", {}), resolve=True)
    return get_obj_from_str(target)(**params)


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def masked_mse(pred, target, pad_mask, fmri_channels, fmri_spatial):
    """Compute MSE only on real (non-padded) voxels.

    Args:
        pred, target: (B, C, H, W)
        pad_mask: (V_pad,) boolean — True for real voxels
        fmri_channels, fmri_spatial: reshape parameters
    """
    B = pred.shape[0]
    pred_flat = pred.reshape(B, -1)       # (B, C*H*W)
    target_flat = target.reshape(B, -1)   # (B, C*H*W)
    mask = pad_mask.to(pred.device)       # (V_pad,)
    diff_sq = (pred_flat - target_flat) ** 2  # (B, V_pad)
    # Mask: zero out padded positions, average over real voxels
    masked = diff_sq * mask.unsqueeze(0).float()
    return masked.sum() / (mask.sum() * B)


@torch.no_grad()
def pearson_corr_per_sample(pred, target, pad_mask):
    """Per-sample Pearson r between pred and target fMRI (profile correlation).

    Args:
        pred, target: (B, C, H, W)
        pad_mask: (V_pad,)
    Returns:
        (B,) tensor of Pearson r values
    """
    B = pred.shape[0]
    pred_flat = pred.reshape(B, -1)[:, pad_mask]
    target_flat = target.reshape(B, -1)[:, pad_mask]
    pred_centered = pred_flat - pred_flat.mean(dim=1, keepdim=True)
    target_centered = target_flat - target_flat.mean(dim=1, keepdim=True)
    num = (pred_centered * target_centered).sum(dim=1)
    den = torch.sqrt(
        (pred_centered ** 2).sum(dim=1) * (target_centered ** 2).sum(dim=1)
    )
    return num / (den + 1e-8)


def create_logger(log_dir):
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        handlers = [
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(log_dir, "train.log")),
        ]
    else:
        handlers = [logging.StreamHandler()]
    logging.basicConfig(
        level=logging.INFO,
        format="[\033[34m%(asctime)s\033[0m] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )
    return logging.getLogger(__name__)


# ==========================================================================
# Main
# ==========================================================================

def main():
    parser = argparse.ArgumentParser(description="CSFM fMRI Synthesis Training")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--exps_dir", type=str, default="exps")
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--ckpt", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--resume_last", action="store_true")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--max_steps", type=int, default=None, help="Stop after N steps (for debugging)")
    args = parser.parse_args()

    # --- Load config ---
    cfg = OmegaConf.load(args.config)
    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
    train_cfg = OmegaConf.to_container(cfg.training, resolve=True)
    loss_cfg = OmegaConf.to_container(cfg.losses, resolve=True)
    transport_cfg = OmegaConf.to_container(cfg.transport.get("params", {}), resolve=True)
    sampler_cfg = OmegaConf.to_container(cfg.sampler, resolve=True)

    # --- Device ---
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # --- Seed ---
    seed = int(train_cfg.get("global_seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)

    # --- Experiment directory ---
    if args.exp_name is None:
        args.exp_name = f"csfm_fmri_sub{data_cfg['subject']}"
    exp_dir = os.path.join(args.exps_dir, args.exp_name)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = create_logger(exp_dir)

    # Save config
    config_save_path = os.path.join(exp_dir, "config.yaml")
    if not os.path.exists(config_save_path):
        OmegaConf.save(cfg, config_save_path)

    # --- Dataset ---
    train_ds = CSFMfMRIDataset(
        data_dir=data_cfg["data_dir"],
        subject=data_cfg["subject"],
        mode="train",
        fmri_mode=data_cfg["fmri_mode"],
        clip_feature=data_cfg["clip_feature"],
        n_voxels=data_cfg["n_voxels"],
        pad_to=data_cfg["pad_to"],
        fmri_channels=data_cfg["fmri_channels"],
        fmri_spatial=data_cfg["fmri_spatial"],
    )
    test_ds = CSFMfMRIDataset(
        data_dir=data_cfg["data_dir"],
        subject=data_cfg["subject"],
        mode="test",
        fmri_mode=data_cfg["fmri_mode"],
        clip_feature=data_cfg["clip_feature"],
        n_voxels=data_cfg["n_voxels"],
        pad_to=data_cfg["pad_to"],
        fmri_channels=data_cfg["fmri_channels"],
        fmri_spatial=data_cfg["fmri_spatial"],
    )

    batch_size = int(train_cfg.get("batch_size", 64))
    num_workers = int(train_cfg.get("num_workers", 4))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )

    logger.info(f"Train: {len(train_ds)} samples, Test: {len(test_ds)} samples")
    logger.info(f"Batch size: {batch_size}, Steps/epoch: {len(train_loader)}")

    # --- Pad mask (shared) ---
    pad_mask = torch.zeros(data_cfg["pad_to"], dtype=torch.bool)
    pad_mask[: data_cfg["n_voxels"]] = True
    pad_mask = pad_mask.to(device)

    # --- Models ---
    dit = instantiate_from_config(cfg.stage_2)
    source_encoder = instantiate_from_config(cfg.source_encoder)

    dit_params = sum(p.numel() for p in dit.parameters())
    se_params = sum(p.numel() for p in source_encoder.parameters())
    logger.info(f"DiT params: {dit_params / 1e6:.2f}M")
    logger.info(f"SourceEncoder params: {se_params / 1e6:.2f}M")
    logger.info(f"Total trainable: {(dit_params + se_params) / 1e6:.2f}M")

    # Wrap for unified state_dict
    wrapper = Wrapper(dit=dit, source_encoder=source_encoder).to(device)
    ema = deepcopy(wrapper).to(device)

    # --- Transport ---
    fmri_spatial = data_cfg["fmri_spatial"]
    fmri_channels = data_cfg["fmri_channels"]
    latent_size = (fmri_channels, fmri_spatial, fmri_spatial)
    shift_dim = math.prod(latent_size)
    shift_base = transport_cfg.pop("time_dist_shift", 4096)
    time_dist_shift = math.sqrt(shift_dim / shift_base)

    transport = create_transport(
        **transport_cfg,
        time_dist_shift=time_dist_shift,
    )
    assert transport.model_type == ModelType.VELOCITY

    # --- Sampler for eval ---
    transport_sampler = Sampler(transport)
    sampler_mode = sampler_cfg.get("mode", "ODE").upper()
    sampler_params = dict(sampler_cfg.get("params", {}))
    if sampler_mode == "ODE":
        eval_sampler_fn = transport_sampler.sample_ode(**sampler_params)
    else:
        raise NotImplementedError(f"Sampler mode {sampler_mode}")

    # --- Optimizer & Scheduler ---
    opt, opt_msg = build_optimizer(wrapper.parameters(), train_cfg)
    epochs = int(train_cfg.get("epochs", 200))
    grad_accum = int(train_cfg.get("grad_accum_steps", 1))
    steps_per_epoch = len(train_loader) // grad_accum
    if steps_per_epoch <= 0:
        steps_per_epoch = 1

    # Patch warmup into train_cfg for build_scheduler
    train_cfg.setdefault("decay_start_epoch", 0)
    if "warmup_steps" not in train_cfg:
        train_cfg["warmup_steps"] = int(train_cfg.get("warmup_steps", 500))
    if "decay_end_epoch" not in train_cfg:
        train_cfg["decay_end_epoch"] = epochs

    # build_scheduler expects these under "scheduler" sub-key or top-level
    sched_cfg_for_builder = dict(train_cfg)
    sched_cfg_for_builder["scheduler"] = {
        "warmup_steps": train_cfg.get("warmup_steps", 500),
        "decay_end_steps": epochs * steps_per_epoch,
    }
    sched, sched_msg = build_scheduler(opt, steps_per_epoch, sched_cfg_for_builder)
    logger.info(opt_msg)
    logger.info(sched_msg)

    # --- Load checkpoint ---
    train_steps = 0
    start_epoch = 0
    best_val_pcc = -1.0

    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location="cpu")
        wrapper.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])
        if "scheduler" in ckpt:
            sched.load_state_dict(ckpt["scheduler"])
        train_steps = int(ckpt.get("train_steps", 0))
        start_epoch = int(ckpt.get("epoch", 0))
        best_val_pcc = float(ckpt.get("best_val_pcc", -1.0))
        logger.info(f"Resumed from {args.ckpt}, step={train_steps}, epoch={start_epoch}, best_val_pcc={best_val_pcc:.4f}")
        del ckpt
        if device == "cuda":
            torch.cuda.empty_cache()

    if args.resume_last:
        last_ckpts = glob(os.path.join(ckpt_dir, "last-*.pt"))
        if last_ckpts:
            last_ckpt_path = max(last_ckpts, key=lambda p: int(p.split("last-")[-1].replace(".pt", "")))
            ckpt = torch.load(last_ckpt_path, map_location="cpu")
            wrapper.load_state_dict(ckpt["model"])
            ema.load_state_dict(ckpt["ema"])
            if "opt" in ckpt:
                opt.load_state_dict(ckpt["opt"])
            if "scheduler" in ckpt:
                sched.load_state_dict(ckpt["scheduler"])
            train_steps = int(ckpt.get("train_steps", 0))
            start_epoch = int(ckpt.get("epoch", 0))
            best_val_pcc = float(ckpt.get("best_val_pcc", -1.0))
            logger.info(f"Resumed last from {last_ckpt_path}, step={train_steps}, best_val_pcc={best_val_pcc:.4f}")
            del ckpt
            if device == "cuda":
                torch.cuda.empty_cache()

    # --- Training config ---
    use_bf16 = train_cfg.get("precision", "fp32") == "bf16"
    autocast_kwargs = dict(device_type=device.split(":")[0], dtype=torch.bfloat16, enabled=use_bf16)
    clip_grad_val = float(train_cfg.get("clip_grad", 1.0))
    ema_decay = float(train_cfg.get("ema_decay", 0.9999))
    log_every = int(train_cfg.get("log_every", 50))
    ckpt_every = int(train_cfg.get("ckpt_every", 5000))
    sample_every = int(train_cfg.get("sample_every", 2000))
    val_every = int(train_cfg.get("val_every", 1))

    # Loss config
    use_kld = loss_cfg.get("use_kld_loss", True)
    kld_weight = float(loss_cfg.get("kld_loss_weight", 5.0))
    kld_type = loss_cfg.get("kld_loss_type", "var_kld")
    kld_reduction = loss_cfg.get("kld_reduction", "mean")
    kld_target_std = float(loss_cfg.get("kld_target_std", 1.0))
    use_align = loss_cfg.get("use_align_loss", True)
    align_weight = float(loss_cfg.get("align_loss_weight", 1.0))
    align_type = loss_cfg.get("align_loss_type", "normalized_l2")
    detach_ut = loss_cfg.get("detach_ut", False)
    clip_logvar_min = float(loss_cfg.get("clip_logvar_min", -10.0))
    clip_logvar_max = float(loss_cfg.get("clip_logvar_max", 10.0))

    # --- Init EMA ---
    update_ema(ema, wrapper, decay=0)
    wrapper.train()
    ema.eval()

    # --- CSV history logging ---
    history_path = os.path.join(exp_dir, "history.csv")

    # Initialize or append to history.csv
    history_exists = os.path.exists(history_path)
    history_file = open(history_path, "a", newline="")
    history_writer = csv.writer(history_file)
    if not history_exists:
        history_writer.writerow([
            "step", "epoch", "train_loss", "train_diff_loss", "train_kld_loss", "train_align_loss", "val_mse", "val_profile_r", "lr"
        ])
        history_file.flush()

    # --- Training loop ---
    running_loss = 0.0
    running_diff = 0.0
    running_kld = 0.0
    running_align = 0.0
    epoch_loss = 0.0
    epoch_diff = 0.0
    epoch_kld = 0.0
    epoch_align = 0.0
    epoch_steps = 0
    # Per-step accumulators (reset after each optimizer step)
    step_loss = 0.0
    step_diff = 0.0
    step_kld = 0.0
    step_align = 0.0
    log_steps = 0
    accum_counter = 0
    start_time = time()

    logger.info(f"Starting training for {epochs} epochs, grad_accum={grad_accum}...")

    for epoch in range(start_epoch, epochs):
        wrapper.train()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", dynamic_ncols=True)
        for batch in pbar:
            x1 = batch["fmri"].to(device)           # (B, C, H, W) — target
            clip_tokens = batch["clip_tokens"].to(device)  # (B, T, D)
            clip_pool = batch["clip_pool"].to(device)      # (B, D_pool)

            with autocast(**autocast_kwargs):
                # 1) Source x₀ from PerceiverVE
                # x0_tok: (B, num_queries, out_channels) e.g. (B, 256, 64)
                x0_tok, mu, log_var = wrapper("source_encoder",
                                              text_tokens=clip_tokens)
                if log_var is not None:
                    log_var = torch.clamp(log_var, min=clip_logvar_min, max=clip_logvar_max)

                # Reshape to (B, C, H, W): permute → (B, D_out, Q), view → (B, C, H, W)
                B = x1.shape[0]
                x0 = x0_tok.permute(0, 2, 1).contiguous().view(B, *latent_size)

                # 2) Sample timestep & interpolate
                t = transport.sample_timestep(x1)
                t, xt, ut = transport.path_sampler.plan(t, x0, x1)

                # 3) Predict velocity
                v_pred = wrapper("dit", x=xt, t=t, y=clip_pool)

                # 4) Diffusion loss (masked)
                ut_target = ut.detach() if detach_ut else ut
                diff_loss = masked_mse(v_pred, ut_target, pad_mask,
                                       fmri_channels, fmri_spatial)

                loss = diff_loss

                # 5) KLD loss
                kld_loss_val = torch.tensor(0.0, device=device)
                if use_kld and log_var is not None:
                    kld_loss_val = kld_loss_factory(
                        mu, log_var, kld_type,
                        reduction=kld_reduction,
                        kld_target_std=kld_target_std,
                    )
                    loss = loss + kld_weight * kld_loss_val

                # 6) Alignment loss
                align_loss_val = torch.tensor(0.0, device=device)
                if use_align:
                    mu_reshaped = mu.permute(0, 2, 1).contiguous().view(B, *latent_size)
                    if align_type == "normalized_l2":
                        mu_norm = F.normalize(mu_reshaped.flatten(1), p=2, dim=-1)
                        x1_norm = F.normalize(x1.flatten(1), p=2, dim=-1)
                        align_loss_val = F.mse_loss(mu_norm, x1_norm)
                    elif align_type == "l2":
                        align_loss_val = F.mse_loss(mu_reshaped, x1)
                    loss = loss + align_weight * align_loss_val

            # Backward (scale by grad_accum)
            (loss / grad_accum).backward()
            accum_counter += 1

            # Update progress bar postfix
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{sched.get_last_lr()[0]:.2e}")

            # Accumulate loss stats per micro-batch
            _loss_val = loss.detach().item() / grad_accum
            _diff_val = diff_loss.detach().item() / grad_accum
            _kld_val = kld_loss_val.detach().item() / grad_accum
            _align_val = align_loss_val.detach().item() / grad_accum
            running_loss += _loss_val
            running_diff += _diff_val
            running_kld += _kld_val
            running_align += _align_val
            step_loss += _loss_val
            step_diff += _diff_val
            step_kld += _kld_val
            step_align += _align_val

            # Only step optimizer after grad_accum micro-batches
            if accum_counter < grad_accum:
                continue

            accum_counter = 0
            if clip_grad_val > 0:
                torch.nn.utils.clip_grad_norm_(wrapper.parameters(), clip_grad_val)
            opt.step()
            sched.step()
            update_ema(ema, wrapper, decay=ema_decay)
            opt.zero_grad(set_to_none=True)

            log_steps += 1
            train_steps += 1

            # Epoch-level accumulators (use per-step values, not running sums)
            epoch_loss += step_loss
            epoch_diff += step_diff
            epoch_kld += step_kld
            epoch_align += step_align
            epoch_steps += 1
            step_loss = step_diff = step_kld = step_align = 0.0

            # --- Logging ---
            if train_steps % log_every == 0:
                avg_loss = running_loss / log_steps
                avg_diff = running_diff / log_steps
                avg_kld = running_kld / log_steps
                avg_align = running_align / log_steps
                elapsed = time() - start_time
                steps_sec = log_steps / elapsed
                cur_lr = sched.get_last_lr()[0]

                logger.info(
                    f"[step={train_steps:07d} ep={epoch}] "
                    f"loss={avg_loss:.5f} diff={avg_diff:.5f} "
                    f"kld={avg_kld:.5f} align={avg_align:.5f} "
                    f"lr={cur_lr:.2e} "
                    f"steps/s={steps_sec:.1f}"
                )

                running_loss = running_diff = running_kld = running_align = 0.0
                log_steps = 0
                start_time = time()

            # --- Checkpoint ---
            if train_steps % ckpt_every == 0 and train_steps > 0:
                ckpt_path = os.path.join(ckpt_dir, f"{train_steps:07d}.pt")
                torch.save({
                    "model": wrapper.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "scheduler": sched.state_dict(),
                    "train_steps": train_steps,
                    "epoch": epoch,
                    "best_val_pcc": best_val_pcc,
                }, ckpt_path)
                logger.info(f"Saved checkpoint: {ckpt_path}")

            # --- Last checkpoint (rolling) ---
            if train_steps % 1000 == 0 and train_steps > 0:
                for old in glob(os.path.join(ckpt_dir, "last-*.pt")):
                    os.remove(old)
                last_path = os.path.join(ckpt_dir, f"last-{train_steps}.pt")
                torch.save({
                    "model": wrapper.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "scheduler": sched.state_dict(),
                    "train_steps": train_steps,
                    "epoch": epoch,
                    "best_val_pcc": best_val_pcc,
                }, last_path)

            # --- Sample & Evaluate ---
            if train_steps % sample_every == 0 and train_steps > 0:
                logger.info("Running evaluation on test subset...")
                ema.eval()
                with torch.no_grad():
                    # Take first 64 test samples
                    n_eval = min(64, len(test_ds))
                    eval_loader = DataLoader(test_ds, batch_size=n_eval,
                                             shuffle=False, num_workers=0)
                    eval_batch = next(iter(eval_loader))
                    eval_fmri = eval_batch["fmri"].to(device)
                    eval_clip_tok = eval_batch["clip_tokens"].to(device)
                    eval_clip_pool = eval_batch["clip_pool"].to(device)

                    # Generate source from EMA
                    x0_tok_eval, _, _ = ema.source_encoder(
                        text_tokens=eval_clip_tok
                    )
                    x0_eval = x0_tok_eval.permute(0, 2, 1).contiguous().view(
                        n_eval, *latent_size
                    )

                    # ODE sampling
                    with autocast(**autocast_kwargs):
                        traj = eval_sampler_fn(
                            x0_eval,
                            ema.dit.forward,
                            y=eval_clip_pool,
                        )
                    pred_fmri = traj[-1]  # (B, C, H, W)

                    # Profile Pearson r
                    corr = pearson_corr_per_sample(pred_fmri, eval_fmri, pad_mask)
                    mean_corr = corr.mean().item()
                    mse_val = masked_mse(pred_fmri, eval_fmri, pad_mask,
                                         fmri_channels, fmri_spatial).item()
                    logger.info(
                        f"  [Eval step={train_steps}] "
                        f"profile_r={mean_corr:.4f} mse={mse_val:.5f}"
                    )

            # --- Early stop for debugging ---
            if args.max_steps and train_steps >= args.max_steps:
                logger.info(f"Reached max_steps={args.max_steps}, stopping.")
                break

        # --- End-of-epoch validation ---
        is_val_epoch = (epoch + 1) % val_every == 0 or (epoch + 1) == epochs
        if not (args.max_steps and train_steps >= args.max_steps) and is_val_epoch:
            logger.info(f"End of epoch {epoch + 1}, running validation...")
            ema.eval()
            val_mse_sum = 0.0
            val_corr_sum = 0.0
            val_n = 0
            eval_batch_size = min(8, len(test_ds))
            eval_loader = DataLoader(test_ds, batch_size=eval_batch_size,
                                     shuffle=False, num_workers=0)
            with torch.no_grad():
                for eval_batch in tqdm(eval_loader, desc="Validating", leave=False, dynamic_ncols=True):
                    eval_fmri = eval_batch["fmri"].to(device)
                    eval_clip_tok = eval_batch["clip_tokens"].to(device)
                    eval_clip_pool = eval_batch["clip_pool"].to(device)
                    B_eval = eval_fmri.shape[0]

                    x0_tok_eval, _, _ = ema.source_encoder(
                        text_tokens=eval_clip_tok
                    )
                    x0_eval = x0_tok_eval.permute(0, 2, 1).contiguous().view(
                        B_eval, *latent_size
                    )

                    with autocast(**autocast_kwargs):
                        traj = eval_sampler_fn(
                            x0_eval,
                            ema.dit.forward,
                            y=eval_clip_pool,
                        )
                    pred_fmri = traj[-1]

                    corr = pearson_corr_per_sample(pred_fmri, eval_fmri, pad_mask)
                    mse_val = masked_mse(pred_fmri, eval_fmri, pad_mask,
                                         fmri_channels, fmri_spatial).item()
                    val_mse_sum += mse_val * B_eval
                    val_corr_sum += corr.sum().item()
                    val_n += B_eval

            val_mse_avg = val_mse_sum / val_n
            val_corr_avg = val_corr_sum / val_n
            logger.info(
                f"  [Val epoch={epoch + 1}] "
                f"profile_r={val_corr_avg:.4f} mse={val_mse_avg:.5f} "
                f"n_samples={val_n}"
            )

            # Average train losses since last validation
            train_loss_avg = epoch_loss / max(1, epoch_steps)
            train_diff_avg = epoch_diff / max(1, epoch_steps)
            train_kld_avg = epoch_kld / max(1, epoch_steps)
            train_align_avg = epoch_align / max(1, epoch_steps)
            cur_lr = sched.get_last_lr()[0]

            # Write to history.csv
            history_writer.writerow([
                train_steps, epoch,
                f"{train_loss_avg:.6f}", f"{train_diff_avg:.6f}",
                f"{train_kld_avg:.6f}", f"{train_align_avg:.6f}",
                f"{val_mse_avg:.6f}", f"{val_corr_avg:.6f}",
                f"{cur_lr:.2e}",
            ])
            history_file.flush()

            # Reset epoch-level accumulators
            epoch_loss = epoch_diff = epoch_kld = epoch_align = 0.0
            epoch_steps = 0

            # --- Save Best Checkpoint (PCC) ---
            if val_corr_avg > best_val_pcc:
                best_val_pcc = val_corr_avg
                best_path = os.path.join(ckpt_dir, "best.pt")
                torch.save({
                    "model": wrapper.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "scheduler": sched.state_dict(),
                    "train_steps": train_steps,
                    "epoch": epoch,
                    "best_val_pcc": best_val_pcc,
                    "val_mse": val_mse_avg,
                }, best_path)
                logger.info(f"New best validation PCC: {best_val_pcc:.4f}! Saved checkpoint to {best_path}")

        if args.max_steps and train_steps >= args.max_steps:
            break

    # --- Final checkpoint ---
    final_path = os.path.join(ckpt_dir, f"final-{train_steps}.pt")
    torch.save({
        "model": wrapper.state_dict(),
        "ema": ema.state_dict(),
        "opt": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "train_steps": train_steps,
        "epoch": epochs,
        "best_val_pcc": best_val_pcc,
    }, final_path)
    logger.info(f"Training complete. Final checkpoint: {final_path}")

    # Close CSV files
    history_file.close()
    logger.info(f"History saved to: {history_path}")


if __name__ == "__main__":
    main()
