"""
factflow_trainer.py
===================
Training orchestrator for FactFlow-based fMRI synthesis.

Pipeline:  CLIP image features → PerceiverVE (learned source x₀)
           → Flow Matching (velocity matching) → fMRI voxels

Adapted from the CSFM paper's training logic with:
  - No RAE / stage-1 encoder — operates directly on fMRI voxels
  - No CLIP text encoder — uses pre-extracted CLIP visual features
  - Masked loss on padded voxels
  - Single-GPU training
  - Pearson correlation evaluation
"""

from __future__ import annotations

import csv
import math
import os
from argparse import Namespace
from time import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.factflow_fmri_dataset import FactFlowfMRIDataset
from model.factflow_factory import build_models, build_transport, build_sampler
from utils.checkpoint import (
    load_checkpoint,
    find_last_checkpoint,
    save_checkpoint,
    save_rolling_last,
)
from utils.fmri_utils import create_pad_mask, get_latent_size
from utils.logging_utils import create_logger
from utils.metrics import masked_mse, pearson_corr_per_sample
from utils.training_utils import update_ema, build_optimizer_and_scheduler

# ── KLD losses (self-contained) ──────────────────────


def _kld_loss(
    mu: torch.Tensor,
    log_var: torch.Tensor,
    loss_type: str,
    reduction: str = "mean",
    target_std: float = 1.0,
) -> torch.Tensor:
    """KL-divergence regularisation on the variational source encoder.

    Supports:
      - ``"kld"``:     standard KLD: 𝔼[-½(1 + logσ² - μ² - σ²)]
      - ``"naive_kld"``: stability variant: replaces μ² with (0.3μ)⁶
      - ``"var_kld"``:  variance-only KLD (ignores μ term)
    """
    var = log_var.exp()
    if target_std != 1.0:
        sigma2_star = target_std ** 2
        var = var / sigma2_star
        log_var = log_var - math.log(sigma2_star)

    if loss_type == "kld":
        raw = -0.5 * (1 + log_var - mu ** 2 - var)
    elif loss_type == "naive_kld":
        raw = -0.5 * (1 + log_var - (0.3 * mu) ** 6 - var)
    elif loss_type == "var_kld":
        raw = -0.5 * (1 + log_var - var)
    else:
        raise ValueError(f"Unknown KLD type: {loss_type}")

    if reduction == "mean":
        return raw.mean()
    elif reduction == "sum":
        return raw.flatten(1).sum(1).mean()
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


# ═══════════════════════════════════════════════════════════════════════════


class FactFlowTrainer:
    """End-to-end trainer for FactFlow fMRI synthesis."""

    def __init__(self, args: Namespace) -> None:
        # ── Config ────────────────────────────────────────────────────
        self.cfg = OmegaConf.load(args.config)
        self.data_cfg = OmegaConf.to_container(self.cfg.data, resolve=True)
        self.train_cfg = OmegaConf.to_container(self.cfg.training, resolve=True)
        self.loss_cfg = OmegaConf.to_container(self.cfg.losses, resolve=True)
        self.args = args

        # ── Device & seed ─────────────────────────────────────────────
        self.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        seed = int(self.train_cfg.get("global_seed", 42))
        torch.manual_seed(seed)
        np.random.seed(seed)
        if self.device == "cuda":
            torch.cuda.manual_seed(seed)

        # ── Experiment dir ────────────────────────────────────────────
        exp_name = args.exp_name or f"factflow_fmri_sub{self.data_cfg['subject']}"
        self.exp_dir = os.path.join(args.exps_dir, exp_name)
        self.ckpt_dir = os.path.join(self.exp_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.logger = create_logger(self.exp_dir, name="factflow_trainer")

        # Save config snapshot
        cfg_path = os.path.join(self.exp_dir, "config.yaml")
        if not os.path.exists(cfg_path):
            OmegaConf.save(self.cfg, cfg_path)

        # ── Data ──────────────────────────────────────────────────────
        self._build_datasets()

        # ── Geometry ──────────────────────────────────────────────────
        self.latent_size = get_latent_size(self.data_cfg)
        self.pad_mask = create_pad_mask(
            self.data_cfg["n_voxels"], self.data_cfg["pad_to"], self.device,
        )

        # ── Models ────────────────────────────────────────────────────
        self.wrapper, self.ema = build_models(self.cfg, self.device)
        self.transport = build_transport(self.cfg, self.latent_size)
        self.sample_fn = build_sampler(self.transport, self.cfg.sampler)

        # ── Optimizer & scheduler ─────────────────────────────────────
        batch_size = int(self.train_cfg.get("batch_size", 64))
        self.grad_accum = int(self.train_cfg.get("grad_accum_steps", 1))
        self.steps_per_epoch = max(len(self.train_loader) // self.grad_accum, 1)
        self.epochs = int(self.train_cfg.get("epochs", 200))

        self.optimizer, self.scheduler, opt_msg, sched_msg = (
            build_optimizer_and_scheduler(
                self.wrapper.parameters(),
                self.train_cfg,
                self.steps_per_epoch,
                self.epochs,
            )
        )
        self.logger.info(opt_msg)
        self.logger.info(sched_msg)

        # ── State ─────────────────────────────────────────────────────
        self.train_steps = 0
        self.start_epoch = 0
        self.best_val_pcc = -1.0

        self._maybe_resume()

        # ── Training hyper-params ─────────────────────────────────────
        use_bf16 = self.train_cfg.get("precision", "fp32") == "bf16"
        self.autocast_kwargs = dict(
            device_type=self.device.split(":")[0],
            dtype=torch.bfloat16,
            enabled=use_bf16,
        )
        self.clip_grad = float(self.train_cfg.get("clip_grad", 1.0))
        self.ema_decay = float(self.train_cfg.get("ema_decay", 0.9999))
        self.log_every = int(self.train_cfg.get("log_every", 50))
        self.ckpt_every = int(self.train_cfg.get("ckpt_every", 5000))
        self.sample_every = int(self.train_cfg.get("sample_every", 2000))
        self.val_every = int(self.train_cfg.get("val_every", 1))

        # Loss config shortcuts
        self.use_kld = self.loss_cfg.get("use_kld_loss", True)
        self.kld_weight = float(self.loss_cfg.get("kld_loss_weight", 5.0))
        self.kld_type = self.loss_cfg.get("kld_loss_type", "var_kld")
        self.kld_reduction = self.loss_cfg.get("kld_reduction", "mean")
        self.kld_target_std = float(self.loss_cfg.get("kld_target_std", 1.0))
        self.use_align = self.loss_cfg.get("use_align_loss", True)
        self.align_weight = float(self.loss_cfg.get("align_loss_weight", 1.0))
        self.align_type = self.loss_cfg.get("align_loss_type", "normalized_l2")
        self.detach_ut = self.loss_cfg.get("detach_ut", False)
        self.clip_logvar_min = float(self.loss_cfg.get("clip_logvar_min", -10.0))
        self.clip_logvar_max = float(self.loss_cfg.get("clip_logvar_max", 10.0))

        # ── Initialise EMA ────────────────────────────────────────────
        update_ema(self.ema, self.wrapper, decay=0)  # exact copy
        self.wrapper.train()
        self.ema.eval()

    # ──────────────────────────────────────────────────────────────────
    # Dataset construction
    # ──────────────────────────────────────────────────────────────────

    def _build_datasets(self) -> None:
        dc = self.data_cfg
        ds_kwargs = dict(
            data_dir=dc["data_dir"],
            subject=dc["subject"],
            fmri_mode=dc["fmri_mode"],
            clip_feature=dc["clip_feature"],
            n_voxels=dc["n_voxels"],
            pad_to=dc["pad_to"],
            fmri_channels=dc["fmri_channels"],
            fmri_spatial=dc["fmri_spatial"],
        )
        self.train_ds = FactFlowfMRIDataset(mode="train", **ds_kwargs)
        self.test_ds = FactFlowfMRIDataset(mode="test", **ds_kwargs)

        batch_size = int(self.train_cfg.get("batch_size", 64))
        num_workers = int(self.train_cfg.get("num_workers", 4))
        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=num_workers > 0,
        )
        self.logger.info(
            "Train: %d samples, Test: %d samples", len(self.train_ds), len(self.test_ds),
        )
        self.logger.info(
            "Batch size: %d, Steps/epoch: %d", batch_size, len(self.train_loader),
        )

    # ──────────────────────────────────────────────────────────────────
    # Checkpoint resume
    # ──────────────────────────────────────────────────────────────────

    def _maybe_resume(self) -> None:
        ckpt_path: Optional[str] = None

        if self.args.ckpt:
            ckpt_path = self.args.ckpt
        elif self.args.resume_last:
            ckpt_path = find_last_checkpoint(self.ckpt_dir)
            if ckpt_path is None:
                self.logger.info("No last checkpoint found; starting fresh.")
                return

        if ckpt_path is not None:
            info = load_checkpoint(
                ckpt_path, self.wrapper, self.ema,
                self.optimizer, self.scheduler, self.device,
            )
            self.train_steps = info["train_steps"]
            self.start_epoch = info["epoch"]
            self.best_val_pcc = info["best_val_pcc"]

    # ──────────────────────────────────────────────────────────────────
    # Forward + loss
    # ──────────────────────────────────────────────────────────────────

    def _compute_loss(
        self,
        x1: torch.Tensor,
        clip_tokens: torch.Tensor,
        clip_pool: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """One forward pass: source encoding → interpolation → velocity prediction → losses.

        Returns a dict with keys: ``total``, ``diff``, ``kld``, ``align``.
        """
        B = x1.shape[0]

        # 1) Source x₀ from PerceiverVE
        x0_tok, mu, log_var = self.wrapper.encode_source(clip_tokens)
        if log_var is not None:
            log_var = torch.clamp(
                log_var, min=self.clip_logvar_min, max=self.clip_logvar_max,
            )

        # Reshape: (B, Q, D) → (B, C, H, W)
        x0 = x0_tok.permute(0, 2, 1).contiguous().view(B, *self.latent_size)

        # 2) Sample timestep & interpolate along the flow path
        t = self.transport.sample_timestep(x1)
        t, xt, ut = self.transport.path_sampler.plan(t, x0, x1)

        # 3) Predict velocity
        v_pred = self.wrapper.predict_velocity(x=xt, t=t, y=clip_pool)

        # 4) Diffusion loss (masked MSE on real voxels)
        ut_target = ut.detach() if self.detach_ut else ut
        diff_loss = masked_mse(v_pred, ut_target, self.pad_mask)
        total = diff_loss

        # 5) KLD regularisation on the variational source encoder
        kld_val = torch.tensor(0.0, device=self.device)
        if self.use_kld and log_var is not None:
            kld_val = _kld_loss(
                mu, log_var, self.kld_type,
                reduction=self.kld_reduction,
                target_std=self.kld_target_std,
            )
            total = total + self.kld_weight * kld_val

        # 6) Alignment loss (source mean ↔ target)
        align_val = torch.tensor(0.0, device=self.device)
        if self.use_align:
            mu_reshaped = mu.permute(0, 2, 1).contiguous().view(B, *self.latent_size)
            if self.align_type == "normalized_l2":
                mu_norm = F.normalize(mu_reshaped.flatten(1), p=2, dim=-1)
                x1_norm = F.normalize(x1.flatten(1), p=2, dim=-1)
                align_val = F.mse_loss(mu_norm, x1_norm)
            elif self.align_type == "l2":
                align_val = F.mse_loss(mu_reshaped, x1)
            total = total + self.align_weight * align_val

        return {
            "total": total,
            "diff": diff_loss.detach(),
            "kld": kld_val.detach(),
            "align": align_val.detach(),
        }

    # ──────────────────────────────────────────────────────────────────
    # Inline evaluation (quick subset during training)
    # ──────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _inline_eval(self) -> None:
        """Quick ODE-based eval on a small test subset."""
        self.ema.eval()
        n_eval = min(64, len(self.test_ds))
        loader = DataLoader(self.test_ds, batch_size=n_eval, shuffle=False, num_workers=0)
        batch = next(iter(loader))

        fmri_gt = batch["fmri"].to(self.device)
        clip_tok = batch["clip_tokens"].to(self.device)
        clip_pool = batch["clip_pool"].to(self.device)

        # Source from EMA encoder
        x0_tok, _, _ = self.ema.encode_source(clip_tok)
        x0 = x0_tok.permute(0, 2, 1).contiguous().view(n_eval, *self.latent_size)

        # ODE sampling
        with autocast(**self.autocast_kwargs):
            traj = self.sample_fn(x0, self.ema.dit.forward, y=clip_pool)
        pred = traj[-1]

        corr = pearson_corr_per_sample(pred, fmri_gt, self.pad_mask)
        mouse_val = masked_mse(pred, fmri_gt, self.pad_mask).item()
        self.logger.info(
            "  [Eval step=%d] profile_r=%.4f  mse=%.5f",
            self.train_steps, corr.mean().item(), mouse_val,
        )

    # ──────────────────────────────────────────────────────────────────
    # Full validation (end of epoch)
    # ──────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict[str, float]:
        """Full validation over the entire test set."""
        self.ema.eval()
        eval_bs = min(8, len(self.test_ds))
        loader = DataLoader(self.test_ds, batch_size=eval_bs, shuffle=False, num_workers=0)

        mse_sum, corr_sum, n = 0.0, 0.0, 0

        for batch in tqdm(loader, desc="Validating", leave=False, dynamic_ncols=True):
            fmri_gt = batch["fmri"].to(self.device)
            clip_tok = batch["clip_tokens"].to(self.device)
            clip_pool = batch["clip_pool"].to(self.device)
            B = fmri_gt.shape[0]

            x0_tok, _, _ = self.ema.encode_source(clip_tok)
            x0 = x0_tok.permute(0, 2, 1).contiguous().view(B, *self.latent_size)

            with autocast(**self.autocast_kwargs):
                traj = self.sample_fn(x0, self.ema.dit.forward, y=clip_pool)
            pred = traj[-1]

            corr = pearson_corr_per_sample(pred, fmri_gt, self.pad_mask)
            mse_val = masked_mse(pred, fmri_gt, self.pad_mask).item()

            mse_sum += mse_val * B
            corr_sum += corr.sum().item()
            n += B

        val_mse = mse_sum / n
        val_corr = corr_sum / n
        self.logger.info(
            "  [Val epoch=%d] profile_r=%.4f  mse=%.5f  n=%d",
            epoch + 1, val_corr, val_mse, n,
        )
        return {"mse": val_mse, "profile_r": val_corr, "n": n}

    # ──────────────────────────────────────────────────────────────────
    # Main training loop
    # ──────────────────────────────────────────────────────────────────

    def train(self) -> None:
        """Run the full training loop."""
        # ── CSV history ──
        history_path = os.path.join(self.exp_dir, "history.csv")
        history_exists = os.path.exists(history_path)
        history_file = open(history_path, "a", newline="")
        history_writer = csv.writer(history_file)
        if not history_exists:
            history_writer.writerow([
                "step", "epoch",
                "train_loss", "train_diff_loss", "train_kld_loss", "train_align_loss",
                "val_mse", "val_profile_r", "lr",
            ])
            history_file.flush()

        # ── Accumulators ──
        running = {"loss": 0., "diff": 0., "kld": 0., "align": 0.}
        epoch_acc = {"loss": 0., "diff": 0., "kld": 0., "align": 0.}
        step_acc = {"loss": 0., "diff": 0., "kld": 0., "align": 0.}
        epoch_steps = 0
        log_steps = 0
        accum_counter = 0
        wall_start = time()

        self.logger.info(
            "Starting training for %d epochs, grad_accum=%d ...",
            self.epochs, self.grad_accum,
        )

        for epoch in range(self.start_epoch, self.epochs):
            self.wrapper.train()
            pbar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch + 1}/{self.epochs}",
                dynamic_ncols=True,
            )

            for batch in pbar:
                x1 = batch["fmri"].to(self.device)
                clip_tokens = batch["clip_tokens"].to(self.device)
                clip_pool = batch["clip_pool"].to(self.device)

                with autocast(**self.autocast_kwargs):
                    losses = self._compute_loss(x1, clip_tokens, clip_pool)

                # Backward (scale by grad_accum)
                (losses["total"] / self.grad_accum).backward()
                accum_counter += 1

                pbar.set_postfix(
                    loss=f"{losses['total'].item():.4f}",
                    lr=f"{self.scheduler.get_last_lr()[0]:.2e}",
                )

                # Accumulate micro-batch stats
                scale = 1.0 / self.grad_accum
                for k in running:
                    val = losses.get(k, losses.get("total")).item() * scale
                    if k == "loss":
                        val = losses["total"].item() * scale
                    else:
                        val = losses[k].item() * scale
                    running[k] += val
                    step_acc[k] += val

                if accum_counter < self.grad_accum:
                    continue

                # ── Optimizer step ──
                accum_counter = 0
                if self.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.wrapper.parameters(), self.clip_grad,
                    )
                self.optimizer.step()
                self.scheduler.step()
                update_ema(self.ema, self.wrapper, decay=self.ema_decay)
                self.optimizer.zero_grad(set_to_none=True)

                log_steps += 1
                self.train_steps += 1

                for k in epoch_acc:
                    epoch_acc[k] += step_acc[k]
                epoch_steps += 1
                step_acc = {k: 0. for k in step_acc}

                # ── Periodic logging ──
                if self.train_steps % self.log_every == 0:
                    elapsed = time() - wall_start
                    sps = log_steps / elapsed if elapsed > 0 else 0
                    lr = self.scheduler.get_last_lr()[0]
                    self.logger.info(
                        "[step=%07d ep=%d] loss=%.5f diff=%.5f kld=%.5f "
                        "align=%.5f lr=%.2e steps/s=%.1f",
                        self.train_steps, epoch,
                        running["loss"] / log_steps,
                        running["diff"] / log_steps,
                        running["kld"] / log_steps,
                        running["align"] / log_steps,
                        lr, sps,
                    )
                    running = {k: 0. for k in running}
                    log_steps = 0
                    wall_start = time()

                # ── Periodic checkpoint ──
                if self.train_steps % self.ckpt_every == 0 and self.train_steps > 0:
                    save_checkpoint(
                        os.path.join(self.ckpt_dir, f"{self.train_steps:07d}.pt"),
                        self.wrapper, self.ema, self.optimizer, self.scheduler,
                        self.train_steps, epoch, self.best_val_pcc,
                    )

                # ── Rolling last checkpoint ──
                if self.train_steps % 1000 == 0 and self.train_steps > 0:
                    save_rolling_last(
                        self.ckpt_dir, self.wrapper, self.ema,
                        self.optimizer, self.scheduler,
                        self.train_steps, epoch, self.best_val_pcc,
                    )

                # ── Inline sample eval ──
                if self.train_steps % self.sample_every == 0 and self.train_steps > 0:
                    self.logger.info("Running inline evaluation...")
                    self._inline_eval()

                # ── Early stop (debug) ──
                if self.args.max_steps and self.train_steps >= self.args.max_steps:
                    self.logger.info("Reached max_steps=%d, stopping.", self.args.max_steps)
                    break

            # ── End-of-epoch validation ──
            is_val_epoch = (epoch + 1) % self.val_every == 0 or (epoch + 1) == self.epochs
            hit_max = self.args.max_steps and self.train_steps >= self.args.max_steps

            if not hit_max and is_val_epoch:
                self.logger.info("End of epoch %d, running validation...", epoch + 1)
                val = self._validate(epoch)

                # Write CSV
                lr = self.scheduler.get_last_lr()[0]
                ep_n = max(1, epoch_steps)
                history_writer.writerow([
                    self.train_steps, epoch,
                    f"{epoch_acc['loss'] / ep_n:.6f}",
                    f"{epoch_acc['diff'] / ep_n:.6f}",
                    f"{epoch_acc['kld'] / ep_n:.6f}",
                    f"{epoch_acc['align'] / ep_n:.6f}",
                    f"{val['mse']:.6f}",
                    f"{val['profile_r']:.6f}",
                    f"{lr:.2e}",
                ])
                history_file.flush()

                # Reset epoch accumulators
                epoch_acc = {k: 0. for k in epoch_acc}
                epoch_steps = 0

                # Save best checkpoint
                if val["profile_r"] > self.best_val_pcc:
                    self.best_val_pcc = val["profile_r"]
                    save_checkpoint(
                        os.path.join(self.ckpt_dir, "best.pt"),
                        self.wrapper, self.ema, self.optimizer, self.scheduler,
                        self.train_steps, epoch, self.best_val_pcc,
                        val_mse=val["mse"],
                    )
                    self.logger.info(
                        "New best PCC: %.4f!", self.best_val_pcc,
                    )

            if hit_max:
                break

        # ── Final checkpoint ──
        save_checkpoint(
            os.path.join(self.ckpt_dir, f"final-{self.train_steps}.pt"),
            self.wrapper, self.ema, self.optimizer, self.scheduler,
            self.train_steps, self.epochs, self.best_val_pcc,
        )
        history_file.close()
        self.logger.info("Training complete. History: %s", history_path)
