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
from utils.metrics import masked_mse, voxel_pearson, compute_voxel_reliability
from utils.training_utils import build_optimizer_and_scheduler

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
        self.wrapper = build_models(self.cfg, self.device)
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
        self.best_voxel_r = -1.0

        self._maybe_resume()

        # ── Training hyper-params ─────────────────────────────────────
        use_bf16 = self.train_cfg.get("precision", "fp32") == "bf16"
        self.autocast_kwargs = dict(
            device_type=self.device.split(":")[0],
            dtype=torch.bfloat16,
            enabled=use_bf16,
        )
        self.clip_grad = float(self.train_cfg.get("clip_grad", 1.0))
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
        # Correlation-aware reconstruction loss in x̂₁ space (metric-aligned).
        self.use_recon = bool(self.loss_cfg.get("use_recon_loss", False))
        self.recon_weight = float(self.loss_cfg.get("recon_loss_weight", 0.0))
        self.recon_type = self.loss_cfg.get("recon_loss_type", "pearson")
        self.detach_ut = self.loss_cfg.get("detach_ut", False)
        self.clip_logvar_min = float(self.loss_cfg.get("clip_logvar_min", -10.0))
        self.clip_logvar_max = float(self.loss_cfg.get("clip_logvar_max", 10.0))
        self.use_source = bool(self.cfg.get("use_source_encoder", True))
        # Use the source-encoder mean (μ) instead of a sampled x₀ at eval time.
        self.eval_use_mean = bool(self.cfg.get("eval_use_mean", False))

        # ── Cross-attention conditioning (CLIP / DINOv2 tokens → DiT) ──
        self.use_cross_attn = bool(
            self.cfg.stage_2.get("params", {}).get("use_cross_attn", False)
        )
        self.use_dino = self.use_cross_attn and self.data_cfg.get("dino_feature") is not None

        # ── SNR-weighted voxel loss ───────────────────────────────────
        self.voxel_weight = self._build_voxel_weight()

        self.wrapper.train()

    def _build_voxel_weight(self) -> Optional[torch.Tensor]:
        """Per-voxel noise-ceiling weight over ``pad_to`` (None if disabled)."""
        if not self.loss_cfg.get("use_snr_weight", False):
            return None
        nc = compute_voxel_reliability(
            self.train_ds.fmri_data, self.train_ds.n_reps,
        )  # (V,)
        w = np.zeros(self.data_cfg["pad_to"], dtype=np.float64)
        w[: self.data_cfg["n_voxels"]] = nc
        # Normalise so the mean weight over real voxels is 1 (keeps loss scale).
        real = w[: self.data_cfg["n_voxels"]]
        mean_w = real.mean()
        if mean_w > 0:
            w = w / mean_w
        self.logger.info(
            "SNR-weighted loss ON: noise-ceiling mean=%.3f median=%.3f, "
            "voxels with nc>0.1: %d/%d",
            float(nc.mean()), float(np.median(nc)),
            int((nc > 0.1).sum()), nc.shape[0],
        )
        return torch.tensor(w, dtype=torch.float32, device=self.device)

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
            fmri_channels=dc.get("fmri_channels", 1),
            fmri_spatial=dc.get("fmri_spatial", None),
            avg_reps=dc.get("avg_reps", False),
            dino_feature=dc.get("dino_feature", None),
        )
        self.train_ds = FactFlowfMRIDataset(mode="train", **ds_kwargs)
        self.test_ds  = FactFlowfMRIDataset(mode="test",  **ds_kwargs)

        # ── Rep-averaged validation dataset ───────────────────────────
        # Always validate against rep-averaged GT (mean of 3 reps) so that:
        #   - ceiling R ≈ 0.620  (vs single-trial ceiling R ≈ 0.148)
        #   - metric is meaningful and comparable to published benchmarks
        # This is independent of whether training uses avg_reps or not.
        val_kwargs = {**ds_kwargs, "avg_reps": True}
        self.val_ds = FactFlowfMRIDataset(mode="test", **val_kwargs)

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
            "Train: %d samples, Test: %d samples (val_avg: %d images)",
            len(self.train_ds), len(self.test_ds), len(self.val_ds),
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
                ckpt_path, self.wrapper,
                self.optimizer, self.scheduler, self.device,
            )
            self.train_steps = info["train_steps"]
            self.start_epoch = info["epoch"]
            self.best_voxel_r = info["best_val_pcc"]

    # ──────────────────────────────────────────────────────────────────
    # Forward + loss
    # ──────────────────────────────────────────────────────────────────

    def _reparam_source(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Sample source tokens x₀ = μ + ε·σ with σ from the *clamped* log-variance.

        The encoder's ``forward`` samples internally using the raw, unclamped
        log-variance, which can diverge early in training while the KLD term only
        ever sees the clamped value. Re-sampling here from the clamped
        log-variance keeps the noise injected into x₀ consistent with the
        regulariser. Falls back to μ for a non-variational encoder.
        """
        if log_var is None:
            return mu
        log_var = torch.clamp(
            log_var, min=self.clip_logvar_min, max=self.clip_logvar_max,
        )
        std = (0.5 * log_var).exp()
        return mu + torch.randn_like(mu) * std

    def _compute_loss(
        self,
        x1: torch.Tensor,
        clip_tokens: torch.Tensor,
        clip_pool: torch.Tensor,
        dino_tokens: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """One forward pass: source encoding → interpolation → velocity prediction → losses.

        Returns a dict with keys: ``total``, ``diff``, ``kld``, ``align``.
        """
        B = x1.shape[0]

        # 1) Source x₀
        if self.use_source:
            _, mu, log_var = self.wrapper.encode_source(clip_tokens)
            if log_var is not None:
                log_var = torch.clamp(
                    log_var, min=self.clip_logvar_min, max=self.clip_logvar_max,
                )
            # Sample x₀ = μ + ε·σ from the clamped log-variance (consistent with
            # the KLD term), then reshape (B, Q, D) → (B, C, H, W).
            x0_tok = self._reparam_source(mu, log_var)
            x0 = x0_tok.permute(0, 2, 1).contiguous().view(B, *self.latent_size)
        else:
            # Pure Gaussian noise source (standard flow matching)
            x0 = torch.randn_like(x1)
            mu, log_var = None, None

        # 2) Sample timestep & interpolate along the flow path
        t = self.transport.sample_timestep(x1)
        t, xt, ut = self.transport.path_sampler.plan(t, x0, x1)

        # 3) Predict velocity (cross-attend to CLIP/DINOv2 tokens if enabled)
        context = clip_tokens if self.use_cross_attn else None
        context2 = dino_tokens if self.use_dino else None
        v_pred = self.wrapper.predict_velocity(
            x=xt, t=t, y=clip_pool, context=context, context2=context2,
        )

        # 4) Diffusion loss (masked, optionally SNR-weighted, MSE on real voxels)
        ut_target = ut.detach() if self.detach_ut else ut
        diff_loss = masked_mse(v_pred, ut_target, self.pad_mask, weight=self.voxel_weight)
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
        if self.use_align and mu is not None:
            mu_reshaped = mu.permute(0, 2, 1).contiguous().view(B, *self.latent_size)
            if self.align_type == "normalized_l2":
                mu_norm = F.normalize(mu_reshaped.flatten(1), p=2, dim=-1)
                x1_norm = F.normalize(x1.flatten(1), p=2, dim=-1)
                align_val = F.mse_loss(mu_norm, x1_norm)
            elif self.align_type == "l2":
                align_val = F.mse_loss(mu_reshaped, x1)
            total = total + self.align_weight * align_val

        # 7) Reconstruction loss in x̂₁ space (correlation-aware, metric-aligned).
        # From the linear path xₜ=(1-t)x₀+t·x₁ and uₜ=x₁-x₀, the target is
        # recovered exactly as x₁ = xₜ + (1-t)·uₜ; substituting the predicted
        # velocity gives a differentiable estimate x̂₁ to score against x₁.
        recon_val = torch.tensor(0.0, device=self.device)
        if self.use_recon:
            t_b = t.view(B, *([1] * (x1.dim() - 1)))
            x1_hat = xt + (1.0 - t_b) * v_pred
            recon_val = self._recon_loss(x1_hat, x1)
            total = total + self.recon_weight * recon_val

        return {
            "total": total,
            "diff": diff_loss.detach(),
            "kld": kld_val.detach(),
            "align": align_val.detach(),
            "recon": recon_val.detach(),
        }

    def _recon_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Loss on the reconstructed target x̂₁, restricted to real voxels.

        ``pearson`` — 1 − profile Pearson r per sample (over voxels), averaged.
        Scale/shift-invariant, so it must *not* be the velocity loss, but as an
        auxiliary on x̂₁ it directly rewards the ``voxel_pearson`` eval metric.
        ``mse`` — plain masked MSE on x̂₁ (magnitude-aware).
        """
        B = pred.shape[0]
        mask = self.pad_mask.to(pred.device)
        p = pred.reshape(B, -1)[:, mask].float()
        g = target.reshape(B, -1)[:, mask].float()
        if self.recon_type == "pearson":
            p = p - p.mean(dim=1, keepdim=True)
            g = g - g.mean(dim=1, keepdim=True)
            num = (p * g).sum(dim=1)
            den = p.norm(dim=1) * g.norm(dim=1) + 1e-8
            return (1.0 - num / den).mean()
        elif self.recon_type == "mse":
            return F.mse_loss(p, g)
        raise ValueError(f"Unknown recon_loss_type: {self.recon_type}")

    # ──────────────────────────────────────────────────────────────────
    # Inline evaluation (quick subset during training)
    # ──────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _inline_eval(self) -> None:
        """Quick ODE-based eval on a small test subset."""
        self.wrapper.eval()
        n_eval = min(64, len(self.test_ds))
        loader = DataLoader(self.test_ds, batch_size=n_eval, shuffle=False, num_workers=0)
        batch = next(iter(loader))

        fmri_gt = batch["fmri"].to(self.device)
        clip_tok = batch["clip_tokens"].to(self.device)
        clip_pool = batch["clip_pool"].to(self.device)
        dino_tok = batch["dino_tokens"].to(self.device) if self.use_dino else None

        # Source from raw model encoder
        src_std = None
        src_corr = None
        if self.use_source:
            _, mu, log_var = self.wrapper.encode_source(clip_tok)
            src = mu if self.eval_use_mean else self._reparam_source(mu, log_var)
            x0 = src.permute(0, 2, 1).contiguous().view(n_eval, *self.latent_size)
            # Diagnostics: is the source a genuine distribution (σ>0) that does
            # NOT merely copy the target (low corr(μ, x₁))?
            if log_var is not None:
                src_std = log_var.float().mul(0.5).exp().mean().item()
            mu_flat = mu.permute(0, 2, 1).contiguous().view(n_eval, -1)[:, self.pad_mask]
            src_corr = voxel_pearson(
                mu_flat.float(),
                fmri_gt.reshape(n_eval, -1)[:, self.pad_mask].float(),
            ).mean().item()
        else:
            x0 = torch.randn(n_eval, *self.latent_size, device=self.device)

        # ODE sampling
        context = clip_tok if self.use_cross_attn else None
        context2 = dino_tok if self.use_dino else None
        with autocast(**self.autocast_kwargs):
            traj = self.sample_fn(
                x0, self.wrapper.dit.forward, y=clip_pool,
                context=context, context2=context2,
            )
        pred = traj[-1].float()

        preds_flat = pred.reshape(n_eval, -1)[:, self.pad_mask]
        gts_flat = fmri_gt.reshape(n_eval, -1)[:, self.pad_mask]
        voxel_r = voxel_pearson(preds_flat, gts_flat).mean().item()
        mse_val = masked_mse(pred, fmri_gt, self.pad_mask).item()
        src_msg = ""
        if src_std is not None or src_corr is not None:
            src_msg = "  [src σ=%s corr(μ,x₁)=%s]" % (
                f"{src_std:.3f}" if src_std is not None else "n/a",
                f"{src_corr:.3f}" if src_corr is not None else "n/a",
            )
        self.logger.info(
            "  [Eval step=%d] voxel_r=%.4f  mse=%.5f%s",
            self.train_steps, voxel_r, mse_val, src_msg,
        )
        self.wrapper.train()

    # ──────────────────────────────────────────────────────────────────
    # Full validation (end of epoch)
    # ──────────────────────────────────────────────────────────────────

    def _run_val_pass(
        self,
        model: torch.nn.Module,
        loader,
        label: str,
        epoch: int,
    ) -> Dict[str, float]:
        """Single validation pass over *loader* using *model*.

        Accumulates all predictions/targets, then computes voxel-wise
        (encoding) Pearson r across the sample axis — the honest NSD metric.
        """
        model.eval()
        preds_all, gts_all = [], []
        mse_sum, n = 0.0, 0

        for batch in loader:
            fmri_gt = batch["fmri"].to(self.device)
            clip_tok = batch["clip_tokens"].to(self.device)
            clip_pool = batch["clip_pool"].to(self.device)
            dino_tok = batch["dino_tokens"].to(self.device) if self.use_dino else None
            B = fmri_gt.shape[0]

            if self.use_source:
                _, mu, log_var = model.encode_source(clip_tok)
                src = mu if self.eval_use_mean else self._reparam_source(mu, log_var)
                x0 = src.permute(0, 2, 1).contiguous().view(B, *self.latent_size)
            else:
                x0 = torch.randn(B, *self.latent_size, device=self.device)

            context = clip_tok if self.use_cross_attn else None
            context2 = dino_tok if self.use_dino else None
            with autocast(**self.autocast_kwargs):
                traj = self.sample_fn(
                    x0, model.dit.forward, y=clip_pool,
                    context=context, context2=context2,
                )
            pred = traj[-1].float()

            mse_sum += masked_mse(pred, fmri_gt, self.pad_mask).item() * B
            n += B
            preds_all.append(pred.reshape(B, -1)[:, self.pad_mask].cpu())
            gts_all.append(fmri_gt.reshape(B, -1)[:, self.pad_mask].cpu())

        preds_all = torch.cat(preds_all, dim=0)
        gts_all = torch.cat(gts_all, dim=0)
        val_voxel_r = voxel_pearson(preds_all, gts_all).mean().item()
        val_mse = mse_sum / n
        self.logger.info(
            "  [Val epoch=%d %s] voxel_r=%.4f  mse=%.5f  n=%d",
            epoch + 1, label, val_voxel_r, val_mse, n,
        )
        return {"mse": val_mse, "voxel_r": val_voxel_r, "n": n}

    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict[str, float]:
        """Full validation over the rep-averaged test set (val_ds, avg_reps=True).

        Reports voxel-wise (encoding) Pearson r: per-voxel correlation across
        images. Rep-averaging the GT removes measurement noise so the metric
        reflects true signal rather than the single-trial noise floor.
        """
        val_bs = 32
        loader = DataLoader(
            self.val_ds,
            batch_size=val_bs,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )

        self.wrapper.eval()
        metrics = self._run_val_pass(self.wrapper, loader, "avg_reps", epoch)
        self.wrapper.train()
        return metrics

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
                "epoch", "step", "train_loss", "train_diff", "train_kld", "train_align", "train_recon", "val_mse", "val_voxel_r", "lr",
            ])
            history_file.flush()

        # ── Accumulators ──
        running_loss = 0.0   # since last log_every
        running_diff = 0.0
        running_kld = 0.0
        running_align = 0.0
        running_recon = 0.0
        epoch_loss = 0.0     # since last validation
        epoch_diff = 0.0
        epoch_kld = 0.0
        epoch_align = 0.0
        epoch_recon = 0.0
        step_loss = 0.0      # current optimizer step (across micro-batches)
        step_diff = 0.0
        step_kld = 0.0
        step_align = 0.0
        step_recon = 0.0
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
                dino_tokens = (
                    batch["dino_tokens"].to(self.device) if self.use_dino else None
                )

                with autocast(**self.autocast_kwargs):
                    losses = self._compute_loss(
                        x1, clip_tokens, clip_pool, dino_tokens,
                    )

                # Backward (scale by grad_accum)
                (losses["total"] / self.grad_accum).backward()
                accum_counter += 1

                pbar.set_postfix(
                    loss=f"{losses['total'].item():.4f}",
                    lr=f"{self.scheduler.get_last_lr()[0]:.2e}",
                )

                # Accumulate micro-batch loss (rescaled to full-step units)
                micro_loss = losses["total"].item() / self.grad_accum
                micro_diff = losses["diff"].item() / self.grad_accum
                micro_kld = losses["kld"].item() / self.grad_accum
                micro_align = losses["align"].item() / self.grad_accum
                micro_recon = losses["recon"].item() / self.grad_accum

                running_loss += micro_loss
                running_diff += micro_diff
                running_kld += micro_kld
                running_align += micro_align
                running_recon += micro_recon

                step_loss += micro_loss
                step_diff += micro_diff
                step_kld += micro_kld
                step_align += micro_align
                step_recon += micro_recon

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
                self.optimizer.zero_grad(set_to_none=True)

                log_steps += 1
                self.train_steps += 1

                epoch_loss += step_loss
                epoch_diff += step_diff
                epoch_kld += step_kld
                epoch_align += step_align
                epoch_recon += step_recon
                epoch_steps += 1

                step_loss = 0.0
                step_diff = 0.0
                step_kld = 0.0
                step_align = 0.0
                step_recon = 0.0

                # ── Periodic logging ──
                if self.train_steps % self.log_every == 0:
                    elapsed = time() - wall_start
                    sps = log_steps / elapsed if elapsed > 0 else 0
                    lr = self.scheduler.get_last_lr()[0]
                    self.logger.info(
                        "[step=%07d ep=%d] loss=%.5f (diff=%.5f kld=%.5f align=%.5f recon=%.5f) lr=%.2e steps/s=%.1f",
                        self.train_steps, epoch,
                        running_loss / log_steps,
                        running_diff / log_steps, running_kld / log_steps,
                        running_align / log_steps, running_recon / log_steps,
                        lr, sps,
                    )
                    running_loss = 0.0
                    running_diff = 0.0
                    running_kld = 0.0
                    running_align = 0.0
                    running_recon = 0.0
                    log_steps = 0
                    wall_start = time()

                # ── Periodic checkpoint ──
                if self.train_steps % self.ckpt_every == 0 and self.train_steps > 0:
                    save_checkpoint(
                        os.path.join(self.ckpt_dir, f"{self.train_steps:07d}.pt"),
                        self.wrapper, self.optimizer, self.scheduler,
                        self.train_steps, epoch, self.best_voxel_r,
                    )

                # ── Rolling last checkpoint ──
                if self.train_steps % 1000 == 0 and self.train_steps > 0:
                    save_rolling_last(
                        self.ckpt_dir, self.wrapper,
                        self.optimizer, self.scheduler,
                        self.train_steps, epoch, self.best_voxel_r,
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

                lr = self.scheduler.get_last_lr()[0]
                ep_n = max(1, epoch_steps)
                history_writer.writerow([
                    epoch + 1, self.train_steps,
                    f"{epoch_loss / ep_n:.6f}",
                    f"{epoch_diff / ep_n:.6f}",
                    f"{epoch_kld / ep_n:.6f}",
                    f"{epoch_align / ep_n:.6f}",
                    f"{epoch_recon / ep_n:.6f}",
                    f"{val['mse']:.6f}",
                    f"{val['voxel_r']:.6f}",
                    f"{lr:.2e}",
                ])
                history_file.flush()

                # Reset epoch accumulators
                epoch_loss = 0.0
                epoch_diff = 0.0
                epoch_kld = 0.0
                epoch_align = 0.0
                epoch_recon = 0.0
                epoch_steps = 0

                # Save best checkpoint by voxel-wise Pearson r
                if val["voxel_r"] > self.best_voxel_r:
                    self.best_voxel_r = val["voxel_r"]
                    save_checkpoint(
                        os.path.join(self.ckpt_dir, "best.pt"),
                        self.wrapper, self.optimizer, self.scheduler,
                        self.train_steps, epoch, self.best_voxel_r,
                        val_mse=val["mse"],
                    )
                    self.logger.info(
                        "New best voxel_r: %.4f", self.best_voxel_r,
                    )

            if hit_max:
                break

        # ── Final checkpoint ──
        save_checkpoint(
            os.path.join(self.ckpt_dir, f"final-{self.train_steps}.pt"),
            self.wrapper, self.optimizer, self.scheduler,
            self.train_steps, self.epochs, self.best_voxel_r,
        )
        history_file.close()
        self.logger.info("Training complete. History: %s", history_path)
