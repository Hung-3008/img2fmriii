"""
factflow_evaluator.py
=====================
Full evaluation for FactFlow-based fMRI synthesis.

Supports three inference scenarios:

  1. **deterministic** — PerceiverVE mean μ → ODE solver (fully deterministic).
  2. **perceiver_stochastic** — PerceiverVE samples x₀ = μ + ε·σ → ODE solver.
     With K > 1, generates K predictions per image and averages them.
  3. **flow_stochastic** — PerceiverVE mean μ → SDE solver (noise-injected flow).
     With K > 1, generates K predictions per image and averages them.

Metrics:
  - Per-voxel Pearson r (encoding metric, mean & median)
  - Profile Pearson r (mean)
  - MSE
  - Image-level metrics (rep-averaged)

Results can be saved to ``.npz`` and/or appended to a CSV file.
"""

from __future__ import annotations

import csv
import logging
import math
import os
from argparse import Namespace
from typing import Dict, Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.factflow_fmri_dataset import FactFlowfMRIDataset
from model.factflow_factory import build_models, build_transport, build_sampler
from model.transport import Sampler
from utils.checkpoint import load_checkpoint
from utils.fmri_utils import get_latent_size
from utils.logging_utils import create_logger
from utils.metrics import compute_full_metrics

logger = logging.getLogger(__name__)


class FactFlowEvaluator:
    """Evaluate a trained FactFlow fMRI model on the test set."""

    def __init__(self, args: Namespace) -> None:
        # ── Config ────────────────────────────────────────────────────
        self.cfg = OmegaConf.load(args.config)
        self.data_cfg = OmegaConf.to_container(self.cfg.data, resolve=True)
        self.loss_cfg = OmegaConf.to_container(
            self.cfg.get("losses", {}), resolve=True,
        )
        self.args = args

        # ── Scenario ──────────────────────────────────────────────────
        self.scenario = getattr(args, "scenario", "deterministic")
        self.k_trials = getattr(args, "k_trials", 1)
        if self.scenario == "deterministic":
            self.k_trials = 1  # deterministic → always 1 trial

        # ── Device ────────────────────────────────────────────────────
        self.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        use_bf16 = self.cfg.training.get("precision", "fp32") == "bf16"
        self.autocast_kwargs = dict(
            device_type=self.device.split(":")[0],
            dtype=torch.bfloat16,
            enabled=use_bf16,
        )

        self.logger = create_logger(name="factflow_eval")

        # ── Data ──────────────────────────────────────────────────────
        dc = self.data_cfg
        self.n_voxels = dc["n_voxels"]
        self.test_ds = FactFlowfMRIDataset(
            data_dir=dc["data_dir"],
            subject=dc["subject"],
            mode="test",
            fmri_mode=dc["fmri_mode"],
            clip_feature=dc["clip_feature"],
            n_voxels=dc["n_voxels"],
            pad_to=dc["pad_to"],
            fmri_channels=dc.get("fmri_channels", 1),
            fmri_spatial=dc.get("fmri_spatial", None),
            dino_feature=dc.get("dino_feature", None),
        )
        self.test_loader = DataLoader(
            self.test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        # ── Geometry ──────────────────────────────────────────────────
        self.latent_size = get_latent_size(self.data_cfg)
        self.use_cross_attn = bool(
            self.cfg.stage_2.get("params", {}).get("use_cross_attn", False)
        )
        self.use_dino = (
            self.use_cross_attn
            and self.data_cfg.get("dino_feature") is not None
        )

        # ── Source encoder config ─────────────────────────────────────
        self.clip_logvar_min = float(self.loss_cfg.get("clip_logvar_min", -10.0))
        self.clip_logvar_max = float(self.loss_cfg.get("clip_logvar_max", 10.0))

        # ── Model ─────────────────────────────────────────────────────
        self.wrapper = build_models(self.cfg, self.device)
        self.transport = build_transport(self.cfg, self.latent_size)

        # Build ODE sampler (always needed for scenarios 1 & 2)
        self.sample_fn_ode = build_sampler(self.transport, self.cfg.sampler)

        # Build SDE sampler (for scenario 3: flow_stochastic)
        self.sample_fn_sde = None
        if self.scenario == "flow_stochastic":
            sde_num_steps = getattr(args, "sde_num_steps", 250)
            sde_diffusion_norm = getattr(args, "sde_diffusion_norm", 1.0)
            sde_sampler = Sampler(self.transport)
            self.sample_fn_sde = sde_sampler.sample_sde(
                sampling_method="euler",
                diffusion_form="SBDM",
                diffusion_norm=sde_diffusion_norm,
                last_step="Mean",
                last_step_size=0.04,
                num_steps=sde_num_steps,
            )

        # Load checkpoint
        ckpt = torch.load(args.ckpt, map_location="cpu")
        self.wrapper.load_state_dict(ckpt["model"], strict=False)
        self.logger.info("Loaded model weights from %s", args.ckpt)
        del ckpt
        if str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()

        self.wrapper.eval()

    # ──────────────────────────────────────────────────────────────────
    # Source encoding helpers
    # ──────────────────────────────────────────────────────────────────

    def _get_x0_deterministic(
        self, clip_tokens: torch.Tensor, B: int,
    ) -> torch.Tensor:
        """Scenario 1 & 3: use PerceiverVE mean μ (no sampling)."""
        _, mu, _ = self.wrapper.encode_source(clip_tokens)
        return mu.permute(0, 2, 1).contiguous().view(B, *self.latent_size)

    def _get_x0_stochastic(
        self, clip_tokens: torch.Tensor, B: int,
    ) -> torch.Tensor:
        """Scenario 2: sample x₀ = μ + ε·σ from PerceiverVE."""
        _, mu, log_var = self.wrapper.encode_source(clip_tokens)
        if log_var is not None:
            log_var = torch.clamp(
                log_var,
                min=self.clip_logvar_min,
                max=self.clip_logvar_max,
            )
            std = (0.5 * log_var).exp()
            x0_tok = mu + torch.randn_like(mu) * std
        else:
            x0_tok = mu
        return x0_tok.permute(0, 2, 1).contiguous().view(B, *self.latent_size)

    # ──────────────────────────────────────────────────────────────────
    # Single-pass inference
    # ──────────────────────────────────────────────────────────────────

    def _run_single_pass(self) -> np.ndarray:
        """Run one complete inference pass over the test set.

        Returns:
            ``(N, V)`` predicted voxels (unpadded).
        """
        all_preds = []

        sample_fn = (
            self.sample_fn_sde
            if self.scenario == "flow_stochastic"
            else self.sample_fn_ode
        )

        for batch in tqdm(self.test_loader, desc="Inference", dynamic_ncols=True):
            clip_tokens = batch["clip_tokens"].to(self.device)
            clip_pool = batch["clip_pool"].to(self.device)
            dino_tokens = (
                batch["dino_tokens"].to(self.device) if self.use_dino else None
            )
            B = clip_tokens.shape[0]

            # Source x₀
            if self.scenario == "perceiver_stochastic":
                x0 = self._get_x0_stochastic(clip_tokens, B)
            else:
                # deterministic or flow_stochastic: use μ
                x0 = self._get_x0_deterministic(clip_tokens, B)

            # Flow sampling (ODE or SDE)
            context = clip_tokens if self.use_cross_attn else None
            context2 = dino_tokens if self.use_dino else None
            with autocast(**self.autocast_kwargs):
                traj = sample_fn(
                    x0, self.wrapper.dit.forward, y=clip_pool,
                    context=context, context2=context2,
                )
            pred = traj[-1].float()

            pred_flat = pred.reshape(B, -1)[:, : self.n_voxels]
            all_preds.append(pred_flat.cpu().numpy())

        return np.concatenate(all_preds, axis=0)

    # ──────────────────────────────────────────────────────────────────
    # Main evaluation
    # ──────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Run inference and compute all metrics."""

        self.logger.info("=" * 60)
        self.logger.info(
            "Scenario: %s  |  K trials: %d", self.scenario, self.k_trials,
        )
        self.logger.info(
            "Running inference on %d test samples...", len(self.test_ds),
        )
        self.logger.info("=" * 60)

        # ── Collect ground truth ──────────────────────────────────────
        all_targets = []
        for batch in self.test_loader:
            fmri_gt = batch["fmri"]
            B = fmri_gt.shape[0]
            gt_flat = fmri_gt.reshape(B, -1)[:, : self.n_voxels]
            all_targets.append(gt_flat.numpy())
        targets = np.concatenate(all_targets, axis=0)

        # ── Run K trials ──────────────────────────────────────────────
        preds_accum = None
        for k in range(self.k_trials):
            if self.k_trials > 1:
                self.logger.info("Trial %d / %d ...", k + 1, self.k_trials)

            preds_k = self._run_single_pass()

            if preds_accum is None:
                preds_accum = preds_k.astype(np.float64)
            else:
                preds_accum += preds_k.astype(np.float64)

        preds = (preds_accum / self.k_trials).astype(np.float32)

        self.logger.info(
            "Predictions: %s,  Targets: %s", preds.shape, targets.shape,
        )

        # ── Metrics ───────────────────────────────────────────────────
        metrics = compute_full_metrics(
            preds, targets,
            n_voxels=self.n_voxels,
            n_reps=self.test_ds.n_reps,
            n_images=self.test_ds.n_images,
        )

        # ── Report ────────────────────────────────────────────────────
        n_samples = preds.shape[0]
        self.logger.info("=" * 60)
        self.logger.info("EVALUATION RESULTS")
        self.logger.info("  Scenario: %s  |  K trials: %d", self.scenario, self.k_trials)
        self.logger.info("=" * 60)
        self.logger.info("  Single-trial metrics (N=%d):", n_samples)
        self.logger.info("    Per-voxel Pearson r (mean):   %.4f", metrics.mean_voxel_r)
        self.logger.info("    Per-voxel Pearson r (median): %.4f", metrics.median_voxel_r)
        self.logger.info("    Profile Pearson r (mean):     %.4f", metrics.mean_profile_r)
        self.logger.info("    MSE:                          %.6f", metrics.mse)

        if metrics.mean_img_profile_r is not None:
            self.logger.info(
                "  Image-level metrics (N=%d, rep-averaged):",
                self.test_ds.n_images,
            )
            self.logger.info(
                "    Per-voxel Pearson r (mean):   %.4f",
                metrics.mean_img_voxel_r,
            )
            self.logger.info(
                "    Profile Pearson r (mean):     %.4f",
                metrics.mean_img_profile_r,
            )
        self.logger.info("=" * 60)

        # ── Save .npz ────────────────────────────────────────────────
        if self.args.output:
            os.makedirs(os.path.dirname(self.args.output) or ".", exist_ok=True)
            save_dict = dict(
                preds=preds,
                targets=targets,
                voxel_r=metrics.voxel_r,
                profile_r=metrics.profile_r,
            )
            if metrics.img_profile_r is not None:
                save_dict["img_profile_r"] = metrics.img_profile_r
                save_dict["img_voxel_r"] = metrics.img_voxel_r
            np.savez(self.args.output, **save_dict)
            self.logger.info("Saved results to %s", self.args.output)

        # ── Append to CSV ─────────────────────────────────────────────
        csv_path = getattr(self.args, "csv_out", None)
        if csv_path:
            os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
            header = [
                "scenario", "k_trials",
                "mean_voxel_r", "median_voxel_r",
                "mean_profile_r", "mse",
                "mean_img_voxel_r", "mean_img_profile_r",
                "ckpt",
            ]
            write_header = not os.path.exists(csv_path)
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(header)
                writer.writerow([
                    self.scenario,
                    self.k_trials,
                    f"{metrics.mean_voxel_r:.6f}",
                    f"{metrics.median_voxel_r:.6f}",
                    f"{metrics.mean_profile_r:.6f}",
                    f"{metrics.mse:.6f}",
                    f"{metrics.mean_img_voxel_r:.6f}" if metrics.mean_img_voxel_r is not None else "",
                    f"{metrics.mean_img_profile_r:.6f}" if metrics.mean_img_profile_r is not None else "",
                    self.args.ckpt,
                ])
            self.logger.info("Appended CSV row to %s", csv_path)

        return {
            "mean_voxel_r": metrics.mean_voxel_r,
            "median_voxel_r": metrics.median_voxel_r,
            "mean_profile_r": metrics.mean_profile_r,
            "mse": metrics.mse,
        }
