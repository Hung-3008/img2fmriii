"""
factflow_evaluator.py
=====================
Full evaluation for FactFlow-based fMRI synthesis.

Loads a trained checkpoint, runs ODE inference on the complete test set,
and computes comprehensive metrics:
  - Per-voxel Pearson r (mean, median)
  - Profile Pearson r (mean)
  - MSE
  - Image-level metrics (rep-averaged)

Results can optionally be saved to a ``.npz`` file.
"""

from __future__ import annotations

import logging
import os
from argparse import Namespace
from typing import Dict

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.factflow_fmri_dataset import FactFlowfMRIDataset
from model.factflow_factory import build_models, build_transport, build_sampler
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
        self.args = args

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
        self.use_dino = self.use_cross_attn and self.data_cfg.get("dino_feature") is not None

        # ── Model ─────────────────────────────────────────────────────
        self.model = build_models(self.cfg, self.device)
        self.transport = build_transport(self.cfg, self.latent_size)
        self.sample_fn = build_sampler(self.transport, self.cfg.sampler)

        # Load checkpoint (raw model weights)
        ckpt = torch.load(args.ckpt, map_location="cpu")
        self.model.load_state_dict(ckpt["model"], strict=False)
        self.logger.info("Loaded model weights from %s", args.ckpt)
        del ckpt
        if str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()

        self.model.eval()

    # ──────────────────────────────────────────────────────────────────
    # Inference + metrics
    # ──────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Run inference and compute all metrics."""
        all_preds, all_targets = [], []

        self.logger.info(
            "Running inference on %d test samples...", len(self.test_ds),
        )

        for batch in tqdm(self.test_loader, desc="Inference", dynamic_ncols=True):
            clip_tokens = batch["clip_tokens"].to(self.device)
            clip_pool = batch["clip_pool"].to(self.device)
            fmri_gt = batch["fmri"].to(self.device)
            dino_tokens = batch["dino_tokens"].to(self.device) if self.use_dino else None
            B = clip_tokens.shape[0]

            # Source x0 (pure noise)
            x0 = torch.randn(B, *self.latent_size, device=self.device)

            # ODE sampling
            context = clip_tokens if self.use_cross_attn else None
            context2 = dino_tokens if self.use_dino else None
            with autocast(**self.autocast_kwargs):
                traj = self.sample_fn(
                    x0, self.model, y=clip_pool,
                    context=context, context2=context2,
                )
            pred = traj[-1].float()

            # Flatten and unpad
            pred_flat = pred.reshape(B, -1)[:, : self.n_voxels]
            gt_flat = fmri_gt.reshape(B, -1)[:, : self.n_voxels]
            all_preds.append(pred_flat.cpu().numpy())
            all_targets.append(gt_flat.cpu().numpy())

        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)
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

        # ── Save ──────────────────────────────────────────────────────
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

        return {
            "mean_voxel_r": metrics.mean_voxel_r,
            "mean_profile_r": metrics.mean_profile_r,
            "mse": metrics.mse,
        }
