"""
factflow_evaluator.py
=====================
Full evaluation for FactFlow fMRI synthesis (no-source flow matching).

The source x₀ is scaled Gaussian noise (``eval_noise_scale``) integrated with an
ODE solver. With ``eval_noise_scale ≈ 0`` the prediction is near-deterministic
(K=1); with full noise, ``--max_trials`` passes are averaged and metrics reported
for each K in ``--k_values`` (e.g. 1,5,10) to reduce sampling variance.

Metrics (rep-averaged GT, 1000 test images):
  - Per-voxel Pearson r (encoding metric, mean & median)
  - Profile Pearson r (mean)
  - MSE

Results can be saved to ``.npz`` and/or appended to a CSV file.
"""

from __future__ import annotations

import csv
import logging
import os
from argparse import Namespace
from typing import Dict, List

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.factflow_fmri_dataset import FactFlowfMRIDataset
from model.factflow_factory import build_models, build_transport, build_sampler
from utils.fmri_utils import auto_size_config, get_latent_size
from utils.logging_utils import create_logger
from utils.metrics import compute_full_metrics

logger = logging.getLogger(__name__)


class FactFlowEvaluator:
    """Evaluate a trained FactFlow fMRI model on the test set."""

    def __init__(self, args: Namespace) -> None:
        # ── Config ────────────────────────────────────────────────────
        self.cfg = OmegaConf.load(args.config)
        # Per-subject native sizing — must match training (no-op unless
        # data.auto_pad is set) so the model matches the checkpoint shapes.
        _autosize_msg = auto_size_config(self.cfg)
        if _autosize_msg:
            logger.info(_autosize_msg)
        self.data_cfg = OmegaConf.to_container(self.cfg.data, resolve=True)
        self.args = args

        # Scale of the Gaussian x₀. CLI --eval_noise_scale overrides config value.
        _cfg_scale = float(self.cfg.get("eval_noise_scale", 1.0))
        _cli_scale = getattr(args, "eval_noise_scale", None)
        if _cli_scale is not None:
            self.eval_noise_scale = float(_cli_scale)
            logger.info(
                "eval_noise_scale overridden by CLI: %.4f (config had %.4f)",
                self.eval_noise_scale, _cfg_scale,
            )
        else:
            self.eval_noise_scale = _cfg_scale


        # Number of stochastic passes; average the first K for each K in k_values.
        self.max_trials = max(1, getattr(args, "max_trials", 1))
        k_str = getattr(args, "k_values", "1")
        self.k_values: List[int] = sorted(int(x) for x in str(k_str).split(","))
        self.max_trials = max(self.max_trials, max(self.k_values))

        # ── Device ────────────────────────────────────────────────────
        self.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        use_bf16 = self.cfg.training.get("precision", "fp32") == "bf16"
        self.autocast_kwargs = dict(
            device_type=self.device.split(":")[0],
            dtype=torch.bfloat16,
            enabled=use_bf16,
        )
        self.logger = create_logger(name="factflow_eval")

        # ── Data (always rep-averaged GT: 1000 images, mean of 3 trials) ──
        dc = self.data_cfg
        self.roi_order = bool(dc.get("roi_order", False))

        # Multi-subject mode: config provides ``subjects`` + ``n_voxels_map``
        # instead of a single ``subject`` / ``n_voxels``. The checkpoint then
        # carries per-subject input/output adapters selected by a contiguous
        # ``subject_id`` (the subject's index in the ``subjects`` list).
        self.multisubject = ("n_voxels_map" in dc) and ("subject" not in dc)
        if self.multisubject:
            self.subjects = [int(s) for s in dc["subjects"]]
            self.n_voxels_map = {int(k): int(v) for k, v in dc["n_voxels_map"].items()}
            sel = getattr(args, "subject", None)
            if sel is not None:
                self.eval_subjects = [int(s) for s in str(sel).split(",")]
            else:
                self.eval_subjects = list(self.subjects)
        else:
            self.subjects = None
            self.n_voxels_map = {int(dc["subject"]): int(dc["n_voxels"])}
            self.eval_subjects = [int(dc["subject"])]

        # Build the first eval subject's data so the model embedder dims
        # (context_dims) and geometry are known before constructing the model.
        self._set_subject(self.eval_subjects[0])
        # Context streams define the per-stream embedder dims (matches training).
        self.cfg.stage_2.params.context_dims = list(self.test_ds.context_dims)

        # ── Geometry / model ──────────────────────────────────────────
        self.latent_size = get_latent_size(self.data_cfg)
        self.use_cross_attn = bool(
            self.cfg.stage_2.get("params", {}).get("use_cross_attn", False)
        )
        self.wrapper = build_models(self.cfg, self.device)
        self.transport = build_transport(self.cfg, self.latent_size)
        self.sample_fn = build_sampler(self.transport, self.cfg.sampler)

        # Load checkpoint
        ckpt = torch.load(args.ckpt, map_location="cpu")
        self.wrapper.load_state_dict(ckpt["model"], strict=False)
        self.logger.info("Loaded model weights from %s", args.ckpt)
        del ckpt
        if str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()
        self.wrapper.eval()

    # ──────────────────────────────────────────────────────────────────
    # Subject selection (single- or multi-subject)
    # ──────────────────────────────────────────────────────────────────

    def _set_subject(self, subject: int) -> None:
        """Build the test set/loader for ``subject`` and set its geometry.

        Sets ``self.n_voxels``, ``self.subject_id`` (contiguous adapter index,
        ``None`` in single-subject mode), ``self.test_ds``, ``self.test_loader``
        and ``self.unsort_idx``.
        """
        dc = self.data_cfg
        subject = int(subject)
        self.subject = subject
        self.n_voxels = self.n_voxels_map[subject]
        self.subject_id = self.subjects.index(subject) if self.multisubject else None
        self.test_ds = FactFlowfMRIDataset(
            data_dir=dc["data_dir"],
            subject=subject,
            mode="test",
            fmri_mode=dc["fmri_mode"],
            clip_feature=dc["clip_feature"],
            n_voxels=self.n_voxels,
            pad_to=dc["pad_to"],
            fmri_channels=dc.get("fmri_channels", 1),
            fmri_spatial=dc.get("fmri_spatial", None),
            dino_feature=dc.get("dino_feature", None),
            avg_reps=True,
            roi_order=self.roi_order,
            context_features=dc.get("context_features", None),
            subdirs=dc.get("subdirs", None),
        )
        # Index that maps ROI-sorted voxels back to anatomical order (or None).
        self.unsort_idx = self.test_ds.unsort_idx if self.roi_order else None
        self.test_loader = DataLoader(
            self.test_ds,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )

    # ──────────────────────────────────────────────────────────────────
    # Single-pass inference
    # ──────────────────────────────────────────────────────────────────

    def _run_single_pass(self) -> np.ndarray:
        """Run one inference pass over the test set. Returns ``(N, V)``."""
        all_preds = []
        for batch in tqdm(self.test_loader, desc="Inference", dynamic_ncols=True):
            clip_pool = batch["clip_pool"].to(self.device)
            contexts = [c.to(self.device) for c in batch["contexts"]]
            B = clip_pool.shape[0]

            x0 = self.eval_noise_scale * torch.randn(B, *self.latent_size, device=self.device)
            ctx = contexts if self.use_cross_attn else None
            with autocast(**self.autocast_kwargs):
                traj = self.sample_fn(
                    x0, self.wrapper.dit.forward, y=clip_pool, contexts=ctx,
                    subject_id=self.subject_id,
                )
            pred = traj[-1].float()
            del traj, x0, clip_pool, contexts, ctx

            pred_flat = pred.reshape(B, -1)[:, : self.n_voxels].cpu().numpy()
            if self.unsort_idx is not None:
                pred_flat = pred_flat[:, self.unsort_idx]  # → anatomical order
            del pred
            all_preds.append(pred_flat)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return np.concatenate(all_preds, axis=0)

    # ──────────────────────────────────────────────────────────────────
    # Metrics helpers
    # ──────────────────────────────────────────────────────────────────

    def _compute_and_report(
        self, preds: np.ndarray, targets: np.ndarray, k: int,
    ) -> Dict[str, float]:
        """Compute metrics for a K-averaged prediction and log results."""
        metrics = compute_full_metrics(
            preds, targets,
            n_voxels=self.n_voxels,
            n_reps=1,
            n_images=self.test_ds.n_images,
        )
        self.logger.info("  ── K=%d ──", k)
        self.logger.info("    Per-voxel Pearson r (mean):   %.4f", metrics.mean_voxel_r)
        self.logger.info("    Per-voxel Pearson r (median): %.4f", metrics.median_voxel_r)
        self.logger.info("    Profile Pearson r (mean):     %.4f", metrics.mean_profile_r)
        self.logger.info("    MSE:                          %.6f", metrics.mse)
        return {
            "mean_voxel_r": metrics.mean_voxel_r,
            "median_voxel_r": metrics.median_voxel_r,
            "mean_profile_r": metrics.mean_profile_r,
            "mse": metrics.mse,
            "voxel_r": metrics.voxel_r,
            "profile_r": metrics.profile_r,
        }

    def _append_csv(self, k: int, result: Dict[str, float], subject) -> None:
        """Append one CSV row for a given K and subject (``"avg"`` for mean)."""
        csv_path = getattr(self.args, "csv_out", None)
        if not csv_path:
            return
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        header = [
            "subject", "k_trials", "mean_voxel_r", "median_voxel_r",
            "mean_profile_r", "mse", "ckpt",
        ]
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
            writer.writerow([
                subject,
                k,
                f"{result['mean_voxel_r']:.6f}",
                f"{result['median_voxel_r']:.6f}",
                f"{result['mean_profile_r']:.6f}",
                f"{result['mse']:.6f}",
                self.args.ckpt,
            ])
        self.logger.info("    → CSV row appended to %s", csv_path)

    # ──────────────────────────────────────────────────────────────────
    # Main evaluation
    # ──────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _evaluate_one_subject(self) -> Dict[int, Dict[str, float]]:
        """Run inference + per-K metrics for the currently active subject."""
        self.logger.info(
            "Running inference on %d test images (subject=%s, subject_id=%s, "
            "avg_reps=True)...",
            len(self.test_ds), self.subject, self.subject_id,
        )

        # ── Ground truth (once) ───────────────────────────────────────
        all_targets = []
        for batch in self.test_loader:
            fmri_gt = batch["fmri"]
            B = fmri_gt.shape[0]
            gt_flat = fmri_gt.reshape(B, -1)[:, : self.n_voxels].numpy()
            if self.unsort_idx is not None:
                gt_flat = gt_flat[:, self.unsort_idx]  # → anatomical order
            all_targets.append(gt_flat)
        targets = np.concatenate(all_targets, axis=0)  # (N, V)

        # ── Forward passes ────────────────────────────────────────────
        all_passes: List[np.ndarray] = []
        out_dir = getattr(self.args, "output", None)
        sub_tag = f"sub{self.subject}"
        for t in range(self.max_trials):
            self.logger.info("  Pass %d / %d ...", t + 1, self.max_trials)
            preds_t = self._run_single_pass()  # (N, V)
            all_passes.append(preds_t)
            if out_dir:
                pass_dir = os.path.join(out_dir, sub_tag, "passes")
                os.makedirs(pass_dir, exist_ok=True)
                np.save(os.path.join(pass_dir, f"pass_{t:02d}.npy"), preds_t)

        # ── Metrics per K ─────────────────────────────────────────────
        results: Dict[int, Dict[str, float]] = {}
        for k in self.k_values:
            preds_avg = np.stack(all_passes[:k], axis=0).mean(axis=0).astype(np.float32)
            result = self._compute_and_report(preds_avg, targets, k)
            self._append_csv(k, result, subject=self.subject)
            if out_dir:
                os.makedirs(os.path.join(out_dir, sub_tag), exist_ok=True)
                np.savez(
                    os.path.join(out_dir, sub_tag, f"avg_k{k:02d}.npz"),
                    preds=preds_avg, targets=targets,
                    voxel_r=result["voxel_r"], profile_r=result["profile_r"],
                )
            results[k] = result
        return results

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Run inference for all selected subjects; report per-subject + average."""
        self.logger.info("=" * 60)
        self.logger.info(
            "max_trials: %d  |  k_values: %s  |  eval_noise_scale: %.3g",
            self.max_trials, self.k_values, self.eval_noise_scale,
        )
        self.logger.info("Subjects to evaluate: %s", self.eval_subjects)
        self.logger.info("=" * 60)

        # subject -> {k -> result}
        per_subject: Dict[int, Dict[int, Dict[str, float]]] = {}
        for subject in self.eval_subjects:
            self.logger.info("─" * 60)
            self.logger.info("SUBJECT %s", subject)
            self._set_subject(subject)
            per_subject[subject] = self._evaluate_one_subject()

        # ── Average across subjects per K ─────────────────────────────
        metric_keys = ["mean_voxel_r", "median_voxel_r", "mean_profile_r", "mse"]
        avg_by_k: Dict[int, Dict[str, float]] = {}
        self.logger.info("=" * 60)
        self.logger.info("AVERAGE OVER SUBJECTS %s", self.eval_subjects)
        self.logger.info("=" * 60)
        for k in self.k_values:
            avg = {mk: float(np.mean([per_subject[s][k][mk] for s in self.eval_subjects]))
                   for mk in metric_keys}
            avg_by_k[k] = avg
            self.logger.info("  ── K=%d (avg) ──", k)
            self.logger.info("    Per-voxel Pearson r (mean):   %.4f", avg["mean_voxel_r"])
            self.logger.info("    Profile Pearson r (mean):     %.4f", avg["mean_profile_r"])
            self.logger.info("    MSE:                          %.6f", avg["mse"])
            if len(self.eval_subjects) > 1:
                self._append_csv(k, {**avg, "voxel_r": None, "profile_r": None},
                                 subject="avg")
        self.logger.info("=" * 60)

        last_k = self.k_values[-1]
        return avg_by_k[last_k]
