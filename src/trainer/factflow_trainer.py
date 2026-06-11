"""
factflow_trainer.py
===================
Training orchestrator for FactFlow fMRI synthesis.

Supports two modes, controlled by ``source_encoder.enabled`` in the config:

**Baseline (no source encoder):**
  Standard flow matching: x₀ ~ N(0, I), velocity-MSE objective.

**Source-Conditioned Flow Matching (SCFM):**
  x₀ ~ N(μ_θ(z_CLIP, C_DINO), σ_θ²(…)) — image-conditioned source.
  Source encoder trained with two auxiliary losses (independent of velocity path):
    L_mu  = MSE(μ_θ, rep_averaged_fMRI)   — mean matches observed response
    L_kl  = KL[N(μ,σ²) || N(0,1)]        — prevents σ collapse
  x₀ is always detached before entering the flow path.

Common:
  - Validation uses noise_scale=0 (x₀ = μ_θ or 0): deterministic capacity ceiling.
  - best.pt selected by profile_r (per-image Pearson across voxels).
  - Single-GPU, bf16-capable.
"""

from __future__ import annotations

import csv
import os
from argparse import Namespace
from time import time
from typing import Dict, Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.factflow_fmri_dataset import FactFlowfMRIDataset
from data.multisubject_fmri_dataset import MultiSubjectfMRIDataset
from model.factflow_factory import build_models, build_transport, build_sampler
from utils.checkpoint import (
    load_checkpoint,
    find_last_checkpoint,
    save_checkpoint,
    save_rolling_last,
)
from utils.fmri_utils import auto_size_config, create_pad_mask, get_latent_size
from utils.logging_utils import create_logger
from utils.metrics import (
    masked_mse,
    pearson_corr_per_sample,
    voxel_pearson,
    compute_voxel_reliability,
)
from utils.training_utils import build_optimizer_and_scheduler


class FactFlowTrainer:
    """End-to-end trainer for FactFlow fMRI synthesis."""

    def __init__(self, args: Namespace) -> None:
        # ── Config ────────────────────────────────────────────────────
        self.cfg = OmegaConf.load(args.config)
        # Per-subject native sizing: derive pad_to/seq_len from n_voxels
        # (no-op unless data.auto_pad is set). Must run before the containers
        # below and before the config snapshot is saved.
        self._autosize_msg = auto_size_config(self.cfg)
        self.data_cfg = OmegaConf.to_container(self.cfg.data, resolve=True)
        self.train_cfg = OmegaConf.to_container(self.cfg.training, resolve=True)
        self.args = args

        # ROI voxel ordering: reorder voxels so same-ROI voxels are contiguous
        # (data-level only; the model is a plain cross-attention DiT). Predictions
        # are un-sorted back to anatomical order at eval/export time.
        self.roi_order = bool(self.data_cfg.get("roi_order", False))

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
        if self._autosize_msg:
            self.logger.info(self._autosize_msg)

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

        # ── Model / transport / sampler ───────────────────────────────
        self.wrapper = build_models(self.cfg, self.device)
        self.transport = build_transport(self.cfg, self.latent_size)
        self.sample_fn = build_sampler(self.transport, self.cfg.sampler)

        # ── ROI-Stratified Feature Routing setup ─────────────────────
        # Must run after model build (needs dit.set_roi_buckets) and after
        # dataset build (needs sort_idx from train_ds).
        self._setup_roi_routing()

        # ── Optimizer & scheduler ─────────────────────────────────────
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
        # Best-checkpoint criterion: profile_r (per-image pattern correlation),
        # which is directly comparable to SynBrain / MindSimulator baselines.
        self.best_metric = -1.0   # tracks best val_profile_r
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

        # Stochastic noise scale used by _inline_eval (quick sanity check).
        # _validate() always uses noise_scale=0 (deterministic ceiling) so that
        # best.pt reflects the true model capacity, independent of K-averaging.
        self.eval_noise_scale = float(self.cfg.get("eval_noise_scale", 1.0))

        # ── Subject-Conditioned Flow Matching ────────────────────────────
        self.use_subject_cond = (
            self.wrapper.dit.subject_embedder is not None
        )
        if self.use_subject_cond:
            sc_cfg = OmegaConf.to_container(self.cfg.get("subject_cond", OmegaConf.create({})), resolve=True)
            self.logger.info(
                "Subject-cond ON: mode=%s  n_subjects=%d  dropout=%.2f",
                sc_cfg.get("mode", "learned"),
                int(sc_cfg.get("n_subjects", 4)),
                float(sc_cfg.get("subject_dropout", 0.1)),
            )
        self.use_source_encoder = self.wrapper.source_encoder is not None
        if self.use_source_encoder:
            se_cfg = OmegaConf.to_container(self.cfg.get("source_encoder", {}), resolve=True)
            self.lambda_mu  = float(se_cfg.get("lambda_mu",  0.1))
            self.lambda_kl  = float(se_cfg.get("lambda_kl",  0.001))
            self.logger.info(
                "SCFM mode ON: λ_mu=%.4f  λ_kl=%.4f",
                self.lambda_mu, self.lambda_kl,
            )
        else:
            self.lambda_mu = self.lambda_kl = 0.0

        # Cross-attention conditioning (DINOv2 / Gabor tokens → DiT).
        self.use_cross_attn = bool(
            self.cfg.stage_2.get("params", {}).get("use_cross_attn", False)
        )

        self.voxel_weight = self._build_voxel_weight()

        self.wrapper.train()

    def _setup_roi_routing(self) -> None:
        """Build patch-level ROI bucket IDs and register them in the DiT.

        Only active when ``use_roi_routing: true`` is set in the config.
        ROI bucket assignment follows the NSD visual hierarchy (streams atlas):

            0 = early visual   (V1v/d, V2v/d, V3v/d)
            1 = mid visual     (V3A, V3B, hV4, VO1, VO2)
            2 = high visual    (everything else: LO, PFS, FBA, FFA, OPA, …)

        The voxels are in ROI-sorted order (roi_order=True in config), so the
        atlas labels from roi_meta must be permuted via sort_idx before mapping
        to patch space.
        """
        dit = self.wrapper.dit
        if not getattr(dit, "use_roi_routing", False):
            return

        n_roi_buckets = int(self.cfg.stage_2.params.get("n_roi_buckets", 3))
        dc = self.data_cfg
        subject = dc["subject"]
        roi_path = os.path.join(
            dc["data_dir"], f"subj0{subject}", f"roi_meta_sub{subject}.npz"
        )
        meta = np.load(roi_path)

        # ``roi_labels`` is an integer array (n_voxels,) with streams atlas IDs.
        # Fallback: use sort_idx position as a proxy bucket if label absent.
        if "roi_labels" in meta:
            roi_labels = meta["roi_labels"].astype(np.int64)  # (n_voxels,)
            # Map streams atlas labels → 3 visual hierarchy buckets.
            # Verified mapping from roi_meta_sub1.npz:
            #   0        : unlabeled (nsdgeneral outside named ROIs)  → high (2)
            #   1-6      : V1v,V1d,V2v,V2d,V3v,V3d                   → early (0)
            #   7-15     : V3A,V3B,V3CD,V4,hV4,VO1,VO2,PHC1,PHC2     → mid   (1)
            #   16-34    : LO,PFS,OPA,PPA,RSC,OFA,FFA,FBA,IPS…        → high  (2)
            bucket = np.full_like(roi_labels, 2)                       # default: high
            bucket[(roi_labels >= 1) & (roi_labels <= 6)] = 0          # early V1/V2/V3
            bucket[(roi_labels >= 7) & (roi_labels <= 15)] = 1         # mid V3A-PHC
        else:
            # No label info: divide voxels into equal thirds by sorted position
            n_voxels = dc["n_voxels"]
            bucket = np.zeros(n_voxels, dtype=np.int64)
            third = n_voxels // 3
            bucket[third:2*third] = 1
            bucket[2*third:] = 2
            self.logger.warning(
                "roi_meta has no 'roi_labels' key — using positional thirds "
                "as ROI buckets (less accurate)."
            )

        # Apply same sort permutation as applied to fMRI voxels in the dataset
        if self.roi_order and self.train_ds.sort_idx is not None:
            bucket = bucket[self.train_ds.sort_idx]  # now in sorted voxel order

        dit.set_roi_buckets(
            voxel_bucket_ids=bucket,
            pad_to=dc["pad_to"],
            n_roi_buckets=n_roi_buckets,
        )
        self.logger.info(
            "ROI routing ON: buckets=%d  early=%d  mid=%d  high=%d  patches=%d",
            n_roi_buckets,
            int((bucket == 0).sum()), int((bucket == 1).sum()),
            int((bucket == 2).sum()),
            dc["pad_to"] // self.data_cfg.get("patch_size",
                self.cfg.stage_2.params.get("patch_size", 32)),
        )

    # ──────────────────────────────────────────────────────────────────
    # Voxel SNR weight
    # ──────────────────────────────────────────────────────────────────

    def _build_voxel_weight(self) -> Optional[torch.Tensor]:
        """Per-voxel noise-ceiling weight over ``pad_to`` (None if disabled)."""
        if not self.cfg.get("losses", {}).get("use_snr_weight", False):
            return None
        nc = compute_voxel_reliability(
            self.train_ds.fmri_data, self.train_ds.n_reps,
        )  # (V,) — original voxel order
        # The loss is computed in the (possibly ROI-reordered) voxel space, so
        # the per-voxel weight must follow the same permutation.
        if self.roi_order and self.train_ds.sort_idx is not None:
            nc = nc[self.train_ds.sort_idx]
        w = np.zeros(self.data_cfg["pad_to"], dtype=np.float64)
        w[: self.data_cfg["n_voxels"]] = nc
        # Normalise so the mean weight over real voxels is 1 (keeps loss scale).
        mean_w = w[: self.data_cfg["n_voxels"]].mean()
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
        subjects_cfg = dc.get("subjects", None)  # multi-subject mode

        common_kwargs = dict(
            data_dir   = dc["data_dir"],
            fmri_mode  = dc["fmri_mode"],
            clip_feature = dc["clip_feature"],
            fmri_channels = dc.get("fmri_channels", 1),
            fmri_spatial  = dc.get("fmri_spatial", None),
            avg_reps      = dc.get("avg_reps", False),
            dino_feature  = dc.get("dino_feature", None),
            roi_order     = self.roi_order,
            context_features = dc.get("context_features", None),
            subdirs       = dc.get("subdirs", None),
        )

        if subjects_cfg and len(subjects_cfg) > 1:
            # ── Multi-subject mode ────────────────────────────────────
            subjects = list(subjects_cfg)
            per_subject_kwargs = {}
            for s in subjects:
                sk = dc.get(f"sub{s}", dc.get(f"subject_{s}", {}))
                per_subject_kwargs[s] = dict(sk) if sk else {}
                # Apply global n_voxels / pad_to unless overridden per-subject
                if "n_voxels" not in per_subject_kwargs[s]:
                    per_subject_kwargs[s]["n_voxels"] = dc["n_voxels"]
                if "pad_to" not in per_subject_kwargs[s]:
                    per_subject_kwargs[s]["pad_to"] = dc["pad_to"]

            common_seq_len = int(dc.get("common_seq_len", dc["pad_to"]))
            patch_size = int(self.cfg.stage_2.params.get("patch_size", 32))

            self.train_ds = MultiSubjectfMRIDataset(
                subjects=subjects, mode="train",
                common_seq_len=common_seq_len, patch_size=patch_size,
                per_subject_kwargs=per_subject_kwargs, **common_kwargs,
            )
            self.test_ds = MultiSubjectfMRIDataset(
                subjects=subjects, mode="test",
                common_seq_len=common_seq_len, patch_size=patch_size,
                per_subject_kwargs=per_subject_kwargs, **common_kwargs,
            )
            self.val_ds = MultiSubjectfMRIDataset(
                subjects=subjects, mode="test",
                common_seq_len=common_seq_len, patch_size=patch_size,
                per_subject_kwargs=per_subject_kwargs,
                **{**common_kwargs, "avg_reps": True},
            )
            self.is_multisubject = True
        else:
            # ── Single-subject mode (original) ───────────────────────────
            ds_kwargs = dict(
                subject=dc["subject"],
                n_voxels=dc["n_voxels"],
                pad_to=dc["pad_to"],
                **common_kwargs,
            )
            self.train_ds = FactFlowfMRIDataset(mode="train", **ds_kwargs)
            self.test_ds  = FactFlowfMRIDataset(mode="test",  **ds_kwargs)
            self.val_ds   = FactFlowfMRIDataset(mode="test",  **{**ds_kwargs, "avg_reps": True})
            self.is_multisubject = False

        # Inject context_dims into DiT config before model build
        self.cfg.stage_2.params.context_dims = list(self.train_ds.context_dims)

        batch_size = int(self.train_cfg.get("batch_size", 64))
        num_workers = int(self.train_cfg.get("num_workers", 4))
        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=bool(self.train_cfg.get("pin_memory", True)),
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
            self.best_metric = info["best_val_pcc"]

    # ──────────────────────────────────────────────────────────────────
    # Forward + loss
    # ──────────────────────────────────────────────────────────────────

    def _compute_loss(
        self,
        x1: torch.Tensor,
        clip_pool: torch.Tensor,
        contexts=None,
        subject_ids=None,
        roi_profile=None,
    ) -> tuple[torch.Tensor, dict]:
        """One training step: sample x₀, interpolate, compute all losses."""
        ctx_for_vel = contexts if self.use_cross_attn else None

        # ── Source x₀ ────────────────────────────────────────────────
        if self.use_source_encoder:
            C_dino = contexts[0] if contexts else None
            x0 = self.wrapper.sample_source(
                z_pool=clip_pool, C_dino=C_dino, noise_scale=1.0,
            )
        else:
            x0 = torch.randn_like(x1)

        # ── Velocity loss ─────────────────────────────────────────
        t = self.transport.sample_timestep(x1)
        t, xt, ut = self.transport.path_sampler.plan(t, x0, x1)
        v_pred = self.wrapper.predict_velocity(
            x=xt, t=t, y=clip_pool, contexts=ctx_for_vel,
            subject_ids=subject_ids, roi_profile=roi_profile,
        )
        loss_vel = masked_mse(v_pred, ut, self.pad_mask, weight=self.voxel_weight)

        # ── Auxiliary source-encoder losses ────────────────────────
        loss_src = torch.zeros((), device=self.device)
        log_dict: dict = {"loss_vel": loss_vel.item()}

        if self.use_source_encoder:
            C_dino = contexts[0] if contexts else None
            loss_mu = self.wrapper.source_encoder.loss_mu(
                z_pool=clip_pool, C_dino=C_dino, x1=x1, pad_mask=self.pad_mask,
            )
            loss_kl = self.wrapper.source_encoder.loss_kl(
                z_pool=clip_pool, C_dino=C_dino, pad_mask=self.pad_mask,
            )
            loss_src = self.lambda_mu * loss_mu + self.lambda_kl * loss_kl
            log_dict["loss_mu"]  = loss_mu.item()
            log_dict["loss_kl"]  = loss_kl.item()
            log_dict["loss_src"] = loss_src.item()

        total_loss = loss_vel + loss_src
        log_dict["loss_total"] = total_loss.item()
        return total_loss, log_dict

    # ──────────────────────────────────────────────────────────────────
    # Inference helper (shared by inline eval & full validation)
    # ──────────────────────────────────────────────────────────────────

    def _sample(self, model: torch.nn.Module, clip_pool, contexts, B: int,
                noise_scale: float | None = None,
                subject_ids=None, roi_profile=None) -> torch.Tensor:
        """ODE-sample fMRI. Returns ``(B, *latent)``."""
        scale = self.eval_noise_scale if noise_scale is None else noise_scale
        ctx = contexts if self.use_cross_attn else None

        if model.source_encoder is not None:
            C_dino = contexts[0] if contexts else None
            with torch.no_grad():
                x0 = model.sample_source(
                    z_pool=clip_pool, C_dino=C_dino, noise_scale=scale,
                )
        else:
            x0 = scale * torch.randn(B, *self.latent_size, device=self.device)

        with autocast(**self.autocast_kwargs):
            traj = self.sample_fn(
                x0, model.dit.forward, y=clip_pool, contexts=ctx,
                subject_ids=subject_ids, roi_profile=roi_profile,
            )
        return traj[-1].float()

    @torch.no_grad()
    def _inline_eval(self) -> None:
        """Quick ODE-based eval on a small test subset (uses eval_noise_scale)."""
        self.wrapper.eval()
        n_eval = min(64, len(self.test_ds))
        loader = DataLoader(self.test_ds, batch_size=n_eval, shuffle=False, num_workers=0)
        batch = next(iter(loader))

        fmri_gt = batch["fmri"].to(self.device)
        clip_pool = batch["clip_pool"].to(self.device)
        contexts = [c.to(self.device) for c in batch["contexts"]]

        pred = self._sample(self.wrapper, clip_pool, contexts, n_eval)
        preds_flat = pred.reshape(n_eval, -1)[:, self.pad_mask]
        gts_flat = fmri_gt.reshape(n_eval, -1)[:, self.pad_mask]
        voxel_r  = voxel_pearson(preds_flat, gts_flat).mean().item()
        profile_r = pearson_corr_per_sample(pred, fmri_gt, self.pad_mask).mean().item()
        mse_val   = masked_mse(pred, fmri_gt, self.pad_mask).item()
        self.logger.info(
            "  [Eval step=%d] voxel_r=%.4f  profile_r=%.4f  mse=%.5f",
            self.train_steps, voxel_r, profile_r, mse_val,
        )
        self.wrapper.train()

    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict[str, float]:
        """Full validation over the rep-averaged test set.

        Always uses **noise_scale=0** (deterministic ODE) so that the metric
        reflects the model's ceiling capacity, independent of stochastic
        averaging.  Reports both voxel_r (encoding) and profile_r (pattern
        correlation, comparable to SynBrain / MindSimulator).
        """
        loader = DataLoader(
            self.val_ds, batch_size=32, shuffle=False, num_workers=0, drop_last=False,
        )
        self.wrapper.eval()
        preds_all, gts_all = [], []
        profile_rs = []
        mse_sum, n = 0.0, 0
        for batch in loader:
            fmri_gt = batch["fmri"].to(self.device)
            clip_pool = batch["clip_pool"].to(self.device)
            contexts = [c.to(self.device) for c in batch["contexts"]]
            B = fmri_gt.shape[0]

            # Deterministic (noise_scale=0): ceiling prediction E[x|c]
            pred = self._sample(self.wrapper, clip_pool, contexts, B, noise_scale=0.0)
            mse_sum += masked_mse(pred, fmri_gt, self.pad_mask).item() * B
            n += B
            preds_all.append(pred.reshape(B, -1)[:, self.pad_mask].cpu())
            gts_all.append(fmri_gt.reshape(B, -1)[:, self.pad_mask].cpu())
            # Profile r: per-image, across voxels (same as SynBrain metric)
            profile_rs.append(
                pearson_corr_per_sample(pred, fmri_gt, self.pad_mask).cpu()
            )

        preds_cat = torch.cat(preds_all)   # (N, V)
        gts_cat   = torch.cat(gts_all)     # (N, V)
        val_voxel_r   = voxel_pearson(preds_cat, gts_cat).mean().item()
        val_profile_r = torch.cat(profile_rs).mean().item()
        val_mse       = mse_sum / n
        self.logger.info(
            "  [Val epoch=%d] voxel_r=%.4f  profile_r=%.4f  mse=%.5f  n=%d  "
            "(noise_scale=0, deterministic)",
            epoch + 1, val_voxel_r, val_profile_r, val_mse, n,
        )
        self.wrapper.train()
        return {"mse": val_mse, "voxel_r": val_voxel_r, "profile_r": val_profile_r, "n": n}

    # ──────────────────────────────────────────────────────────────────
    # Main training loop
    # ──────────────────────────────────────────────────────────────────

    def train(self) -> None:
        """Run the full training loop."""
        history_path = os.path.join(self.exp_dir, "history.csv")
        history_exists = os.path.exists(history_path)
        history_file = open(history_path, "a", newline="")
        history_writer = csv.writer(history_file)
        if not history_exists:
            history_writer.writerow(
                ["epoch", "step", "train_loss",
                 "val_mse", "val_voxel_r", "val_profile_r", "lr"]
            )
            history_file.flush()

        running_loss = 0.0   # since last log_every
        epoch_loss = 0.0     # since last validation
        step_loss = 0.0      # current optimizer step (across micro-batches)
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
                clip_pool = batch["clip_pool"].to(self.device)
                contexts = [c.to(self.device) for c in batch["contexts"]]
                subject_ids = batch["subject_id"].to(self.device) if "subject_id" in batch else None
                roi_profile = batch["roi_profile"].to(self.device) if "roi_profile" in batch else None

                with autocast(**self.autocast_kwargs):
                    loss, log_dict = self._compute_loss(
                        x1, clip_pool, contexts,
                        subject_ids=subject_ids, roi_profile=roi_profile,
                    )

                (loss / self.grad_accum).backward()
                accum_counter += 1

                micro_loss = log_dict["loss_total"] / self.grad_accum
                running_loss += micro_loss
                step_loss += micro_loss
                postfix = {"loss": f"{loss.item():.4f}",
                           "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"}
                if self.use_source_encoder:
                    postfix["mu"] = f"{log_dict.get('loss_mu', 0):.4f}"
                    postfix["kl"] = f"{log_dict.get('loss_kl', 0):.5f}"
                pbar.set_postfix(postfix)

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
                epoch_steps += 1
                step_loss = 0.0

                # ── Periodic logging ──
                if self.train_steps % self.log_every == 0:
                    elapsed = time() - wall_start
                    sps = log_steps / elapsed if elapsed > 0 else 0
                    lr = self.scheduler.get_last_lr()[0]
                    self.logger.info(
                        "[step=%07d ep=%d] loss=%.5f lr=%.2e steps/s=%.1f",
                        self.train_steps, epoch, running_loss / log_steps, lr, sps,
                    )
                    running_loss = 0.0
                    log_steps = 0
                    wall_start = time()

                # ── Periodic checkpoint ──
                if self.train_steps % self.ckpt_every == 0 and self.train_steps > 0:
                    save_checkpoint(
                        os.path.join(self.ckpt_dir, f"{self.train_steps:07d}.pt"),
                        self.wrapper, self.optimizer, self.scheduler,
                        self.train_steps, epoch, self.best_metric,
                    )

                # ── Rolling last checkpoint ──
                if self.train_steps % 1000 == 0 and self.train_steps > 0:
                    save_rolling_last(
                        self.ckpt_dir, self.wrapper,
                        self.optimizer, self.scheduler,
                        self.train_steps, epoch, self.best_metric,
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
                    f"{val['mse']:.6f}",
                    f"{val['voxel_r']:.6f}",
                    f"{val['profile_r']:.6f}",
                    f"{lr:.2e}",
                ])
                history_file.flush()
                epoch_loss = 0.0
                epoch_steps = 0

                # Save best checkpoint by profile_r (per-image pattern correlation,
                # directly comparable to SynBrain / MindSimulator baselines).
                if val["profile_r"] > self.best_metric:
                    self.best_metric = val["profile_r"]
                    save_checkpoint(
                        os.path.join(self.ckpt_dir, "best.pt"),
                        self.wrapper, self.optimizer, self.scheduler,
                        self.train_steps, epoch, self.best_metric,
                        val_mse=val["mse"],
                    )
                    self.logger.info(
                        "New best profile_r: %.4f  (voxel_r: %.4f)",
                        self.best_metric, val["voxel_r"],
                    )

            if hit_max:
                break

        # ── Final checkpoint ──
        save_checkpoint(
            os.path.join(self.ckpt_dir, f"final-{self.train_steps}.pt"),
            self.wrapper, self.optimizer, self.scheduler,
            self.train_steps, self.epochs, self.best_metric,
        )
        history_file.close()
        self.logger.info("Training complete. History: %s", history_path)
