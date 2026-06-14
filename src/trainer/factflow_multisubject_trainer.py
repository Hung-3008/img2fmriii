"""
factflow_multisubject_trainer.py
================================
Multi-subject trainer for FactFlow fMRI synthesis (shared trunk + per-subject
adapters).

Architecture (Option B / MindEye2-style):
  - One shared DiT1D trunk (transformer blocks + t/CLIP/cross-attn embedders).
  - Per-subject input patch-embed and output readout, selected by ``subject_id``.
  - All subjects padded to the same ``pad_to`` (e.g. 16384 = 2^14); flow matching
    runs directly in native-padded voxel space, so metrics stay in each subject's
    native voxel space → directly comparable to published baselines.

Training:
  - Subject-homogeneous batches (SubjectBatchSampler); grad-accum mixes subjects.
  - Per-subject pad_mask + SNR weight + validation; best.pt by MEAN profile_r.

This file deliberately does NOT touch the single-subject FactFlowTrainer.
"""

from __future__ import annotations

import csv
import os
from argparse import Namespace
from time import time
from typing import Dict, List, Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.multi_subject_dataset import (
    build_subject_datasets,
    MultiSubjectDataset,
    SubjectBatchSampler,
)
from model.factflow_factory import build_models, build_transport, build_sampler
from utils.checkpoint import (
    load_checkpoint,
    find_last_checkpoint,
    save_checkpoint,
    save_rolling_last,
)
from utils.fmri_utils import create_pad_mask, get_latent_size
from utils.logging_utils import create_logger
from utils.metrics import (
    masked_mse,
    pearson_corr_per_sample,
    voxel_pearson,
    compute_voxel_reliability,
)
from utils.training_utils import build_optimizer_and_scheduler


class FactFlowMultiSubjectTrainer:
    """Shared-trunk, per-subject-adapter trainer for FactFlow fMRI synthesis."""

    def __init__(self, args: Namespace) -> None:
        # ── Config ────────────────────────────────────────────────────
        self.cfg = OmegaConf.load(args.config)
        self.data_cfg = OmegaConf.to_container(self.cfg.data, resolve=True)
        self.train_cfg = OmegaConf.to_container(self.cfg.training, resolve=True)
        self.args = args

        self.subjects: List[int] = list(self.data_cfg["subjects"])
        self.n_subjects = len(self.subjects)
        self.n_voxels_map = {int(k): int(v) for k, v in self.data_cfg["n_voxels_map"].items()}
        self.pad_to = int(self.data_cfg["pad_to"])
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
        subj_tag = "".join(str(s) for s in self.subjects)
        exp_name = args.exp_name or f"factflow_ms_sub{subj_tag}"
        self.exp_dir = os.path.join(args.exps_dir, exp_name)
        self.ckpt_dir = os.path.join(self.exp_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.logger = create_logger(self.exp_dir, name="factflow_ms_trainer")
        self.logger.info(
            "Multi-subject: subjects=%s  n_voxels=%s  pad_to=%d",
            self.subjects, [self.n_voxels_map[s] for s in self.subjects], self.pad_to,
        )

        cfg_path = os.path.join(self.exp_dir, "config.yaml")
        if not os.path.exists(cfg_path):
            OmegaConf.save(self.cfg, cfg_path)

        # ── Data ──────────────────────────────────────────────────────
        self._build_datasets()

        # ── Geometry ──────────────────────────────────────────────────
        self.latent_size = get_latent_size({**self.data_cfg, "pad_to": self.pad_to})
        # Per-subject pad masks (contiguous subject index → (pad_to,) bool).
        self.pad_masks = {
            sidx: create_pad_mask(self.n_voxels_map[s], self.pad_to, self.device)
            for sidx, s in enumerate(self.subjects)
        }

        # ── Model / transport / sampler ───────────────────────────────
        # Inject multi-subject + per-stream dims into the DiT config before build.
        self.cfg.stage_2.params.n_subjects = self.n_subjects
        self.cfg.stage_2.params.seq_len = self.pad_to
        self.cfg.stage_2.params.context_dims = list(self.train_dss[0].context_dims)
        self.wrapper = build_models(self.cfg, self.device)
        self.transport = build_transport(self.cfg, self.latent_size)
        self.sample_fn = build_sampler(self.transport, self.cfg.sampler)

        # Per-subject ROI-stratified feature routing (no-op unless enabled).
        self._setup_roi_routing()

        # ── Optimizer & scheduler ─────────────────────────────────────
        self.grad_accum = int(self.train_cfg.get("grad_accum_steps", 1))
        self.steps_per_epoch = max(len(self.batch_sampler) // self.grad_accum, 1)
        self.epochs = int(self.train_cfg.get("epochs", 50))
        self.optimizer, self.scheduler, opt_msg, sched_msg = (
            build_optimizer_and_scheduler(
                self.wrapper.parameters(), self.train_cfg,
                self.steps_per_epoch, self.epochs,
            )
        )
        self.logger.info(opt_msg)
        self.logger.info(sched_msg)

        # ── State ─────────────────────────────────────────────────────
        self.train_steps = 0
        self.start_epoch = 0
        self.best_metric = -1.0   # tracks best MEAN val_profile_r
        self._maybe_resume()

        # ── Training hyper-params ─────────────────────────────────────
        use_bf16 = self.train_cfg.get("precision", "fp32") == "bf16"
        self.autocast_kwargs = dict(
            device_type=self.device.split(":")[0],
            dtype=torch.bfloat16, enabled=use_bf16,
        )
        self.clip_grad = float(self.train_cfg.get("clip_grad", 1.0))
        self.log_every = int(self.train_cfg.get("log_every", 50))
        self.ckpt_every = int(self.train_cfg.get("ckpt_every", 5000))
        self.val_every = int(self.train_cfg.get("val_every", 1))
        self.eval_noise_scale = float(self.cfg.get("eval_noise_scale", 1.0))
        self.use_cross_attn = bool(
            self.cfg.stage_2.get("params", {}).get("use_cross_attn", False)
        )

        # Per-subject SNR weights (contiguous index → (pad_to,) tensor or None).
        self.voxel_weights = self._build_voxel_weights()
        self.wrapper.train()

    # ──────────────────────────────────────────────────────────────────
    # Dataset construction
    # ──────────────────────────────────────────────────────────────────

    def _base_kwargs(self) -> dict:
        dc = self.data_cfg
        return dict(
            data_dir=dc["data_dir"],
            fmri_mode=dc["fmri_mode"],
            clip_feature=dc["clip_feature"],
            pad_to=self.pad_to,
            fmri_channels=dc.get("fmri_channels", 1),
            fmri_spatial=dc.get("fmri_spatial", None),
            dino_feature=dc.get("dino_feature", None),
            roi_order=self.roi_order,
            context_features=dc.get("context_features", None),
            subdirs=dc.get("subdirs", None),
            n_voxels_map=self.n_voxels_map,
        )

    def _build_datasets(self) -> None:
        # Train: per-subject datasets (avg_reps from config), concatenated.
        avg_reps = bool(self.data_cfg.get("avg_reps", False))
        self.train_dss = build_subject_datasets(
            self.subjects, self._base_kwargs(), mode="train", avg_reps=avg_reps,
        )
        self.train_ds = MultiSubjectDataset(self.train_dss)

        # Val: always rep-averaged test set, kept per-subject for per-subject
        # validation (each subject's metric in its native voxel space).
        self.val_dss = build_subject_datasets(
            self.subjects, self._base_kwargs(), mode="test", avg_reps=True,
        )

        batch_size = int(self.train_cfg.get("batch_size", 32))
        num_workers = int(self.train_cfg.get("num_workers", 2))
        self.batch_sampler = SubjectBatchSampler(
            self.train_ds.boundaries, batch_size=batch_size,
            shuffle=True, drop_last=True,
        )
        self.train_loader = DataLoader(
            self.train_ds,
            batch_sampler=self.batch_sampler,
            num_workers=num_workers,
            pin_memory=bool(self.train_cfg.get("pin_memory", False)),
            persistent_workers=num_workers > 0,
        )
        self.logger.info(
            "Train: %d samples over %d subjects, %d batches/epoch (bs=%d)",
            len(self.train_ds), self.n_subjects, len(self.batch_sampler), batch_size,
        )

    def _setup_roi_routing(self) -> None:
        """Build PER-SUBJECT patch-level ROI bucket IDs and register them in the DiT.

        Only active when ``use_roi_routing: true``. The StreamRouter is shared
        across subjects (the early/mid/high buckets are universal); only the
        patch->bucket map differs per subject, since each has its own roi_order
        voxel layout. Bucket scheme (NSD streams atlas):

            0 = early visual  (V1v/d, V2v/d, V3v/d)
            1 = mid visual    (V3A, V3B, hV4, VO1, VO2, PHC)
            2 = high visual   (everything else: LO, FFA, PPA, OPA, …)
        """
        dit = self.wrapper.dit
        if not getattr(dit, "use_roi_routing", False):
            return

        n_roi_buckets = int(self.cfg.stage_2.params.get("n_roi_buckets", 3))
        dc = self.data_cfg
        for sidx, s in enumerate(self.subjects):
            ds = self.train_dss[sidx]
            roi_path = os.path.join(
                dc["data_dir"], f"subj0{s}", f"roi_meta_sub{s}.npz"
            )
            meta = np.load(roi_path) if os.path.exists(roi_path) else {}

            if "roi_labels" in meta:
                roi_labels = meta["roi_labels"].astype(np.int64)  # (n_voxels,)
                bucket = np.full_like(roi_labels, 2)                    # default: high
                if int(roi_labels.max()) <= 7:
                    # NSD "streams" atlas (0=unknown, 1=early, 2-4=mid{ventral,
                    # lateral,parietal}, 5-7={ventral,lateral,parietal}=high).
                    bucket[roi_labels == 1] = 0                          # early
                    bucket[(roi_labels >= 2) & (roi_labels <= 4)] = 1   # mid
                    # labels 5-7 and 0(unknown) keep default high (2)
                else:
                    # Legacy finer atlas (labels up to ~34).
                    bucket[(roi_labels >= 1) & (roi_labels <= 6)] = 0   # early
                    bucket[(roi_labels >= 7) & (roi_labels <= 15)] = 1  # mid
            else:
                n_voxels = ds.n_voxels
                bucket = np.zeros(n_voxels, dtype=np.int64)
                third = n_voxels // 3
                bucket[third:2 * third] = 1
                bucket[2 * third:] = 2
                self.logger.warning(
                    "subj%d roi_meta missing 'roi_labels' — using positional "
                    "thirds as ROI buckets (less accurate).", s,
                )

            # Match the permutation applied to fMRI voxels in the dataset.
            if self.roi_order and ds.sort_idx is not None:
                bucket = bucket[ds.sort_idx]

            dit.set_roi_buckets(
                voxel_bucket_ids=bucket,
                pad_to=self.pad_to,
                n_roi_buckets=n_roi_buckets,
                subject_idx=sidx,
            )
            self.logger.info(
                "ROI routing subj%d (idx=%d): early=%d  mid=%d  high=%d",
                s, sidx,
                int((bucket == 0).sum()), int((bucket == 1).sum()),
                int((bucket == 2).sum()),
            )

    def _build_voxel_weights(self) -> Dict[int, Optional[torch.Tensor]]:
        weights: Dict[int, Optional[torch.Tensor]] = {}
        use_snr = bool(self.cfg.get("losses", {}).get("use_snr_weight", False))
        for sidx, ds in enumerate(self.train_dss):
            if not use_snr:
                weights[sidx] = None
                continue
            nc = compute_voxel_reliability(ds.fmri_data, ds.n_reps)  # (V,)
            if self.roi_order and ds.sort_idx is not None:
                nc = nc[ds.sort_idx]
            w = np.zeros(self.pad_to, dtype=np.float64)
            w[: ds.n_voxels] = nc
            mean_w = w[: ds.n_voxels].mean()
            if mean_w > 0:
                w = w / mean_w
            weights[sidx] = torch.tensor(w, dtype=torch.float32, device=self.device)
        if use_snr:
            self.logger.info("SNR-weighted loss ON (per-subject noise-ceiling).")
        return weights

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

    def _compute_loss(self, x1, clip_pool, contexts, sidx: int) -> torch.Tensor:
        x0 = torch.randn_like(x1)
        t = self.transport.sample_timestep(x1)
        t, xt, ut = self.transport.path_sampler.plan(t, x0, x1)
        ctx = contexts if self.use_cross_attn else None
        v_pred = self.wrapper.predict_velocity(
            x=xt, t=t, y=clip_pool, contexts=ctx, subject_id=sidx,
        )
        return masked_mse(
            v_pred, ut, self.pad_masks[sidx], weight=self.voxel_weights[sidx],
        )

    def _sample(self, clip_pool, contexts, B: int, sidx: int,
                noise_scale: float | None = None) -> torch.Tensor:
        scale = self.eval_noise_scale if noise_scale is None else noise_scale
        x0 = scale * torch.randn(B, *self.latent_size, device=self.device)
        ctx = contexts if self.use_cross_attn else None
        with autocast(**self.autocast_kwargs):
            traj = self.sample_fn(
                x0, self.wrapper.dit.forward,
                y=clip_pool, contexts=ctx, subject_id=sidx,
            )
        return traj[-1].float()

    # ──────────────────────────────────────────────────────────────────
    # Validation (per-subject, native space; macro-average across subjects)
    # ──────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict[str, float]:
        self.wrapper.eval()
        per_subj = {}
        for sidx, ds in enumerate(self.val_dss):
            loader = DataLoader(ds, batch_size=32, shuffle=False,
                                num_workers=0, drop_last=False)
            pad_mask = self.pad_masks[sidx]
            preds_all, gts_all, profile_rs = [], [], []
            mse_sum, n = 0.0, 0
            for batch in loader:
                fmri_gt = batch["fmri"].to(self.device)
                clip_pool = batch["clip_pool"].to(self.device)
                contexts = [c.to(self.device) for c in batch["contexts"]]
                B = fmri_gt.shape[0]
                pred = self._sample(clip_pool, contexts, B, sidx, noise_scale=0.0)
                mse_sum += masked_mse(pred, fmri_gt, pad_mask).item() * B
                n += B
                preds_all.append(pred.reshape(B, -1)[:, pad_mask].cpu())
                gts_all.append(fmri_gt.reshape(B, -1)[:, pad_mask].cpu())
                profile_rs.append(
                    pearson_corr_per_sample(pred, fmri_gt, pad_mask).cpu()
                )
            preds_cat = torch.cat(preds_all)
            gts_cat = torch.cat(gts_all)
            voxel_r = voxel_pearson(preds_cat, gts_cat).mean().item()
            profile_r = torch.cat(profile_rs).mean().item()
            per_subj[self.subjects[sidx]] = {
                "voxel_r": voxel_r, "profile_r": profile_r, "mse": mse_sum / n,
            }
            self.logger.info(
                "  [Val ep=%d subj=%d] voxel_r=%.4f profile_r=%.4f mse=%.5f n=%d",
                epoch + 1, self.subjects[sidx], voxel_r, profile_r, mse_sum / n, n,
            )
        mean_profile_r = float(np.mean([v["profile_r"] for v in per_subj.values()]))
        mean_voxel_r = float(np.mean([v["voxel_r"] for v in per_subj.values()]))
        self.logger.info(
            "  [Val ep=%d MEAN] voxel_r=%.4f profile_r=%.4f",
            epoch + 1, mean_voxel_r, mean_profile_r,
        )
        self.wrapper.train()
        return {"profile_r": mean_profile_r, "voxel_r": mean_voxel_r,
                "per_subj": per_subj}

    # ──────────────────────────────────────────────────────────────────
    # Main training loop
    # ──────────────────────────────────────────────────────────────────

    def train(self) -> None:
        history_path = os.path.join(self.exp_dir, "history.csv")
        history_exists = os.path.exists(history_path)
        history_file = open(history_path, "a", newline="")
        history_writer = csv.writer(history_file)
        if not history_exists:
            history_writer.writerow(
                ["epoch", "step", "train_loss", "val_voxel_r", "val_profile_r", "lr"]
            )
            history_file.flush()

        running_loss = 0.0
        epoch_loss = 0.0
        epoch_steps = 0
        log_steps = 0
        accum_counter = 0
        wall_start = time()

        self.logger.info(
            "Starting multi-subject training for %d epochs, grad_accum=%d ...",
            self.epochs, self.grad_accum,
        )

        for epoch in range(self.start_epoch, self.epochs):
            self.wrapper.train()
            self.batch_sampler.set_epoch(epoch)
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}",
                        dynamic_ncols=True)
            step_loss = 0.0
            for batch in pbar:
                sidx = int(batch["subject_id"][0])
                x1 = batch["fmri"].to(self.device)
                clip_pool = batch["clip_pool"].to(self.device)
                contexts = [c.to(self.device) for c in batch["contexts"]]

                with autocast(**self.autocast_kwargs):
                    loss = self._compute_loss(x1, clip_pool, contexts, sidx)

                (loss / self.grad_accum).backward()
                accum_counter += 1
                micro_loss = loss.item() / self.grad_accum
                running_loss += micro_loss
                step_loss += micro_loss
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}", subj=self.subjects[sidx],
                    lr=f"{self.scheduler.get_last_lr()[0]:.2e}",
                )

                if accum_counter < self.grad_accum:
                    continue

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

                if self.train_steps % self.ckpt_every == 0 and self.train_steps > 0:
                    save_checkpoint(
                        os.path.join(self.ckpt_dir, f"{self.train_steps:07d}.pt"),
                        self.wrapper, self.optimizer, self.scheduler,
                        self.train_steps, epoch, self.best_metric,
                    )
                if self.train_steps % 1000 == 0 and self.train_steps > 0:
                    save_rolling_last(
                        self.ckpt_dir, self.wrapper,
                        self.optimizer, self.scheduler,
                        self.train_steps, epoch, self.best_metric,
                    )

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
                    f"{val['voxel_r']:.6f}", f"{val['profile_r']:.6f}", f"{lr:.2e}",
                ])
                history_file.flush()
                epoch_loss = 0.0
                epoch_steps = 0
                if val["profile_r"] > self.best_metric:
                    self.best_metric = val["profile_r"]
                    save_checkpoint(
                        os.path.join(self.ckpt_dir, "best.pt"),
                        self.wrapper, self.optimizer, self.scheduler,
                        self.train_steps, epoch, self.best_metric,
                    )
                    self.logger.info(
                        "New best MEAN profile_r: %.4f  (voxel_r: %.4f)",
                        self.best_metric, val["voxel_r"],
                    )
            if hit_max:
                break

        save_checkpoint(
            os.path.join(self.ckpt_dir, f"final-{self.train_steps}.pt"),
            self.wrapper, self.optimizer, self.scheduler,
            self.train_steps, self.epochs, self.best_metric,
        )
        history_file.close()
        self.logger.info("Training complete. History: %s", history_path)
