"""
factflow_core.py
================
Shared primitives for the FactFlow fMRI trainers and evaluator (no-source flow
matching). Everything that used to be copy-pasted across the single-subject,
multi-subject, few-shot and evaluation code lives here exactly once:

  * ``build_ds_kwargs``    — dataset constructor kwargs from a ``data`` config.
  * ``flow_matching_loss`` — the velocity-MSE training step.
  * ``ode_sample``         — one ODE inference pass from scaled Gaussian noise.
  * ``evaluate_metrics``   — the rep-averaged metric loop (voxel_r / profile_r /
                             mse / cosine), with K-sample averaging.
  * ``setup_roi_routing``  — patch-level ROI bucket assignment (single + multi).
  * ``build_voxel_weight`` — per-voxel SNR (noise-ceiling) loss weight.
  * adapter helpers        — warm-start / freeze for cross-subject few-shot.
  * ``FactFlowBase``       — common device/seed/transport/checkpoint setup.
  * ``TrainerLoopMixin``   — the grad-accum epoch loop + history/checkpointing.

Single- vs multi-subject differ only by the per-subject ``subject_id``, so the
stateless helpers all take it as an optional argument (``None`` / ``0`` for the
single-subject DiT).
"""

from __future__ import annotations

import csv
import os
from argparse import Namespace
from time import time
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.factflow_factory import build_transport, build_sampler
from utils.checkpoint import (
    find_last_checkpoint,
    load_checkpoint,
    save_checkpoint,
    save_rolling_last,
)
from utils.metrics import (
    compute_voxel_reliability,
    masked_mse,
    pearson_corr_per_sample,
    voxel_pearson,
)


# ════════════════════════════════════════════════════════════════════════════
# Dataset kwargs
# ════════════════════════════════════════════════════════════════════════════


def build_ds_kwargs(
    data_cfg: Dict,
    *,
    pad_to: int,
    roi_order: bool,
    subject: Optional[int] = None,
    n_voxels: Optional[int] = None,
    n_voxels_map: Optional[Dict[int, int]] = None,
    avg_reps: Optional[bool] = None,
) -> dict:
    """Build :class:`FactFlowfMRIDataset` (or ``build_subject_datasets``) kwargs.

    Pass ``subject`` + ``n_voxels`` for the single-subject path; pass
    ``n_voxels_map`` (and omit ``subject``/``n_voxels``) for the multi-subject
    ``build_subject_datasets`` path, which injects ``avg_reps`` itself — leave
    ``avg_reps=None`` there so it is not double-supplied.
    """
    dc = data_cfg
    kw = dict(
        data_dir=dc["data_dir"],
        fmri_mode=dc["fmri_mode"],
        clip_feature=dc["clip_feature"],
        pad_to=pad_to,
        fmri_channels=dc.get("fmri_channels", 1),
        fmri_spatial=dc.get("fmri_spatial", None),
        dino_feature=dc.get("dino_feature", None),
        roi_order=roi_order,
        context_features=dc.get("context_features", None),
        subdirs=dc.get("subdirs", None),
    )
    if subject is not None:
        kw["subject"] = subject
    if n_voxels is not None:
        kw["n_voxels"] = n_voxels
    if n_voxels_map is not None:
        kw["n_voxels_map"] = n_voxels_map
    if avg_reps is not None:
        kw["avg_reps"] = avg_reps
    return kw


# ════════════════════════════════════════════════════════════════════════════
# Flow-matching step + inference
# ════════════════════════════════════════════════════════════════════════════


def flow_matching_loss(
    *,
    transport,
    wrapper: torch.nn.Module,
    x1: torch.Tensor,
    clip_pool: torch.Tensor,
    contexts,
    pad_mask: torch.Tensor,
    subject_id: Optional[int] = None,
    voxel_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """One flow-matching step: Gaussian source → interpolate → velocity MSE.

    ``contexts`` should already be gated by the caller (``None`` when
    cross-attention is disabled).
    """
    x0 = torch.randn_like(x1)
    t = transport.sample_timestep(x1)
    t, xt, ut = transport.path_sampler.plan(t, x0, x1)
    v_pred = wrapper.predict_velocity(
        x=xt, t=t, y=clip_pool, contexts=contexts, subject_id=subject_id,
    )
    return masked_mse(v_pred, ut, pad_mask, weight=voxel_weight)


def ode_sample(
    *,
    sample_fn: Callable,
    wrapper: torch.nn.Module,
    clip_pool: torch.Tensor,
    contexts,
    B: int,
    latent_size: Sequence[int],
    device: str,
    autocast_kwargs: dict,
    noise_scale: float,
    subject_id: Optional[int] = None,
) -> torch.Tensor:
    """ODE-sample fMRI from scaled Gaussian noise. Returns ``(B, *latent)``.

    ``noise_scale=0.0`` gives a deterministic (ceiling) prediction.
    """
    x0 = noise_scale * torch.randn(B, *latent_size, device=device)
    with autocast(**autocast_kwargs):
        traj = sample_fn(
            x0, wrapper.dit.forward, y=clip_pool,
            contexts=contexts, subject_id=subject_id,
        )
    return traj[-1].float()


@torch.no_grad()
def evaluate_metrics(
    *,
    wrapper: torch.nn.Module,
    sample_fn: Callable,
    ds,
    pad_mask: torch.Tensor,
    latent_size: Sequence[int],
    device: str,
    autocast_kwargs: dict,
    use_cross_attn: bool,
    noise_scale: float,
    subject_id: Optional[int] = None,
    n_inf_trials: int = 1,
    batch_size: int = 32,
) -> Dict[str, float]:
    """Score predictions vs (rep-averaged) GT.

    Draws ``n_inf_trials`` stochastic samples and averages them before scoring.
    Returns ``{voxel_r, profile_r, mse, cosine, n}``.
    """
    was_training = wrapper.training
    wrapper.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    preds, gts, profs, mses, coses = [], [], [], [], []
    for batch in loader:
        fmri_gt = batch["fmri"].to(device)
        clip_pool = batch["clip_pool"].to(device)
        contexts = ([c.to(device) for c in batch["contexts"]]
                    if use_cross_attn else None)
        B = fmri_gt.shape[0]
        acc = torch.zeros(B, *latent_size, device=device)
        for _ in range(n_inf_trials):
            acc += ode_sample(
                sample_fn=sample_fn, wrapper=wrapper, clip_pool=clip_pool,
                contexts=contexts, B=B, latent_size=latent_size, device=device,
                autocast_kwargs=autocast_kwargs, noise_scale=noise_scale,
                subject_id=subject_id,
            )
        pred = acc / n_inf_trials
        p = pred.reshape(B, -1)[:, pad_mask].cpu()
        g = fmri_gt.reshape(B, -1)[:, pad_mask].cpu()
        preds.append(p)
        gts.append(g)
        profs.append(pearson_corr_per_sample(pred, fmri_gt, pad_mask).cpu())
        mses.append(((p - g) ** 2).mean(dim=1))
        coses.append(torch.nn.functional.cosine_similarity(p, g, dim=1))
    if was_training:
        wrapper.train()
    return {
        "voxel_r": voxel_pearson(torch.cat(preds), torch.cat(gts)).mean().item(),
        "profile_r": torch.cat(profs).mean().item(),
        "mse": torch.cat(mses).mean().item(),
        "cosine": torch.cat(coses).mean().item(),
        "n": int(sum(g.shape[0] for g in gts)),
    }


# ════════════════════════════════════════════════════════════════════════════
# ROI-stratified feature routing
# ════════════════════════════════════════════════════════════════════════════


def assign_roi_buckets(
    roi_labels: Optional[np.ndarray],
    n_voxels: int,
) -> np.ndarray:
    """Map NSD streams-atlas labels → 3 visual-hierarchy buckets (early/mid/high).

        0 = early visual   (V1/V2/V3)
        1 = mid visual     (V3A/B, hV4, VO, PHC)
        2 = high visual    (LO, FFA, PPA, OPA, … and unlabeled)

    Handles both the coarse "streams" atlas (max label ≤ 7) and the legacy finer
    atlas (labels up to ~34). With no labels, falls back to equal positional
    thirds. ``roi_labels`` is in original (un-sorted) voxel order; permute the
    result with ``sort_idx`` afterwards if voxels were ROI-reordered.
    """
    if roi_labels is not None:
        roi_labels = roi_labels.astype(np.int64)
        bucket = np.full_like(roi_labels, 2)  # default: high
        if int(roi_labels.max()) <= 7:
            # streams atlas: 0=unknown, 1=early, 2-4=mid, 5-7=high
            bucket[roi_labels == 1] = 0
            bucket[(roi_labels >= 2) & (roi_labels <= 4)] = 1
        else:
            # legacy finer atlas
            bucket[(roi_labels >= 1) & (roi_labels <= 6)] = 0
            bucket[(roi_labels >= 7) & (roi_labels <= 15)] = 1
        return bucket
    bucket = np.zeros(n_voxels, dtype=np.int64)
    third = n_voxels // 3
    bucket[third:2 * third] = 1
    bucket[2 * third:] = 2
    return bucket


def setup_roi_routing(
    *,
    dit: torch.nn.Module,
    cfg,
    data_cfg: Dict,
    subjects: Sequence[int],
    datasets: Sequence,
    roi_order: bool,
    pad_to: int,
    logger,
) -> None:
    """Build per-subject patch-level ROI buckets and register them in the DiT.

    No-op unless ``dit.use_roi_routing``. ``subjects[i]`` corresponds to
    ``datasets[i]`` and contiguous ``subject_idx=i``.
    """
    if not getattr(dit, "use_roi_routing", False):
        return
    n_roi_buckets = int(cfg.stage_2.params.get("n_roi_buckets", 3))
    for sidx, (s, ds) in enumerate(zip(subjects, datasets)):
        roi_path = os.path.join(
            data_cfg["data_dir"], f"subj0{s}", f"roi_meta_sub{s}.npz",
        )
        meta = np.load(roi_path) if os.path.exists(roi_path) else {}
        roi_labels = meta["roi_labels"] if "roi_labels" in meta else None
        if roi_labels is None:
            logger.warning(
                "subj%d roi_meta missing 'roi_labels' — using positional thirds "
                "as ROI buckets (less accurate).", s,
            )
        bucket = assign_roi_buckets(roi_labels, ds.n_voxels)
        if roi_order and ds.sort_idx is not None:
            bucket = bucket[ds.sort_idx]
        dit.set_roi_buckets(
            voxel_bucket_ids=bucket, pad_to=pad_to,
            n_roi_buckets=n_roi_buckets, subject_idx=sidx,
        )
        logger.info(
            "ROI routing subj%d (idx=%d): early=%d  mid=%d  high=%d",
            s, sidx, int((bucket == 0).sum()),
            int((bucket == 1).sum()), int((bucket == 2).sum()),
        )


def build_voxel_weight(
    *,
    ds,
    pad_to: int,
    n_voxels: int,
    roi_order: bool,
    device: str,
) -> torch.Tensor:
    """Per-voxel noise-ceiling (SNR) weight over ``pad_to``, mean-normalised to 1.

    Follows the same ROI permutation as the fMRI voxels. Padded positions get 0.
    """
    nc = compute_voxel_reliability(ds.fmri_data, ds.n_reps)  # (V,) original order
    if roi_order and ds.sort_idx is not None:
        nc = nc[ds.sort_idx]
    w = np.zeros(pad_to, dtype=np.float64)
    w[:n_voxels] = nc
    mean_w = w[:n_voxels].mean()
    if mean_w > 0:
        w = w / mean_w
    return torch.tensor(w, dtype=torch.float32, device=device)


# ════════════════════════════════════════════════════════════════════════════
# Cross-subject adapter helpers (few-shot)
# ════════════════════════════════════════════════════════════════════════════


@torch.no_grad()
def warm_start_adapter(dit: torch.nn.Module, held_idx: int) -> None:
    """Init the held-out adapter from the mean of the trained adapters.

    Trained adapters occupy contiguous indices ``[0, held_idx)``. Averaging
    their input patch-embed and output readout gives a far better start than a
    random / zero init, especially with little adaptation data.
    """
    for mod_list in (dit.x_embedders, dit.final_layers):
        new_params = dict(mod_list[held_idx].named_parameters())
        trained = [dict(mod_list[j].named_parameters()) for j in range(held_idx)]
        for pname, p in new_params.items():
            p.copy_(torch.stack([t[pname] for t in trained]).mean(0))


def freeze_to_adapter(wrapper: torch.nn.Module, held_idx: int) -> List[torch.Tensor]:
    """Freeze everything except subject ``held_idx``'s input/output adapter.

    Returns the list of trainable adapter parameters.
    """
    trainable = []
    for name, p in wrapper.named_parameters():
        is_new = (f"x_embedders.{held_idx}." in name
                  or f"final_layers.{held_idx}." in name)
        p.requires_grad = is_new
        if is_new:
            trainable.append(p)
    return trainable


# ════════════════════════════════════════════════════════════════════════════
# Base trainer (common setup) + epoch loop mixin
# ════════════════════════════════════════════════════════════════════════════


class FactFlowBase:
    """Shared config / device / transport / checkpoint setup for trainers.

    Subclasses are responsible for building datasets, the model and the
    optimizer/scheduler (the parts that genuinely differ), then calling
    :meth:`_finalize_setup` once the wrapper exists.
    """

    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.cfg = OmegaConf.load(args.config)

    # ── Device / seed ──────────────────────────────────────────────────
    def _init_device_seed(self, seed: int) -> None:
        self.device = self.args.device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.manual_seed(seed)
        np.random.seed(seed)
        if self.device == "cuda":
            torch.cuda.manual_seed(seed)

    # ── Experiment dir + logger ────────────────────────────────────────
    def _init_exp_dir(self, exp_name: str, logger_name: str, logger) -> None:
        self.exp_dir = os.path.join(self.args.exps_dir, exp_name)
        self.ckpt_dir = os.path.join(self.exp_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.logger = logger
        cfg_path = os.path.join(self.exp_dir, "config.yaml")
        if not os.path.exists(cfg_path):
            OmegaConf.save(self.cfg, cfg_path)

    # ── Transport / sampler / autocast ─────────────────────────────────
    def _init_flow(self, latent_size, train_cfg: Dict) -> None:
        self.latent_size = latent_size
        self.transport = build_transport(self.cfg, self.latent_size)
        self.sample_fn = build_sampler(self.transport, self.cfg.sampler)
        use_bf16 = train_cfg.get("precision", "fp32") == "bf16"
        self.autocast_kwargs = dict(
            device_type=self.device.split(":")[0],
            dtype=torch.bfloat16, enabled=use_bf16,
        )
        self.use_cross_attn = bool(
            self.cfg.stage_2.get("params", {}).get("use_cross_attn", False)
        )
        self.eval_noise_scale = float(self.cfg.get("eval_noise_scale", 1.0))

    # ── Checkpoint resume ──────────────────────────────────────────────
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


class TrainerLoopMixin:
    """The grad-accum training loop shared by single- and multi-subject trainers.

    Requires the host to provide:
      * attributes: ``wrapper``, ``optimizer``, ``scheduler``, ``logger``,
        ``device``, ``autocast_kwargs``, ``exp_dir``, ``ckpt_dir``, ``epochs``,
        ``start_epoch``, ``train_steps``, ``best_metric``, ``grad_accum``,
        ``clip_grad``, ``log_every``, ``ckpt_every``, ``val_every``, ``args``,
        ``train_loader``.
      * methods:
        - ``_loss_for_batch(batch) -> (loss, postfix_dict)``
        - ``_validate(epoch) -> dict`` (must contain ``profile_r`` / ``voxel_r``)
        - ``_on_epoch_start(epoch)`` (optional hook, e.g. sampler.set_epoch)
        - ``_history_header() -> list[str]``
        - ``_history_row(epoch, train_loss, val) -> list``
    """

    def _on_epoch_start(self, epoch: int) -> None:  # overridable hook
        pass

    def train(self) -> None:
        history_path = os.path.join(self.exp_dir, "history.csv")
        history_exists = os.path.exists(history_path)
        history_file = open(history_path, "a", newline="")
        history_writer = csv.writer(history_file)
        if not history_exists:
            history_writer.writerow(self._history_header())
            history_file.flush()

        running_loss = 0.0
        epoch_loss = 0.0
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
            self._on_epoch_start(epoch)
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}",
                        dynamic_ncols=True)
            step_loss = 0.0
            for batch in pbar:
                with autocast(**self.autocast_kwargs):
                    loss, postfix = self._loss_for_batch(batch)

                (loss / self.grad_accum).backward()
                accum_counter += 1
                micro_loss = loss.item() / self.grad_accum
                running_loss += micro_loss
                step_loss += micro_loss
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    lr=f"{self.scheduler.get_last_lr()[0]:.2e}",
                    **(postfix or {}),
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
                if self._maybe_inline_eval():
                    pass

                if self.args.max_steps and self.train_steps >= self.args.max_steps:
                    self.logger.info(
                        "Reached max_steps=%d, stopping.", self.args.max_steps)
                    break

            # ── End-of-epoch validation ──
            is_val_epoch = ((epoch + 1) % self.val_every == 0
                            or (epoch + 1) == self.epochs)
            hit_max = self.args.max_steps and self.train_steps >= self.args.max_steps
            if not hit_max and is_val_epoch:
                self.logger.info("End of epoch %d, running validation...", epoch + 1)
                val = self._validate(epoch)
                train_loss = epoch_loss / max(1, epoch_steps)
                history_writer.writerow(self._history_row(epoch, train_loss, val))
                history_file.flush()
                epoch_loss = 0.0
                epoch_steps = 0
                if val["profile_r"] > self.best_metric:
                    self.best_metric = val["profile_r"]
                    save_checkpoint(
                        os.path.join(self.ckpt_dir, "best.pt"),
                        self.wrapper, self.optimizer, self.scheduler,
                        self.train_steps, epoch, self.best_metric,
                        **self._best_ckpt_extra(val),
                    )
                    self.logger.info(
                        "New best profile_r: %.4f  (voxel_r: %.4f)",
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

    # ── Overridable hooks with sensible defaults ───────────────────────
    def _maybe_inline_eval(self) -> bool:
        return False

    def _best_ckpt_extra(self, val: Dict) -> Dict:
        return {}
