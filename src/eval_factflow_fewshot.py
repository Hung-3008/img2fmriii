"""
eval_factflow_fewshot.py
========================
Few-shot cross-subject evaluation for the shared-trunk FactFlow model.

Given a multi-subject checkpoint trained on {1, 2, 5}, adapt to a HELD-OUT
subject (default 7) by:

  1. Building the model with one extra subject slot (the held-out subject at the
     last contiguous index).
  2. Loading the trunk + trained adapters from the checkpoint (strict=False, so
     the new adapter starts randomly initialised).
  3. Freezing EVERYTHING except the held-out subject's input patch-embed and
     output readout.
  4. Fitting only that adapter on K training images of the held-out subject.
  5. Evaluating on the held-out subject's test set in its NATIVE voxel space
     (profile_r / voxel_r), directly comparable to published baselines.

Sweeps K (e.g. 10/50/200/all) to produce the few-shot curve.

Usage::

    python src/eval_factflow_fewshot.py \
        --config src/configs/factflow/multisubject/factflow_ms_sub125.yaml \
        --ckpt exps/factflow_ms_sub125/checkpoints/best.pt \
        --held_out 7 --k_list 10 50 200 -1 --adapt_steps 300
"""

import argparse
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import autocast
from torch.utils.data import DataLoader, Subset

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from data.factflow_fmri_dataset import FactFlowfMRIDataset
from model.factflow_factory import build_models, build_transport, build_sampler
from utils.fmri_utils import create_pad_mask, get_latent_size
from utils.metrics import masked_mse, pearson_corr_per_sample, voxel_pearson


def _ds_kwargs(data_cfg, subject, n_voxels, pad_to, avg_reps):
    return dict(
        data_dir=data_cfg["data_dir"],
        subject=subject,
        fmri_mode=data_cfg["fmri_mode"],
        clip_feature=data_cfg["clip_feature"],
        n_voxels=n_voxels,
        pad_to=pad_to,
        fmri_channels=data_cfg.get("fmri_channels", 1),
        fmri_spatial=data_cfg.get("fmri_spatial", None),
        avg_reps=avg_reps,
        dino_feature=data_cfg.get("dino_feature", None),
        roi_order=bool(data_cfg.get("roi_order", False)),
        context_features=data_cfg.get("context_features", None),
        subdirs=data_cfg.get("subdirs", None),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Few-shot cross-subject FactFlow eval")
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--held_out", type=int, default=7)
    ap.add_argument("--k_list", type=int, nargs="+", default=[10, 50, 200, -1],
                    help="Number of adaptation images (-1 = all train images)")
    ap.add_argument("--adapt_steps", type=int, default=300,
                    help="Minimum number of adapter updates (floor for small K)")
    ap.add_argument("--adapt_epochs", type=int, default=40,
                    help="Passes over the K images; actual updates = "
                         "max(adapt_steps, adapt_epochs * n_batches). Ensures the "
                         "budget scales with K so large K is not undertrained.")
    ap.add_argument("--no_warm_start", action="store_true",
                    help="Disable warm-starting the new adapter from the mean of "
                         "the trained subjects' adapters (default: warm-start ON)")
    ap.add_argument("--adapt_lr", type=float, default=1e-3)
    ap.add_argument("--adapt_bs", type=int, default=32)
    ap.add_argument("--max_test_images", type=int, default=-1,
                    help="Cap test images for a quick run (-1 = all)")
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = OmegaConf.load(args.config)
    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
    pad_to = int(data_cfg["pad_to"])
    n_voxels_map = {int(k): int(v) for k, v in data_cfg["n_voxels_map"].items()}
    train_subjects = list(data_cfg["subjects"])
    held = args.held_out
    held_idx = len(train_subjects)            # new adapter goes at the last slot
    n_voxels = n_voxels_map[held]

    # ── Build model with +1 subject slot, load trunk + trained adapters ──
    cfg.stage_2.params.n_subjects = len(train_subjects) + 1
    cfg.stage_2.params.seq_len = pad_to
    # context dims from the held-out subject's train set (same features as train).
    probe = FactFlowfMRIDataset(mode="train",
                                **_ds_kwargs(data_cfg, held, n_voxels, pad_to, True))
    cfg.stage_2.params.context_dims = list(probe.context_dims)

    wrapper = build_models(cfg, device)
    ckpt = torch.load(args.ckpt, map_location=device)
    missing, unexpected = wrapper.load_state_dict(ckpt["model"], strict=False)
    new_keys = [k for k in missing if f".{held_idx}." in k]
    print(f"[load] strict=False — missing={len(missing)} (new adapter={len(new_keys)}) "
          f"unexpected={len(unexpected)}")

    # ── Freeze everything except the held-out subject's adapter ──
    trainable = []
    for name, p in wrapper.named_parameters():
        is_new = (f"x_embedders.{held_idx}." in name) or (f"final_layers.{held_idx}." in name)
        p.requires_grad = is_new
        if is_new:
            trainable.append(p)
    n_train_params = sum(p.numel() for p in trainable)
    print(f"[freeze] trainable adapter params: {n_train_params/1e6:.3f}M "
          f"(held_out subj={held}, idx={held_idx})")

    # ── Transport / sampler / geometry ──
    latent_size = get_latent_size({**data_cfg, "pad_to": pad_to})
    transport = build_transport(cfg, latent_size)
    sample_fn = build_sampler(transport, cfg.sampler)
    pad_mask = create_pad_mask(n_voxels, pad_to, device)
    use_cross_attn = bool(cfg.stage_2.get("params", {}).get("use_cross_attn", False))
    eval_noise_scale = 0.0   # deterministic ceiling prediction
    use_bf16 = cfg.training.get("precision", "fp32") == "bf16"
    ac_kwargs = dict(device_type=device.split(":")[0], dtype=torch.bfloat16, enabled=use_bf16)

    # ── Datasets (held-out subject) ──
    train_ds = FactFlowfMRIDataset(mode="train",
                                   **_ds_kwargs(data_cfg, held, n_voxels, pad_to, True))
    test_ds = FactFlowfMRIDataset(mode="test",
                                  **_ds_kwargs(data_cfg, held, n_voxels, pad_to, True))
    if args.max_test_images > 0:
        test_ds = Subset(test_ds, list(range(min(args.max_test_images, len(test_ds)))))

    @torch.no_grad()
    def _warm_start_adapter():
        """Init the held-out adapter from the mean of the trained adapters.

        Trained adapters live at contiguous indices [0, held_idx). Averaging
        their input patch-embed and output readout gives a sensible starting
        point (vs random / zero), which helps especially at small K.
        """
        n_train = held_idx
        for mod_list in (wrapper.dit.x_embedders, wrapper.dit.final_layers):
            new_params = dict(mod_list[held_idx].named_parameters())
            trained = [dict(mod_list[j].named_parameters()) for j in range(n_train)]
            for pname, p in new_params.items():
                p.copy_(torch.stack([t[pname] for t in trained]).mean(0))

    def adapt(k: int):
        """Reset the new adapter, fit it on K images, return nothing (in place)."""
        # Re-initialise the new adapter for a clean run per K.
        if args.no_warm_start:
            wrapper.dit._x_embedder(held_idx).apply(_xavier_reset)
            _zero_final(wrapper.dit._final_layer(held_idx))
        else:
            _warm_start_adapter()
        idx = list(range(len(train_ds)))
        rng = np.random.RandomState(args.seed)
        rng.shuffle(idx)
        if k > 0:
            idx = idx[:k]
        sub = Subset(train_ds, idx)
        loader = DataLoader(sub, batch_size=min(args.adapt_bs, len(sub)),
                            shuffle=True, num_workers=0, drop_last=False)
        opt = torch.optim.AdamW([p for p in wrapper.parameters() if p.requires_grad],
                                lr=args.adapt_lr, betas=(0.9, 0.95), weight_decay=0.0)
        wrapper.train()
        # Budget scales with K: enough passes so large K is not undertrained,
        # with a floor so tiny K still gets enough updates.
        total_updates = max(args.adapt_steps, args.adapt_epochs * len(loader))
        step = 0
        while step < total_updates:
            for batch in loader:
                x1 = batch["fmri"].to(device)
                clip_pool = batch["clip_pool"].to(device)
                contexts = [c.to(device) for c in batch["contexts"]] if use_cross_attn else None
                x0 = torch.randn_like(x1)
                t = transport.sample_timestep(x1)
                t, xt, ut = transport.path_sampler.plan(t, x0, x1)
                with autocast(**ac_kwargs):
                    v = wrapper.predict_velocity(x=xt, t=t, y=clip_pool,
                                                 contexts=contexts, subject_id=held_idx)
                    loss = masked_mse(v, ut, pad_mask)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                step += 1
                if step >= total_updates:
                    break

    @torch.no_grad()
    def evaluate() -> dict:
        wrapper.eval()
        loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
        preds_all, gts_all, profile_rs = [], [], []
        for batch in loader:
            fmri_gt = batch["fmri"].to(device)
            clip_pool = batch["clip_pool"].to(device)
            contexts = [c.to(device) for c in batch["contexts"]] if use_cross_attn else None
            B = fmri_gt.shape[0]
            x0 = eval_noise_scale * torch.randn(B, *latent_size, device=device)
            with autocast(**ac_kwargs):
                traj = sample_fn(x0, wrapper.dit.forward, y=clip_pool,
                                 contexts=contexts, subject_id=held_idx)
            pred = traj[-1].float()
            preds_all.append(pred.reshape(B, -1)[:, pad_mask].cpu())
            gts_all.append(fmri_gt.reshape(B, -1)[:, pad_mask].cpu())
            profile_rs.append(pearson_corr_per_sample(pred, fmri_gt, pad_mask).cpu())
        voxel_r = voxel_pearson(torch.cat(preds_all), torch.cat(gts_all)).mean().item()
        profile_r = torch.cat(profile_rs).mean().item()
        return {"voxel_r": voxel_r, "profile_r": profile_r}

    print(f"\n=== Few-shot adaptation to subject {held} "
          f"(trunk trained on {train_subjects}) ===")
    print(f"{'K':>8}  {'voxel_r':>8}  {'profile_r':>9}")
    for k in args.k_list:
        adapt(k)
        m = evaluate()
        ktxt = "all" if k < 0 else str(k)
        print(f"{ktxt:>8}  {m['voxel_r']:>8.4f}  {m['profile_r']:>9.4f}")


def _xavier_reset(module):
    if isinstance(module, torch.nn.Conv1d):
        w = module.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        if module.bias is not None:
            torch.nn.init.constant_(module.bias, 0)
    elif isinstance(module, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            torch.nn.init.constant_(module.bias, 0)


def _zero_final(final_layer):
    """Match DiT init: zero AdaLN + zero output linear for a stable start."""
    torch.nn.init.constant_(final_layer.adaLN_modulation[-1].weight, 0)
    torch.nn.init.constant_(final_layer.adaLN_modulation[-1].bias, 0)
    torch.nn.init.constant_(final_layer.linear.weight, 0)
    torch.nn.init.constant_(final_layer.linear.bias, 0)


if __name__ == "__main__":
    main()
