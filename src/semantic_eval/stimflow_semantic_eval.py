"""
stimflow_semantic_eval.py
=========================
Semantic-level evaluation of StimFlow-synthesized fMRI using the *correct*
MindEye2 inference recipe (from this repo's recon_inference.py), i.e. the
MindSimulator/SynBrain protocol done right:

    fMRI voxels (per-session per-voxel z-score = MindEye2 "renorm")
      --> ridge --> backbone --> {clip_voxels, backbone}
      --> diffusion_prior.p_sample_loop(timesteps=20)
      --> unclip_recon (SDXL unCLIP)
      --> reconstructed image
      --> image-similarity metrics vs the ground-truth stimulus.

Key correction vs the earlier SynBrain-based attempt: there is **no** `/2000`
"scale" step and **no** `norm_mean_scale` standardization. We verified that
StimFlow's `fmri_mode=zscore` is the *same space and voxel order* as MindEye2's
renorm betas (per-voxel corr ~0.95, identical ordering, std ratio 1.004), so the
synthesized voxels are fed **directly** into the ridge.

Modes
-----
--mode gt_sanity   : decode MindEye2's real renorm betas (3-rep averaged) for the
                     1000 shared-test images. Validates the whole decoder; should
                     reproduce the published GT upper bound (Incep~0.96/CLIP~0.95).
--mode stimflow    : decode StimFlow's synthesized fMRI from an eval .npz
                     (results/.../sub{S}/avg_k{K}.npz, key 'preds'), fed directly.

Assets (all local):
  MindEye2 ckpt : NSD/checkpoints/mindeye2/[train_logs/final_subj0{S}_pretrained_40sess_24bs/]last.pth
  SDXL unCLIP   : NSD/checkpoints/sdxl_unclip/unclip6_epoch0_step110000.ckpt (+ unclip6.yaml)
  renorm betas  : NSD/data/mindeye_nsd/subject_{S}/betas_all_subj0{S}_fp32_renorm.hdf5
  test behav    : NSD/data/mindeye_nsd/wds/subj0{S}/new_test/0.tar
  COCO images   : NSD/data/coco_images_224_float16.hdf5
  StimFlow stim : NSD/data/nsd/subj0{S}/nsd_test_stim_sub{S}.npy   (my image order)

Example::
    python reproduces/MindEyeV2/src/stimflow_semantic_eval.py \
        --mode gt_sanity --subject 1 --max_images 100 \
        --out_dir results/semantic_eval/gt_sanity_sub1

    python reproduces/MindEyeV2/src/stimflow_semantic_eval.py \
        --mode stimflow --subject 1 \
        --synth_npz results/ms_sub1257_rfr_eval/sub1/avg_k01.npz \
        --out_dir results/semantic_eval/stimflow_sub1_T1
"""

import argparse
import io
import json
import os
import sys
import tarfile

import h5py
import numpy as np
import scipy.io as spio
import torch
import torch.nn as nn
from torchvision import transforms

# ── Paths / imports ─────────────────────────────────────────────────────────
# Location-independent: walk up to the repo root (has reproduces/ and NSD/),
# then point at the vendored MindEyeV2 source so its models/utils/sgm import.
THIS = os.path.dirname(os.path.abspath(__file__))
REPO = THIS
while REPO != os.path.dirname(REPO) and not (
        os.path.isdir(os.path.join(REPO, "reproduces"))
        and os.path.isdir(os.path.join(REPO, "NSD"))):
    REPO = os.path.dirname(REPO)
MINDEYE_SRC = os.path.join(REPO, "reproduces", "MindEyeV2", "src")
for p in (REPO, MINDEYE_SRC, os.path.join(MINDEYE_SRC, "generative_models")):
    if p not in sys.path:
        sys.path.insert(0, p)

CKPT_DIR = os.path.join(REPO, "NSD", "checkpoints")
SDXL_CKPT = os.path.join(CKPT_DIR, "sdxl_unclip", "unclip6_epoch0_step110000.ckpt")
SDXL_CFG = os.path.join(CKPT_DIR, "sdxl_unclip", "unclip6.yaml")
MINDEYE_NSD = os.path.join(REPO, "NSD", "data", "mindeye_nsd")
COCO_HDF5 = os.path.join(REPO, "NSD", "data", "coco_images_224_float16.hdf5")
EXPDESIGN = os.path.join(REPO, "NSD", "data", "nsddata", "experiments", "nsd", "nsd_expdesign.mat")
VOXELS = {1: 15724, 2: 14278, 5: 13039, 7: 12682}


def mindeye_ckpt(subject):
    cands = [os.path.join(CKPT_DIR, "mindeye2", "train_logs",
                          f"final_subj0{subject}_pretrained_40sess_24bs", "last.pth")]
    if subject == 1:
        cands.append(os.path.join(CKPT_DIR, "mindeye2", "last.pth"))
    for c in cands:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"MindEye2 ckpt not found for subj{subject}: {cands}")


# ── Model definitions (mirror recon_inference.py) ───────────────────────────
class MindEyeModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class RidgeRegression(nn.Module):
    def __init__(self, input_sizes, out_features):
        super().__init__()
        self.out_features = out_features
        self.linears = nn.ModuleList([nn.Linear(s, out_features) for s in input_sizes])

    def forward(self, x, subj_idx):
        return self.linears[subj_idx](x[:, 0]).unsqueeze(1)


def load_mindeye(subject, device):
    from models import BrainNetwork, PriorNetwork, BrainDiffusionPrior

    hidden, clip_seq, clip_emb = 4096, 256, 1664
    model = MindEyeModule()
    model.ridge = RidgeRegression([VOXELS[subject]], out_features=hidden)
    model.backbone = BrainNetwork(h=hidden, in_dim=hidden, seq_len=1,
                                  clip_size=clip_emb, out_dim=clip_emb * clip_seq)
    prior_network = PriorNetwork(dim=clip_emb, depth=6, dim_head=52,
                                 heads=clip_emb // 52, causal=False,
                                 num_tokens=clip_seq, learned_query_mode="pos_emb")
    model.diffusion_prior = BrainDiffusionPrior(
        net=prior_network, image_embed_dim=clip_emb,
        condition_on_text_encodings=False, timesteps=100,
        cond_drop_prob=0.2, image_embed_scale=None)
    model.to(device)

    path = mindeye_ckpt(subject)
    print(f"[mindeye2] loading {path}")
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["model_state_dict"]
    try:
        model.load_state_dict(state, strict=True)
    except Exception as e:
        print(f"[mindeye2] strict load failed ({str(e)[:80]}...), retrying strict=False")
        model.load_state_dict(state, strict=False)
    del ckpt
    model.eval().requires_grad_(False)
    return model


def load_unclip(device):
    import utils as me_utils
    me_utils.device = device  # unclip_recon reads this module global
    from omegaconf import OmegaConf
    from sgm.models.diffusion import DiffusionEngine

    cfg = OmegaConf.to_container(OmegaConf.load(SDXL_CFG), resolve=True)
    p = cfg["model"]["params"]
    p["first_stage_config"]["target"] = "sgm.models.autoencoder.AutoencoderKL"
    p["sampler_config"]["params"]["num_steps"] = 38
    # xformers isn't installed on this torch build; use plain attention in the VAE.
    ddc = p["first_stage_config"].get("params", {}).get("ddconfig", {})
    if ddc.get("attn_type") == "vanilla-xformers":
        ddc["attn_type"] = "vanilla"

    # The conditioner's OpenCLIP-bigG embedder points at a relative .bin path;
    # rewrite it to the absolute local checkpoint.
    openclip_bin = os.path.join(CKPT_DIR, "sdxl_unclip", "open_clip_pytorch_model.bin")
    for emb in p["conditioner_config"]["params"]["emb_models"]:
        ep = emb.get("params", {})
        if isinstance(ep.get("version"), str) and ep["version"].endswith("open_clip_pytorch_model.bin"):
            ep["version"] = openclip_bin
    engine = DiffusionEngine(
        network_config=p["network_config"], denoiser_config=p["denoiser_config"],
        first_stage_config=p["first_stage_config"], conditioner_config=p["conditioner_config"],
        sampler_config=p["sampler_config"], scale_factor=p["scale_factor"],
        disable_first_stage_autocast=p["disable_first_stage_autocast"])
    engine.eval().requires_grad_(False)  # stays on CPU for now

    print(f"[sdxl] loading {SDXL_CKPT}")
    sd = torch.load(SDXL_CKPT, map_location="cpu")
    engine.load_state_dict(sd["state_dict"])
    del sd

    # Compute vector_suffix once on CPU (uses the ~2.5B-param bigG conditioner),
    # then keep the conditioner OFF the GPU — unclip_recon never uses it again.
    batch = {"jpg": torch.randn(1, 3, 1, 1),
             "original_size_as_tuple": torch.ones(1, 2) * 768,
             "crop_coords_top_left": torch.zeros(1, 2)}
    with torch.no_grad():
        vector_suffix = engine.conditioner(batch)["vector"].to(device)
    # Move only the modules used during sampling to the GPU.
    engine.model.to(device)
    engine.denoiser.to(device)
    engine.first_stage_model.to(device)
    engine.conditioner.to("cpu")
    torch.cuda.empty_cache()
    return engine, vector_suffix


# ── Decode: fMRI -> reconstructed images ────────────────────────────────────
@torch.no_grad()
def decode(fmri, model, engine, vector_suffix, device):
    """fmri: (N, n_rep, V) torch on CPU. Returns recons (N,3,256,256), clipvox (N,256,1664)."""
    from utils import unclip_recon
    resize = transforms.Resize((256, 256))
    recons, clipvox = [], []
    N = fmri.shape[0]
    with torch.cuda.amp.autocast(dtype=torch.float16):
        for i in range(N):
            voxel = fmri[i:i + 1].to(device).float()       # (1, n_rep, V)
            nrep = voxel.shape[1]
            backbone = clipv = None
            for rep in range(nrep):
                vr = model.ridge(voxel[:, [rep]], 0)
                bb, cv, _ = model.backbone(vr)
                backbone = bb if backbone is None else backbone + bb
                clipv = cv if clipv is None else clipv + cv
            backbone = backbone / nrep
            clipv = clipv / nrep
            clipvox.append(clipv.cpu())
            prior_out = model.diffusion_prior.p_sample_loop(
                backbone.shape, text_cond=dict(text_embed=backbone),
                cond_scale=1.0, timesteps=20)
            img = unclip_recon(prior_out[[0]], engine, vector_suffix, num_samples=1)
            recons.append(resize(img).float().cpu())
            if (i + 1) % 25 == 0:
                print(f"  decoded {i + 1}/{N}", flush=True)
    return torch.cat(recons, 0), torch.cat(clipvox, 0)


# ── Image-similarity metrics (MindEye2/SynBrain protocol) ────────────────────
def _two_way(gt, pd):
    n = len(gt)
    c = np.corrcoef(gt, pd)[:n, n:]
    return float(np.sum(c < np.diag(c), axis=0).mean() / (n - 1))


def compute_image_metrics(all_images, all_recons, device, batch_size=32):
    import scipy as sp
    from skimage.color import rgb2gray
    from skimage.metrics import structural_similarity as ssim
    from torchvision.models import (AlexNet_Weights, EfficientNet_B1_Weights,
                                     Inception_V3_Weights, alexnet,
                                     efficientnet_b1, inception_v3)
    from torchvision.models.feature_extraction import create_feature_extractor
    import clip as openai_clip

    res = {}
    all_images = all_images.detach().float()
    all_recons = all_recons.detach().float()
    big = transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR)
    gt_f = big(all_images).reshape(len(all_images), -1).cpu().numpy()
    pd_f = big(all_recons).reshape(len(all_recons), -1).cpu().numpy()
    res["PixCorr"] = float(np.mean([np.corrcoef(gt_f[i], pd_f[i])[0, 1] for i in range(len(gt_f))]))
    gg = rgb2gray(big(all_images).permute(0, 2, 3, 1).cpu().numpy())
    gp = rgb2gray(big(all_recons).permute(0, 2, 3, 1).cpu().numpy())
    res["SSIM"] = float(np.mean([
        ssim(gg[i], gp[i], gaussian_weights=True, sigma=1.5,
             use_sample_covariance=False, data_range=1.0) for i in range(len(gg))]))

    @torch.no_grad()
    def feats(model, prep, layer, imgs):
        out = []
        for i in range(0, len(imgs), batch_size):
            b = prep(imgs[i:i + batch_size]).to(device)
            f = model(b) if layer is None else model(b)[layer]
            out.append(f.float().flatten(1).detach().cpu().numpy())
        return np.concatenate(out, 0)

    norm = lambda m, s: transforms.Normalize(mean=m, std=s)
    imnet = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    specs = [("Alex_2", "alex", 256, "features.4"), ("Alex_5", "alex", 256, "features.11"),
             ("Incep", "incep", 342, "avgpool"), ("CLIP", "clip", 224, None),
             ("Eff", "eff", 255, "avgpool"), ("SwAV", "swav", 224, "avgpool")]
    for name, kind, size, layer in specs:
        if kind == "alex":
            m = create_feature_extractor(alexnet(weights=AlexNet_Weights.IMAGENET1K_V1),
                                         return_nodes=["features.4", "features.11"]).to(device).eval()
            prep = transforms.Compose([transforms.Resize(size), norm(*imnet)])
        elif kind == "incep":
            m = create_feature_extractor(inception_v3(weights=Inception_V3_Weights.DEFAULT),
                                         return_nodes=["avgpool"]).to(device).eval()
            prep = transforms.Compose([transforms.Resize(size), norm(*imnet)])
        elif kind == "clip":
            m, _ = openai_clip.load("ViT-L/14", device=device)
            m = m.encode_image
            prep = transforms.Compose([transforms.Resize(224),
                                       norm([0.48145466, 0.4578275, 0.40821073],
                                            [0.26862954, 0.26130258, 0.27577711])])
        elif kind == "eff":
            m = create_feature_extractor(efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT),
                                         return_nodes=["avgpool"]).to(device).eval()
            prep = transforms.Compose([transforms.Resize(size), norm(*imnet)])
        elif kind == "swav":
            m = torch.hub.load("facebookresearch/swav:main", "resnet50")
            m = create_feature_extractor(m, return_nodes=["avgpool"]).to(device).eval()
            prep = transforms.Compose([transforms.Resize(size), norm(*imnet)])
        g = feats(m, prep, layer, all_images)
        p = feats(m, prep, layer, all_recons)
        if name in ("Eff", "SwAV"):
            res[name] = float(np.mean([sp.spatial.distance.correlation(g[i], p[i]) for i in range(len(g))]))
        else:
            res[name] = _two_way(g, p)
        del m
        torch.cuda.empty_cache()
    return res


# ── Data assembly ───────────────────────────────────────────────────────────
def my_image_order(subject):
    """nsdId (== COCO 73k index) for each row of nsd_*_fmri_zscore (my pipeline order)."""
    md = spio.loadmat(EXPDESIGN, squeeze_me=True)
    subjectim, mo = md["subjectim"], md["masterordering"]
    seen, order = set(), []
    for idx in range(40 * 750):
        if mo[idx] <= 1000:
            nsd = int(subjectim[subject - 1, mo[idx] - 1] - 1)
            if nsd not in seen:
                seen.add(nsd)
                order.append(nsd)
    return order


def load_coco_images(coco_ids):
    f = h5py.File(COCO_HDF5, "r")
    imgs = f["images"]
    out = np.stack([imgs[c] for c in coco_ids]).astype(np.float32)  # (N,3,224,224)
    if out.max() > 1.5:
        out /= 255.0
    return transforms.Resize((256, 256))(torch.from_numpy(out))


def assemble_gt_sanity(subject):
    """MindEye2 renorm betas, 3-rep averaged per shared-test image. Returns
    fmri (N,3,V), gt_images (N,3,256,256) in matching COCO order."""
    betas = h5py.File(os.path.join(MINDEYE_NSD, f"subject_{subject}",
                                   f"betas_all_subj0{subject}_fp32_renorm.hdf5"), "r")["betas"][:]
    tar = tarfile.open(os.path.join(MINDEYE_NSD, "wds", f"subj0{subject}", "new_test", "0.tar"))
    names = sorted(n for n in tar.getnames() if n.endswith(".behav.npy")
                   and not any(x in n for x in ("future_", "olds_", "past_")))
    coco, gtr = [], []
    for n in names:
        a = np.load(io.BytesIO(tar.extractfile(n).read()))
        coco.append(int(a[0, 0]))
        gtr.append(int(a[0, 5]))
    coco, gtr = np.array(coco), np.array(gtr)
    uniq = np.unique(coco)
    fmri = np.zeros((len(uniq), 3, betas.shape[1]), dtype=np.float32)
    for i, c in enumerate(uniq):
        locs = gtr[coco == c]
        if len(locs) == 1:
            locs = np.repeat(locs, 3)
        elif len(locs) == 2:
            locs = np.repeat(locs, 2)[:3]
        fmri[i] = betas[locs[:3]]
    return torch.from_numpy(fmri), load_coco_images(list(uniq))


def assemble_stimflow(subject, synth_npz, pred_key):
    """StimFlow synth (zscore, my voxel order) fed directly. Returns
    fmri (N,1,V), gt_images (N,3,256,256) in my image order."""
    synth = np.load(synth_npz)[pred_key].astype(np.float32)        # (N,V)
    fmri = torch.from_numpy(synth)[:, None, :]                     # (N,1,V)
    stim = np.load(os.path.join(REPO, "NSD", "data", "nsd", f"subj0{subject}",
                                f"nsd_test_stim_sub{subject}.npy"))[:len(synth)]  # (N,425,425,3) uint8
    gt = torch.from_numpy(stim).float().permute(0, 3, 1, 2) / 255.0
    gt = transforms.Resize((256, 256))(gt)
    return fmri, gt


def save_images_png(tensor, out_dir):
    """Save (N,3,H,W) float[0,1] tensor as zero-padded PNGs."""
    from torchvision.utils import save_image
    os.makedirs(out_dir, exist_ok=True)
    t = tensor.detach().float().clamp(0, 1)
    for i in range(len(t)):
        save_image(t[i], os.path.join(out_dir, f"{i:04d}.png"))


def decode_and_metrics(tag, fmri, gt_images, model, engine, vector_suffix,
                       device, out_dir, save_pngs):
    """Decode one fMRI set, save recons (+PNGs), compute & save metrics."""
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n[{tag}] fmri {tuple(fmri.shape)}  gt_images {tuple(gt_images.shape)}", flush=True)
    recons, clip_voxels = decode(fmri, model, engine, vector_suffix, device)
    torch.save(recons, os.path.join(out_dir, f"{tag}_recons.pt"))
    torch.save(clip_voxels, os.path.join(out_dir, f"{tag}_clipvoxels.pt"))
    if save_pngs:
        save_images_png(recons, os.path.join(out_dir, "recons"))
        save_images_png(gt_images, os.path.join(out_dir, "gt"))

    metrics = compute_image_metrics(gt_images, recons, device)
    print("=" * 48)
    print(f"SEMANTIC METRICS  ({tag}, n={len(recons)})")
    print("=" * 48)
    for k in ["PixCorr", "SSIM", "Alex_2", "Alex_5", "Incep", "CLIP", "Eff", "SwAV"]:
        print(f"  {k:<10} {metrics[k]:.4f}")
    with open(os.path.join(out_dir, f"{tag}_metrics.json"), "w") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)
    print(f"[saved] {out_dir}/{tag}_metrics.json", flush=True)
    return metrics


def build_job(kind, subject, synth_npz=None, pred_key="preds"):
    if kind == "gt":
        fmri, gt = assemble_gt_sanity(subject)
    else:
        fmri, gt = assemble_stimflow(subject, synth_npz, pred_key)
    return fmri, gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", action="store_true",
                    help="Load models once for --subject and run GT(optional)+T1+T5.")
    ap.add_argument("--mode", choices=["gt_sanity", "stimflow"], default=None)
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--synth_npz", type=str, default=None)
    ap.add_argument("--synth_dir", type=str, default="results/ms_sub1257_rfr_eval",
                    help="Batch mode: dir with sub{S}/avg_k01.npz and avg_k05.npz")
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--out_root", type=str, default="results/semantic_eval",
                    help="Batch mode: results written to {out_root}/sub{S}_{tag}/")
    ap.add_argument("--pred_key", type=str, default="preds")
    ap.add_argument("--do_gt", action="store_true", help="Batch mode: also run GT upper bound.")
    ap.add_argument("--save_pngs", action="store_true")
    ap.add_argument("--max_images", type=int, default=-1)
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()
    device = args.device
    S = args.subject

    # ── Batch: one model load per subject, decode GT(optional)+T1+T5 ──
    if args.batch:
        model = load_mindeye(S, device)
        engine, vector_suffix = load_unclip(device)
        jobs = []
        if args.do_gt:
            jobs.append(("gt", *build_job("gt", S)))
        for k, tag in (("01", "T1"), ("05", "T5")):
            npz = os.path.join(args.synth_dir, f"sub{S}", f"avg_k{k}.npz")
            if os.path.exists(npz):
                jobs.append((tag, *build_job("stimflow", S, npz, args.pred_key)))
            else:
                print(f"[warn] missing {npz}, skipping {tag}")
        for tag, fmri, gt in jobs:
            if args.max_images > 0:
                fmri, gt = fmri[:args.max_images], gt[:args.max_images]
            out_dir = os.path.join(args.out_root, f"sub{S}_{tag}")
            decode_and_metrics(tag, fmri, gt, model, engine, vector_suffix,
                               device, out_dir, args.save_pngs)
        print(f"\n[done] subject {S} batch complete.")
        return

    # ── Single run ──
    assert args.mode and args.out_dir, "single mode needs --mode and --out_dir"
    if args.mode == "gt_sanity":
        fmri, gt_images = build_job("gt", S)
        tag = "gt"
    else:
        assert args.synth_npz, "--synth_npz required for --mode stimflow"
        fmri, gt_images = build_job("stimflow", S, args.synth_npz, args.pred_key)
        tag = "stimflow"
    if args.max_images > 0:
        fmri, gt_images = fmri[:args.max_images], gt_images[:args.max_images]
    model = load_mindeye(S, device)
    engine, vector_suffix = load_unclip(device)
    decode_and_metrics(tag, fmri, gt_images, model, engine, vector_suffix,
                       device, args.out_dir, args.save_pngs)


if __name__ == "__main__":
    main()
