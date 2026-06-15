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
SDXL_BASE_CFG = os.path.join(MINDEYE_SRC, "generative_models", "configs", "inference", "sd_xl_base.yaml")
SDXL_BASE_CKPT = os.path.join(CKPT_DIR, "zavychromaxl_v30.safetensors")
BIGG_TO_L_CKPT = os.path.join(CKPT_DIR, "bigG_to_L_epoch8.pth")
BLURRY_AUTOENC_CKPT = os.path.join(CKPT_DIR, "sd_image_var_autoenc.pth")
OPENCLIP_BIGG_BIN = os.path.join(CKPT_DIR, "sdxl_unclip", "open_clip_pytorch_model.bin")
MINDEYE_NSD = os.path.join(REPO, "NSD", "data", "mindeye_nsd")
COCO_HDF5 = os.path.join(REPO, "NSD", "data", "coco_images_224_float16.hdf5")
EXPDESIGN = os.path.join(REPO, "NSD", "data", "nsddata", "experiments", "nsd", "nsd_expdesign.mat")
VOXELS = {1: 15724, 2: 14278, 5: 13039, 7: 12682}


def mindeye_ckpt(subject):
    # Only the canonical published 40-session per-subject checkpoint. The old
    # top-level mindeye2/last.pth is a DIFFERENT (weaker) model and silently
    # using it for subj01 sandbagged the absolute metrics — never fall back to it.
    c = os.path.join(CKPT_DIR, "mindeye2", "train_logs",
                     f"final_subj0{subject}_pretrained_40sess_24bs", "last.pth")
    if os.path.exists(c):
        return c
    raise FileNotFoundError(
        f"MindEye2 40sess ckpt not found for subj{subject}: {c}. "
        f"Download via hf_hub_download('pscotti/mindeyev2', "
        f"'train_logs/final_subj0{subject}_pretrained_40sess_24bs/last.pth').")


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
    # strict=True only: a partial load silently degrades semantics (renders fine,
    # CLIP/Incep tank). If keys mismatch we want a hard failure, not a fallback.
    model.load_state_dict(state, strict=True)
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


# ── Enhanced-pipeline model loaders (caption + blurry + SDXL-base refiner) ────
class CLIPConverter(nn.Module):
    """Maps MindEye2 bigG prior output (256,1664) -> CLIP-L caption tokens (257,1024)."""
    def __init__(self, clip_seq_dim=256, clip_emb_dim=1664,
                 clip_text_seq_dim=257, clip_text_emb_dim=1024):
        super().__init__()
        self.linear1 = nn.Linear(clip_seq_dim, clip_text_seq_dim)
        self.linear2 = nn.Linear(clip_emb_dim, clip_text_emb_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.linear1(x)
        x = self.linear2(x.permute(0, 2, 1))
        return x


def load_caption_models():
    """GIT caption model + bigG->L converter, kept on CPU to save VRAM."""
    # transformers>=5 dropped this util; modeling_git imports it (only used in an
    # unused prune_heads path), so shim it before importing modeling_git.
    import transformers.pytorch_utils as _ptu
    if not hasattr(_ptu, "find_pruneable_heads_and_indices"):
        def _fphi(heads, n_heads, head_size, already_pruned_heads):
            mask = torch.ones(n_heads, head_size)
            heads = set(heads) - already_pruned_heads
            for h in heads:
                h = h - sum(1 for x in already_pruned_heads if x < h)
                mask[h] = 0
            mask = mask.view(-1).contiguous().eq(1)
            return heads, torch.arange(len(mask))[mask].long()
        _ptu.find_pruneable_heads_and_indices = _fphi
    from transformers import AutoProcessor
    from modeling_git import GitForCausalLMClipEmb
    processor = AutoProcessor.from_pretrained("microsoft/git-large-coco")
    git_model = GitForCausalLMClipEmb.from_pretrained("microsoft/git-large-coco")
    git_model.eval().requires_grad_(False)
    clip_convert = CLIPConverter()
    sd = torch.load(BIGG_TO_L_CKPT, map_location="cpu")["model_state_dict"]
    clip_convert.load_state_dict(sd, strict=True)
    clip_convert.eval().requires_grad_(False)
    return clip_convert, processor, git_model


def load_blurry_autoenc(device):
    from diffusers import AutoencoderKL
    autoenc = AutoencoderKL(
        down_block_types=["DownEncoderBlock2D"] * 4,
        up_block_types=["UpDecoderBlock2D"] * 4,
        block_out_channels=[128, 256, 512, 512], layers_per_block=2, sample_size=256)
    autoenc.load_state_dict(torch.load(BLURRY_AUTOENC_CKPT, map_location="cpu"))
    autoenc.eval().requires_grad_(False).to(device)
    return autoenc


@torch.no_grad()
def gen_caption(prior_out, clip_convert, processor, git_model):
    emb = clip_convert(prior_out.cpu().float())          # (1,257,1024) on CPU
    ids = git_model.generate(pixel_values=emb, max_length=20)
    return processor.batch_decode(ids, skip_special_tokens=True)[0]


@torch.no_grad()
def gen_blurry(blurry_enc, autoenc, device):
    """blurry_enc: (1,4,h,w) latent -> (3,256,256) image in [0,1]."""
    img = (autoenc.decode(blurry_enc.to(device) / 0.18215).sample / 2 + 0.5).clamp(0, 1)
    return transforms.Resize((256, 256))(img)[0].cpu()


NEG_PROMPT = ("painting, extra fingers, mutated hands, poorly drawn hands, poorly drawn "
              "face, deformed, ugly, blurry, bad anatomy, bad proportions, extra limbs, "
              "cloned face, skinny, glitchy, double torso, extra arms, extra hands, "
              "mangled fingers, missing lips, ugly face, distorted face, extra legs, anime")


def _sdxl_batch(txt, device):
    return {"txt": txt,
            "original_size_as_tuple": torch.ones(1, 2, device=device) * 768,
            "crop_coords_top_left": torch.zeros(1, 2, device=device),
            "target_size_as_tuple": torch.ones(1, 2, device=device) * 1024}


def load_sdxl_base(device):
    """SDXL-base (zavychromaxl) img2img refiner. The engine's own conditioner
    (CLIP-L + bigG text) produces exactly the SDXL crossattn+vector we need, so
    we keep it on the GPU and call it per-image (no separate embedders). UNet +
    conditioner are cast to fp16 to fit 24GB; the VAE stays fp32.

    Returns (engine, uc) where uc is the fixed negative-prompt conditioning."""
    import utils as me_utils
    me_utils.device = device
    from omegaconf import OmegaConf
    from sgm.models.diffusion import DiffusionEngine

    # transformers>=4.48 CLIPTextModel defaults to SDPA attention, which trips on
    # the (77,77) causal mask sgm's FrozenCLIPEmbedder passes. Force eager attn.
    import transformers as _tf
    if not getattr(_tf.CLIPTextModel, "_eager_patched", False):
        _orig_fp = _tf.CLIPTextModel.from_pretrained
        def _fp(*a, **k):
            k.setdefault("attn_implementation", "eager")
            return _orig_fp(*a, **k)
        _tf.CLIPTextModel.from_pretrained = staticmethod(_fp)
        _tf.CLIPTextModel._eager_patched = True

    un = OmegaConf.to_container(OmegaConf.load(SDXL_CFG), resolve=True)["model"]["params"]
    sampler_config = un["sampler_config"]
    sampler_config["params"]["num_steps"] = 38

    cfg = OmegaConf.to_container(OmegaConf.load(SDXL_BASE_CFG), resolve=True)["model"]["params"]
    ddc = cfg["first_stage_config"].get("params", {}).get("ddconfig", {})
    if ddc.get("attn_type") == "vanilla-xformers":
        ddc["attn_type"] = "vanilla"
    for emb in cfg["conditioner_config"]["params"]["emb_models"]:
        ep = emb.get("params", {})
        if ep.get("arch") == "ViT-bigG-14":
            ep["version"] = OPENCLIP_BIGG_BIN       # local bigG, avoid 10GB download

    print(f"[sdxl-base] loading {SDXL_BASE_CKPT}")
    engine = DiffusionEngine(
        network_config=cfg["network_config"], denoiser_config=cfg["denoiser_config"],
        first_stage_config=cfg["first_stage_config"], conditioner_config=cfg["conditioner_config"],
        sampler_config=sampler_config, scale_factor=cfg["scale_factor"],
        disable_first_stage_autocast=cfg["disable_first_stage_autocast"],
        ckpt_path=SDXL_BASE_CKPT)
    engine.eval().requires_grad_(False)
    engine.model.half()           # fp16 UNet + text conditioner to fit 24GB;
    engine.conditioner.half()     # VAE (first_stage_model) stays fp32 for stability
    engine.to(device)
    try:
        engine.sampler.guider.scale = 5
    except Exception:
        pass
    with torch.no_grad():
        uc = dict(engine.conditioner(_sdxl_batch(NEG_PROMPT, device)))
    torch.cuda.empty_cache()
    return engine, uc


# ── Decode: fMRI -> reconstructed images ────────────────────────────────────
def _load_png(path):
    from PIL import Image
    return transforms.functional.to_tensor(Image.open(path).convert("RGB"))


@torch.no_grad()
def decode(fmri, gt_images, model, engine, vector_suffix, device, out_dir,
           save_gt_pngs=True, caption_models=None, autoenc=None):
    """Decode (N,n_rep,V) fMRI -> images, saving EACH image as it is produced.

    Per-image artifacts (enable resume + bounded RAM + the enhanced phase):
      out_dir/recons/{i:04d}.png    base reconstruction
      out_dir/clipvox/{i:04d}.npy   retrieval embedding (256,1664)
      out_dir/gt/{i:04d}.png        ground-truth stimulus (if save_gt_pngs)
      out_dir/captions/{i:04d}.txt  predicted caption (if caption_models)
      out_dir/blurry/{i:04d}.png    low-level blurry recon (if autoenc)
    An index is "done" only when all required artifacts exist, so resume also
    fills in caption/blurry. Returns (recons, clipvox) reloaded from disk.
    """
    from utils import unclip_recon
    from torchvision.utils import save_image
    resize = transforms.Resize((256, 256))
    recons_dir = os.path.join(out_dir, "recons"); os.makedirs(recons_dir, exist_ok=True)
    cv_dir = os.path.join(out_dir, "clipvox"); os.makedirs(cv_dir, exist_ok=True)
    gt_dir = os.path.join(out_dir, "gt")
    cap_dir = os.path.join(out_dir, "captions")
    blur_dir = os.path.join(out_dir, "blurry")
    if save_gt_pngs:
        os.makedirs(gt_dir, exist_ok=True)
    if caption_models is not None:
        os.makedirs(cap_dir, exist_ok=True)
    if autoenc is not None:
        os.makedirs(blur_dir, exist_ok=True)
    N = fmri.shape[0]
    n_skip = 0
    with torch.cuda.amp.autocast(dtype=torch.float16):
        for i in range(N):
            rpath = os.path.join(recons_dir, f"{i:04d}.png")
            cpath = os.path.join(cv_dir, f"{i:04d}.npy")
            cap_path = os.path.join(cap_dir, f"{i:04d}.txt")
            blur_path = os.path.join(blur_dir, f"{i:04d}.png")
            done = os.path.exists(rpath) and os.path.exists(cpath)
            if caption_models is not None:
                done = done and os.path.exists(cap_path)
            if autoenc is not None:
                done = done and os.path.exists(blur_path)
            if done:
                n_skip += 1
                continue
            voxel = fmri[i:i + 1].to(device).float()       # (1, n_rep, V)
            nrep = voxel.shape[1]
            backbone = clipv = blurry = None
            for rep in range(nrep):
                vr = model.ridge(voxel[:, [rep]], 0)
                bb, cv, b = model.backbone(vr)
                backbone = bb if backbone is None else backbone + bb
                clipv = cv if clipv is None else clipv + cv
                be = b[0] if isinstance(b, (tuple, list)) else b
                blurry = be if blurry is None else blurry + be
            backbone = backbone / nrep
            clipv = clipv / nrep
            prior_out = model.diffusion_prior.p_sample_loop(
                backbone.shape, text_cond=dict(text_embed=backbone),
                cond_scale=1.0, timesteps=20)
            img = resize(unclip_recon(prior_out[[0]], engine, vector_suffix,
                                      num_samples=1)).float().cpu()[0].clamp(0, 1)
            np.save(cpath + ".tmp.npy", clipv.cpu().numpy()[0])
            os.replace(cpath + ".tmp.npy", cpath)
            save_image(img, rpath)
            if save_gt_pngs:
                save_image(gt_images[i].float().clamp(0, 1),
                           os.path.join(gt_dir, f"{i:04d}.png"))
            if autoenc is not None:
                save_image(gen_blurry(blurry / nrep, autoenc, device), blur_path)
            if caption_models is not None:
                cap = gen_caption(prior_out, *caption_models)
                with open(cap_path + ".tmp", "w") as f:
                    f.write(cap)
                os.replace(cap_path + ".tmp", cap_path)
            if (i + 1) % 25 == 0:
                print(f"  decoded {i + 1}/{N}", flush=True)
    if n_skip:
        print(f"  [resume] skipped {n_skip}/{N} already-decoded images", flush=True)
    recons = torch.stack([_load_png(os.path.join(recons_dir, f"{i:04d}.png"))
                          for i in range(N)])
    clipvox = torch.from_numpy(np.stack([np.load(os.path.join(cv_dir, f"{i:04d}.npy"))
                                         for i in range(N)]))
    return recons, clipvox


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


def _report_metrics(tag, kind, recons, gt_images, device, mpath):
    metrics = compute_image_metrics(gt_images, recons, device)
    print("=" * 48)
    print(f"SEMANTIC METRICS  ({tag} {kind}, n={len(recons)})")
    print("=" * 48)
    for k in ["PixCorr", "SSIM", "Alex_2", "Alex_5", "Incep", "CLIP", "Eff", "SwAV"]:
        print(f"  {k:<10} {metrics[k]:.4f}")
    with open(mpath, "w") as f:   # written last = the (tag,kind)-complete marker
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)
    print(f"[saved] {mpath}", flush=True)
    return metrics


def base_decode_and_metrics(tag, fmri, gt_images, model, engine, vector_suffix,
                            device, out_dir, save_pngs, caption_models=None, autoenc=None):
    """Phase A: base reconstruction (+ caption/blurry for the enhanced phase),
    incremental save + resume, then BASE metrics -> {tag}_base_metrics.json."""
    os.makedirs(out_dir, exist_ok=True)
    mpath = os.path.join(out_dir, f"{tag}_base_metrics.json")
    if os.path.exists(mpath) and os.path.isdir(os.path.join(out_dir, "recons")):
        print(f"[{tag}] base already complete; skipping decode", flush=True)
        return json.load(open(mpath))
    print(f"\n[{tag}] base: fmri {tuple(fmri.shape)}  gt {tuple(gt_images.shape)}", flush=True)
    recons, clip_voxels = decode(fmri, gt_images, model, engine, vector_suffix,
                                 device, out_dir, save_gt_pngs=True,
                                 caption_models=caption_models, autoenc=autoenc)
    torch.save(clip_voxels, os.path.join(out_dir, f"{tag}_clipvoxels.pt"))
    torch.save(recons, os.path.join(out_dir, f"{tag}_recons.pt"))
    return _report_metrics(tag, "base", recons, gt_images, device, mpath)


@torch.no_grad()
def enhance_and_metrics(tag, sdxl, device, out_dir, img2img_timepoint=13):
    """Phase B: SDXL-base img2img refine of the base recons using the predicted
    caption (per-image save + resume), then ENHANCED metrics on the
    enhanced*0.75 + blurry*0.25 blend -> {tag}_enhanced_metrics.json."""
    from sgm.util import append_dims
    from torchvision.utils import save_image
    engine, uc0 = sdxl
    mpath = os.path.join(out_dir, f"{tag}_enhanced_metrics.json")
    if os.path.exists(mpath):
        print(f"[{tag}] enhanced already complete; skipping", flush=True)
        return json.load(open(mpath))

    recons_dir = os.path.join(out_dir, "recons")
    enh_dir = os.path.join(out_dir, "enhanced"); os.makedirs(enh_dir, exist_ok=True)
    cap_dir = os.path.join(out_dir, "captions")
    blur_dir = os.path.join(out_dir, "blurry")
    gt_dir = os.path.join(out_dir, "gt")
    N = len([f for f in os.listdir(recons_dir) if f.endswith(".png")])
    resize768 = transforms.Resize((768, 768))
    resize256 = transforms.Resize((256, 256))
    print(f"\n[{tag}] enhanced: refining {N} base recons", flush=True)

    def denoiser(x, sigma, cc):
        return engine.denoiser(engine.model, x, sigma, cc)

    n_skip = 0
    with torch.cuda.amp.autocast(dtype=torch.float16), engine.ema_scope():
        for i in range(N):
            epath = os.path.join(enh_dir, f"{i:04d}.png")
            if os.path.exists(epath):
                n_skip += 1
                continue
            image = resize768(_load_png(os.path.join(recons_dir, f"{i:04d}.png")))[None].to(device)
            prompt = open(os.path.join(cap_dir, f"{i:04d}.txt")).read()
            z = engine.encode_first_stage(image * 2 - 1)
            # engine.conditioner produces the SDXL crossattn+vector for this prompt
            c = dict(engine.conditioner(_sdxl_batch(prompt, device)))
            uc = {k: v.clone() for k, v in uc0.items()}

            engine.sampler.num_steps = 25
            sigmas = engine.sampler.discretization(engine.sampler.num_steps).to(device)
            init_z = (z + torch.randn_like(z) * append_dims(sigmas[-img2img_timepoint], z.ndim)
                      ) / torch.sqrt(1.0 + sigmas[0] ** 2.0)
            sigmas = sigmas[-img2img_timepoint:]
            engine.sampler.num_steps = sigmas.shape[-1] - 1
            noised_z, _, _, _, c, uc = engine.sampler.prepare_sampling_loop(
                init_z, cond=c, uc=uc, num_steps=engine.sampler.num_steps)
            for t in range(engine.sampler.num_steps):
                # sigmas[t] would be 0-D; the CFG guider needs a 1-D sigma (it does
                # torch.cat([s]*2)). Slice to keep shape (1,) as in MindEye2's loop.
                noised_z = engine.sampler.sampler_step(
                    sigmas[t:t + 1], sigmas[t + 1:t + 2], denoiser, noised_z,
                    cond=c, uc=uc, gamma=0)
            x = engine.decode_first_stage(noised_z)
            samp = resize256(torch.clamp((x + 1.0) / 2.0, 0.0, 1.0))[0].cpu()
            save_image(samp, epath)
            if (i + 1) % 25 == 0:
                print(f"  enhanced {i + 1}/{N}", flush=True)
    if n_skip:
        print(f"  [resume] skipped {n_skip}/{N} already-enhanced", flush=True)

    enhanced = torch.stack([_load_png(os.path.join(enh_dir, f"{i:04d}.png")) for i in range(N)])
    blurry = torch.stack([_load_png(os.path.join(blur_dir, f"{i:04d}.png")) for i in range(N)])
    gt = torch.stack([_load_png(os.path.join(gt_dir, f"{i:04d}.png")) for i in range(N)])
    blended = (enhanced * 0.75 + blurry * 0.25).clamp(0, 1)   # MindEye2 final blend
    torch.save(enhanced, os.path.join(out_dir, f"{tag}_enhanced_recons.pt"))
    return _report_metrics(tag, "enhanced", blended, gt, device, mpath)


def build_job(kind, subject, synth_npz=None, pred_key="preds"):
    if kind == "gt":
        return assemble_gt_sanity(subject)
    return assemble_stimflow(subject, synth_npz, pred_key)


def _tag_list(args, S):
    tags = []
    if args.do_gt:
        tags.append(("gt", None))
    for k, tag in (("01", "T1"), ("05", "T5")):
        npz = os.path.join(args.synth_dir, f"sub{S}", f"avg_k{k}.npz")
        if os.path.exists(npz):
            tags.append((tag, npz))
        else:
            print(f"[warn] missing {npz}, skipping {tag}")
    return tags


def main():
    import gc
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
    ap.add_argument("--enhanced", action="store_true",
                    help="Also run the MindEye2 enhanced (SDXL-base img2img) phase.")
    ap.add_argument("--save_pngs", action="store_true")
    ap.add_argument("--max_images", type=int, default=-1)
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()
    device = args.device
    S = args.subject

    if args.batch:
        tags = _tag_list(args, S)
        out_dir = lambda tag: os.path.join(args.out_root, f"sub{S}_{tag}")

        def base_done(tag):
            d = out_dir(tag)
            ok = (os.path.exists(os.path.join(d, f"{tag}_base_metrics.json"))
                  and os.path.isdir(os.path.join(d, "recons")))
            if args.enhanced:   # enhanced needs captions + blurry from phase A
                ok = ok and os.path.isdir(os.path.join(d, "captions")) \
                    and os.path.isdir(os.path.join(d, "blurry"))
            return ok

        # ── Phase A: base recon (+ caption/blurry if enhanced) ──
        # Only load the heavy decoders if some tag still needs base decoding.
        need = [(tag, npz) for tag, npz in tags if not base_done(tag)]
        if need:
            model = load_mindeye(S, device)
            engine, vector_suffix = load_unclip(device)
            caption_models = load_caption_models() if args.enhanced else None
            autoenc = load_blurry_autoenc(device) if args.enhanced else None
            for tag, npz in need:
                fmri, gt = build_job("gt" if tag == "gt" else "stimflow", S, npz, args.pred_key)
                if args.max_images > 0:
                    fmri, gt = fmri[:args.max_images], gt[:args.max_images]
                base_decode_and_metrics(tag, fmri, gt, model, engine, vector_suffix,
                                        device, out_dir(tag), args.save_pngs,
                                        caption_models, autoenc)
                del fmri, gt
            del model, engine, caption_models, autoenc
            gc.collect(); torch.cuda.empty_cache()
        else:
            print("[phase A] all base tags already complete; skipping decoders")

        # ── Phase B: enhanced (separate model load to fit VRAM) ──
        # Fail-safe: any error here must NOT lose the base results above.
        if args.enhanced:
            try:
                sdxl = load_sdxl_base(device)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"[enhanced] DISABLED — SDXL-base failed to load in this env "
                      f"({type(e).__name__}: {str(e)[:120]}). Base results are intact.",
                      flush=True)
                sdxl = None
            if sdxl is not None:
                for tag, _ in tags:
                    try:
                        enhance_and_metrics(tag, sdxl, device, out_dir(tag))
                    except Exception as e:
                        import traceback; traceback.print_exc()
                        print(f"[enhanced][{tag}] FAILED ({type(e).__name__}: {str(e)[:120]}); "
                              f"skipping. Base results for {tag} are intact.", flush=True)
                del sdxl
                gc.collect(); torch.cuda.empty_cache()
        print(f"\n[done] subject {S} batch complete.")
        return

    # ── Single run (base only; use --batch for enhanced) ──
    assert args.mode and args.out_dir, "single mode needs --mode and --out_dir"
    kind = "gt" if args.mode == "gt_sanity" else "stimflow"
    fmri, gt_images = build_job(kind, S, args.synth_npz, args.pred_key)
    tag = "gt" if kind == "gt" else "stimflow"
    if args.max_images > 0:
        fmri, gt_images = fmri[:args.max_images], gt_images[:args.max_images]
    model = load_mindeye(S, device)
    engine, vector_suffix = load_unclip(device)
    base_decode_and_metrics(tag, fmri, gt_images, model, engine, vector_suffix,
                            device, args.out_dir, args.save_pngs)


if __name__ == "__main__":
    main()
