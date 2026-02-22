import os
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.train_stage2 import FmriFeatureDataset
from src.model.fmri_mlp_vae import FmriMLPVAE, FmriMLPVAEConfig
from src.model.latent_flow_mlp import LatentFlowMLP, FlowMLPConfig
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher, ExactOptimalTransportConditionalFlowMatcher

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load config
    with open('src/configs/stage2_mlp.yaml') as f:
        cfg = yaml.safe_load(f)

    # 2. Load Data
    data_root = cfg['data']['root']
    subject = cfg['data']['subject']
    sub_num = int(subject.replace("subj", "").lstrip("0"))
    train_fmri_path = os.path.join(data_root, subject, f"nsd_train_fmri_zscore_sub{sub_num}.npy")
    train_dino_path = os.path.join(data_root, subject, f"nsd_dinov2_vitl14_train_sub{sub_num}.npy")
    
    print("\n--- 1. Data Sanity Check ---")
    train_ds = FmriFeatureDataset(train_fmri_path, train_dino_path, split="train", max_samples=256)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=False)
    fmri, dino = next(iter(train_loader))
    fmri = fmri.to(device)
    dino = dino.to(device)
    print(f"fmri basic stats: mean={fmri.mean().item():.4f}, std={fmri.std().item():.4f}, min={fmri.min().item():.4f}, max={fmri.max().item():.4f}")
    print(f"dino basic stats: mean={dino.mean().item():.4f}, std={dino.std().item():.4f}, norm={dino.norm(dim=-1).mean().item():.4f}")

    # 3. Load VAE
    print("\n--- 2. VAE Sanity Check ---")
    vae_cfg_path = os.path.join(os.path.dirname(cfg['data']['vae_checkpoint']), 'config.yaml')
    with open(vae_cfg_path) as f:
         vae_cfg = yaml.safe_load(f)
    vae = FmriMLPVAE(FmriMLPVAEConfig(**vae_cfg['model'])).to(device).eval()
    ckpt = torch.load(cfg['data']['vae_checkpoint'], map_location=device, weights_only=False)
    vae.load_state_dict(ckpt['model_state_dict'])
    for p in vae.parameters(): p.requires_grad = False
    
    with torch.no_grad():
        z1, mu, logvar = vae.encode(fmri, sample_posterior=False)
    print(f"z1 stats: mean={z1.mean().item():.4f}, std={z1.std().item():.4f}, min={z1.min().item():.4f}, max={z1.max().item():.4f}")
    print(f"z1 norm mean: {z1.norm(dim=-1).mean().item():.4f}")

    # 4. Target Scaling & Flow Matcher
    print("\n--- 3. Flow Matcher Targets ---")
    fm = ConditionalFlowMatcher(sigma=0.0) # Using ICFM for simple testing
    x0 = torch.randn_like(z1)
    t, xt, ut = fm.sample_location_and_conditional_flow(x0, z1)
    print(f"x0 (noise) stats: mean={x0.mean().item():.4f}, std={x0.std().item():.4f}")
    print(f"xt (interp) stats: mean={xt.mean().item():.4f}, std={xt.std().item():.4f}")
    print(f"ut (target) stats: mean={ut.mean().item():.4f}, std={ut.std().item():.4f}, norm={ut.norm(dim=-1).mean().item():.4f}")
    
    # Check what happens if a model just outputs 0
    mse_zero = F.mse_loss(torch.zeros_like(ut), ut).item()
    print(f"MSE if model predicts 0: {mse_zero:.4f}")
    # Check what happens if a model predicts the mean of z1 (roughly 0) - x0
    mse_mean = F.mse_loss(0.0 - x0, ut).item()
    print(f"MSE if model predicts unconditional flow to 0: {mse_mean:.4f}")

    # 5. Model Overfitting Test (Single Batch)
    print("\n--- 4. Single Batch Overfitting Test ---")
    # Small model for fast testing
    mlp_config = FlowMLPConfig(latent_dim=1024, hidden_dim=512, depth=4, context_dim=1024, dropout=0.0)
    model = LatentFlowMLP(mlp_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Fix the noise and t for the single batch to perfectly overfit
    x0_fixed = torch.randn_like(z1)
    t_fixed = torch.rand(z1.shape[0], device=device)
    
    print("Training on 1 batch for 500 steps to see if loss goes to ~0...")
    model.train()
    for step in range(500):
        # We can either fix t and x0, or sample them. 
        # Overfitting the whole ODE trajectory requires sampling t.
        # Let's sample t to ensure it learns the field, but fix x0 to z1 pairs.
        t_s, xt_s, ut_s = fm.sample_location_and_conditional_flow(x0_fixed, z1)
        
        optimizer.zero_grad()
        v_pred = model(t_s, xt_s, dino)
        loss = F.mse_loss(v_pred, ut_s)
        loss.backward()
        optimizer.step()
        
        if (step+1) % 100 == 0:
            print(f"  Step {step+1}, Loss: {loss.item():.6f}")

    # Eval on the overfit batch
    model.eval()
    from torchdiffeq import odeint
    class ODEWrapper(torch.nn.Module):
        def __init__(self, m, c): super().__init__(); self.m, self.c = m, c
        def forward(self, t, z): return self.m(t.expand(z.shape[0]), z, self.c)
    
    with torch.no_grad():
        t_span = torch.linspace(0, 1, 50, device=device)
        traj = odeint(ODEWrapper(model, dino), x0_fixed, t_span, method="euler")
        z_gen = traj[-1]
        
        mse = F.mse_loss(z_gen, z1).item()
        
        # Sample-wise PCC
        pred_zm = z_gen - z_gen.mean(dim=1, keepdim=True)
        tgt_zm = z1 - z1.mean(dim=1, keepdim=True)
        num = (pred_zm * tgt_zm).sum(dim=1)
        den = (pred_zm.norm(dim=1) * tgt_zm.norm(dim=1)).clamp(min=1e-8)
        spcc = (num / den).mean().item()
        
        print(f"After overfitting: Latent MSE={mse:.4f}, Latent Sample PCC={spcc:.4f}")
        
        # Test sensitivity to conditioning
        v_cond = model(torch.full((z1.shape[0],), 0.5, device=device), xt_s, dino)
        v_shuf = model(torch.full((z1.shape[0],), 0.5, device=device), xt_s, dino[torch.randperm(z1.shape[0])])
        diff = (v_cond - v_shuf).norm(dim=-1).mean().item()
        print(f"Velocity sensitivity to shuffled context (||v_cond - v_shuf||): {diff:.4f}")

    # 6. Check DINOv2 vs fMRI Repetitions
    print("\n--- 5. DINOv2 vs fMRI Repetitions Check ---")
    print("In SynBrain, 1 image -> 3 fMRI trials. The train dataset expands DINO by 3x.")
    print("Let's check if the raw DINO tokens for adjacent reps are exactly identical.")
    print("If they are, predicting the EXACT trial might be impossible, creating an upper bound on PCC.")
    is_identical = torch.allclose(dino[0], dino[1]) and torch.allclose(dino[1], dino[2])
    print(f"Batch index 0, 1, 2 identical DINO? {is_identical}")
    fmri_diff_01 = (fmri[0] - fmri[1]).norm().item() / fmri[0].norm().item()
    print(f"Relative difference between fMRI trial 0 and 1 (same image): {fmri_diff_01:.4f}")
    z_diff_01 = (z1[0] - z1[1]).norm().item() / z1[0].norm().item()
    print(f"Relative difference between Latent trial 0 and 1: {z_diff_01:.4f}")

if __name__ == '__main__':
    main()
