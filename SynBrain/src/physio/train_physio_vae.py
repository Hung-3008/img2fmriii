
import copy
import os
import sys
# Ensure src is in path to find modules
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

import torch
import torch.nn as nn
import argparse
import numpy as np
import wandb
from tqdm import tqdm
from datetime import datetime
import torchvision.utils as vutils

from utils import seed_everything, load_config, count_params
from dataset import multisub_clip_dataset
from mind_utils import topk, batchwise_cosine_similarity
from vae.brainvae import BrainVAE 
from physio import PhysicalDecoderVAE, ResidualPhysicsVAE

def save_fmri_recon_image(fmri, recon):
    """Simple visualization of fMRI vs Recon"""
    import matplotlib.pyplot as plt
    from io import BytesIO
    from PIL import Image
    
    # Take first sample in batch
    orig = fmri[0, 0].cpu().detach().numpy() # [N]
    rec = recon[0].cpu().detach().numpy()    # [N] or [T, N] - handle shapes
    
    if rec.ndim == 2: # [N, T] or [T, N]
        # If output is time series, take mean or first time point
        # Assuming [N, T] from Conv1D or similar
        rec = rec.mean(axis=-1)
        
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    ax[0].plot(orig[:500], label='Original', alpha=0.7)
    ax[0].set_title('Original fMRI (First 500 voxels)')
    
    ax[1].plot(rec[:500], label='Recon', color='orange', alpha=0.7)
    ax[1].set_title(f'Reconstructed fMRI')
    
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def main(args):
    seed_everything(args.seed)
    
    # 1. Load Config (Reuse BrainVAE config for Encoder params)
    config_path = os.path.join(src_dir, "../configs/brainvae.yaml")
    config = load_config(config_path)
    model_config = config["model"]["params"]
    ddconfig = model_config["ddconfig"]
    
    # 2. Setup Data
    args.local_batch_size = args.batch_size
    train_dataloader, val_dataloader = multisub_clip_dataset(args)
    
    # 3. Initialize Model based on Option
    # We need a dummy "original_vae_config" object that holds the encoder config
    # Creating a temporary BrainVAE to extract encoder config wrapper
    print("Initializing Base BrainVAE to extract Encoder...")
    temp_vae = BrainVAE(ddconfig=ddconfig,
                        clip_weight=args.clip_weight,
                        kl_weight=args.kl_weight,
                        hidden_dim=1024,
                        linear_dim=args.linear_dim,
                        embed_dim=1664)
                        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if args.physio_mode == "physical":
        print(">>> Mode: PhysicalDecoderVAE (Option 1)")
        model = PhysicalDecoderVAE(temp_vae, num_nodes=15724, dt=args.dt) # 15724 voxels for Subj1 usually
    elif args.physio_mode == "residual":
        print(">>> Mode: ResidualPhysicsVAE (Option 2)")
        model = ResidualPhysicsVAE(temp_vae, num_nodes=15724, dt=args.dt)
    else:
        raise ValueError(f"Unknown physio_mode: {args.physio_mode}")
        
    del temp_vae # Free memory
    
    count_params(model)
    model.to(device)
    
    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
        
    # 4. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=0.05)
    
    # 5. WandB
    if args.wandb_log:
        wandb.init(project="SynBrain-Physio", name=args.model_name, config=args)
        
    # 6. Training Loop
    print(f"Starting training for {args.num_epochs} epochs...")
    
    outdir = os.path.join(args.save_path, 'train_logs', args.model_name)
    os.makedirs(outdir, exist_ok=True)
    
    for epoch in range(args.num_epochs):
        model.train()
        train_loss_sum = 0
        
        loop = tqdm(train_dataloader, desc=f"Epoch {epoch}")
        for fmri, z, sub_id in loop:
            fmri = fmri.unsqueeze(1).float().to(device) # [B, 1, N]
            z = z.float().to(device)
            
            optimizer.zero_grad()
            
            # Forward depending on model structure
            # Both new models return (recon, means, logvars) or similar tuple
            # We standardized output in hybrid_model.py check?
            # PhysicalDecoderVAE: returns recon, means, logvars
            # ResidualPhysicsVAE: returns z, recon, recon_loss, kl_loss
            
            if args.physio_mode == "physical":
                recon, means, logvars = model(fmri)
                # Need to manually compute loss here since it's not inside forward for Option 1 yet?
                # Actually let's assume standard VAE loss calc:
                # But wait, original BrainVAE calculates loss INSIDE forward.
                # PhysicalDecoderVAE forward returns (recon, means, logvars). We need loss.
                
                # loss calc
                target = fmri # [B, 1, N]
                if recon.shape != target.shape:
                    # recon might be [B, N, Time] or [B, N]
                    # Let's align shapes. If recon is [B, N, T], we take mean over T for voxel-wise match?
                    # Or we assume fmri input is the target static map.
                    if recon.ndim == 3 and target.ndim == 3 and recon.shape[2] != target.shape[2]:
                         # recon [B, 15724, 1], target [B, 1, 15724]
                         recon = recon.view(target.shape) # Try reshape if dims match size
                    
                recon_loss = nn.functional.mse_loss(recon, target, reduction='sum') / target.shape[0]
                kl_loss = 0.5 * torch.sum(torch.exp(logvars) + means**2 - 1. - logvars) / target.shape[0]
                loss = recon_loss + kl_loss * args.kl_weight
                
            else: # residual (Option 2) - returns losses directly
                # returns z, recon, recon_loss, kl_loss
                z_sample, recon, recon_loss, kl_loss = model(fmri)
                loss = recon_loss + kl_loss * args.kl_weight
            
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.item()
            loop.set_postfix(loss=loss.item())
            
        # Validation output
        print(f"Epoch {epoch} | Train Loss: {train_loss_sum / len(train_dataloader):.4f}")
        
        # Save Checkpoint
        if (epoch + 1) % args.ckpt_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(outdir, 'last.pth'))
            
        if args.wandb_log:
            wandb.log({"train/loss": train_loss_sum / len(train_dataloader)})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Model Selection
    parser.add_argument("--physio_mode", type=str, default="physical", choices=["physical", "residual"], help="Choose Option 1 (physical) or Option 2 (residual)")
    parser.add_argument("--dt", type=float, default=0.1, help="Simulation time step")
    
    # Standard Training Args
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16) 
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--model_name", type=str, default="physio_vae_test")
    parser.add_argument("--save_path", type=str, default="/workspace/sdb1/img2fmri/BrainSyn")
    parser.add_argument("--data_path", type=str, default="/workspace/sdb1/img2fmri/NSD/data/nsd")
    
    # Hyperparams
    parser.add_argument("--base_lr", type=float, default=1e-4)
    parser.add_argument("--clip_weight", type=float, default=1000)
    parser.add_argument("--kl_weight", type=float, default=0.001)
    parser.add_argument("--linear_dim", type=int, default=2048)
    parser.add_argument("--ckpt_interval", type=int, default=5)
    
    # NSD specific defaults required by dataset.py
    parser.add_argument("--subject", type=str, default="[1]")
    parser.add_argument("--valid-sub", type=int, default=1)
    parser.add_argument("--unseen-sub", type=int, default=9)
    parser.add_argument("--hour", type=int, default=36)
    parser.add_argument("--wandb_log", action="store_true")
    
    args = parser.parse_args()
    main(args)
