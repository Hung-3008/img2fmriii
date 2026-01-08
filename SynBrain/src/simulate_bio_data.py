
import os
import torch
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import our new physio module
# Assuming we run this from SynBrain/src
import sys
sys.path.append(os.getcwd()) # Ensure src is in path
from physio import KuramotoLayer, BalloonWindkessel

def simulate_data(args):
    """
    Simulate fMRI data using Kuramoto and Balloon models.
    """
    print(f"Generating {args.num_samples} samples...")
    print(f"Nodes: {args.num_nodes}, Steps: {args.steps}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Setup Models
    kuramoto = KuramotoLayer(args.num_nodes, dt=args.dt, duration=args.steps*args.dt).to(device)
    balloon = BalloonWindkessel(dt=args.dt).to(device)
    
    # 2. Parameters Distribution (Feature Prior)
    # We want to sample reasonable biological parameters
    # Omega: Natural frequencies ~ Gaussian(10Hz, 1Hz) * 2pi
    omega_mean = 10.0 * 2 * np.pi
    omega_std = 1.0 * 2 * np.pi
    
    # Coupling K: Uniform(0.5, 5.0)
    
    all_simulated_fmri = []
    
    batch_size = args.batch_size
    
    for _ in tqdm(range(0, args.num_samples, batch_size)):
        current_batch_size = min(batch_size, args.num_samples - len(all_simulated_fmri))
        if current_batch_size <= 0: break
        
        # Sample Parameters
        omega = torch.randn(current_batch_size, args.num_nodes, device=device) * omega_std + omega_mean
        k_global = torch.rand(current_batch_size, 1, device=device) * 4.5 + 0.5
        theta0 = torch.rand(current_batch_size, args.num_nodes, device=device) * 2 * np.pi
        
        # Forward Pass
        with torch.no_grad():
            # A. Kuramoto Phase Dynamics
            # phases: [Batch, T_sim, N]
            phases = kuramoto(theta0, k_global, omega)
            
            # B. Neural Activity Proxy
            # Simple rate coding proxy: relu(cos(phase) + bias)
            neural_activity = torch.relu(torch.cos(phases) + 1.0)
            
            # C. Hemodynamics
            # bold: [Batch, T_sim, N]
            bold = balloon(neural_activity)
            
            # D. Downsample to fMRI TR (e.g., every 2s)
            # dt=0.1s -> 2s TR = every 20 steps
            tr_steps = int(2.0 / args.dt)
            fmri_bold = bold[:, ::tr_steps, :]
            
        all_simulated_fmri.append(fmri_bold.cpu().numpy())
        
    all_simulated_fmri = np.concatenate(all_simulated_fmri, axis=0)
    print(f"Simulated Data Shape: {all_simulated_fmri.shape}")
    
    # Save 
    os.makedirs(args.out_dir, exist_ok=True)
    save_path = os.path.join(args.out_dir, "simulated_bio_fmri.npy")
    np.save(save_path, all_simulated_fmri)
    print(f"Saved to {save_path}")
    
    # Visualization check
    if args.plot:
        plt.figure(figsize=(12, 4))
        plt.plot(all_simulated_fmri[0, :, :5]) # Plot first 5 nodes
        plt.title("Simulated fMRI Signals (First 5 Nodes)")
        plt.xlabel("TR")
        plt.ylabel("BOLD Signal")
        plt.savefig(os.path.join(args.out_dir, "sim_example.png"))
        print("Saved plot.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_nodes", type=int, default=15724) # NSD Voxels
    # For testing, we might want fewer nodes or map ROI average
    # But SynBrain typically works with full voxel set.
    # Warning: 15k nodes all-to-all Kuramoto is O(N^2) heavy!
    # We strongly recommend using sparse matrix or fewer ROI nodes used in the FC analysis.
    
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--steps", type=int, default=300) # Simulation duration in dt steps
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--out_dir", type=str, default="data/simulated")
    parser.add_argument("--plot", action="store_true")
    
    args = parser.parse_args()
    simulate_data(args)
