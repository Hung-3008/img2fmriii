
import torch
import torch.nn as nn
from .kuramoto import KuramotoLayer
from .hemo import BalloonWindkessel
import sys
import os

# Assuming we can import from existing vae module if path is set correctly
try:
    from vae.brainvae import BrainVAE, MLP
except ImportError:
    pass

class PhysicalDecoderVAE(nn.Module):
    """
    OPTION 1: Hybrid Decoder (Physical Decoder)
    
    Path: z -> [Omega, K] -> Kuramoto -> Neural -> Balloon -> BOLD -> Refinement -> Recon
    
    The decoder IS the physics simulation + a small refinement net.
    """
    def __init__(self, original_vae_config, num_nodes, dt=0.1, simulation_duration=100):
        super().__init__()
        
        # 1. Load basic VAE encoder structure
        self.encoder = original_vae_config.encoder
        self.pre_projector_mean = original_vae_config.pre_projector_mean
        self.pre_projector_logvar = original_vae_config.pre_projector_logvar
        self.kl_weight = getattr(original_vae_config, 'kl_weight', 0.001)
        
        # 2. Physics Decoder Blocks
        self.num_nodes = num_nodes
        self.latent_dim = original_vae_config.embed_dim # 1664
        self.dt = dt
        self.simulation_duration = simulation_duration
        
        # Mappers: Latent z -> Physics Parameters
        self.z_to_omega = nn.Linear(self.latent_dim, num_nodes)
        
        self.z_to_k = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus() 
        )
        
        self.z_to_theta0 = nn.Linear(self.latent_dim, num_nodes)
        
        # Physics Layers
        self.kuramoto = KuramotoLayer(num_nodes, dt=dt, duration=simulation_duration)
        self.balloon = BalloonWindkessel(dt=dt)
        
        # 3. Refinement Layer (Conv1D)
        self.refinement = nn.Sequential(
            nn.Conv1d(num_nodes, num_nodes, kernel_size=3, padding=1, groups=1),
            nn.GELU(),
            nn.Conv1d(num_nodes, num_nodes, kernel_size=1)
        )
        
    def encode(self, x):
        # Calls the standard BrainVAE encoder
        x_enc = self.encoder(x) 
        x_mean = self.pre_projector_mean(x_enc)
        x_logvar = self.pre_projector_logvar(x_enc)
        return x_mean, x_logvar

    def decode(self, z):
        # 1. Map z to parameters
        if z.dim() == 3:
            z_flat = z.mean(dim=1) 
        else:
            z_flat = z
            
        omega = self.z_to_omega(z_flat)       # [B, N]
        k_global = self.z_to_k(z_flat)        # [B, 1]
        theta0 = self.z_to_theta0(z_flat)     # [B, N]
        
        # 2. Kuramoto Dynamics
        phases = self.kuramoto(theta0, k_global, omega) # [B, Time, N]
        
        # Proxy for neural activity (Rectified Cosine Similarity or Rate)
        neural_activity = torch.relu(torch.cos(phases) + 1.0) 
        
        # 3. Hemodynamics
        bold = self.balloon(neural_activity) # [B, Time, N]
        
        # 4. Refinement
        # bold shape: [B, Time, N] -> [B, N, Time] for Conv1D
        bold_t = bold.permute(0, 2, 1)
        recon_t = self.refinement(bold_t)
        
        # Back to [B, Time, N] or [B, N] depending on target
        # BrainVAE typically targets [B, N] (single volume?) or [B, Time, N]?
        # If BrainVAE targets static vector (averaged), we average here.
        # But simulation duration might not match TR. We assume we output time series.
        # For simplicity in this VAE adaptation, let's output the MEAN volume or last volume
        # to match original BrainVAE's likely [B, N] output if it's not time-series based.
        # BUT, original BrainVAE seems to use 1D Conv, implying time or spatial sequence?
        # Re-checking brainvae.py: it takes [B, 1, N] or [B, N]? 
        # Actually NeuroEncoder uses Conv1D, suggesting spatial 1D or temporal 1D.
        # Given "BrainVAE" usually models voxel vectors, it might be Spatial 1D (flattened brain).
        # Let's assume we return the full refined time series or average it if needed.
        # For this hybrid model, let's return the refined time-series [B, N, Time]
        
        return recon_t

    def forward(self, x):
        means, logvars = self.encode(x)
        z = means + torch.randn_like(means) * torch.exp(0.5 * logvars)
        recon = self.decode(z)
        return recon, means, logvars


class ResidualPhysicsVAE(nn.Module):
    """
    OPTION 2: Residual Block (Physics as Skip Connection)
    
    x_out = NeuroDecoder(z) + alpha * PhysicsDecoder(z)
    
    The Neural Decoder does the heavy lifting, Physics adds biological grounding.
    """
    def __init__(self, original_vae_config, num_nodes, dt=0.1, simulation_duration=100):
        super().__init__()
        
        # 1. Standard BrainVAE Components (Encoder + Decoder)
        self.base_vae = BrainVAE(
            ddconfig=original_vae_config.ddconfig,
            hidden_dim=original_vae_config.hidden_dim, 
            embed_dim=original_vae_config.embed_dim
        )
        self.base_vae.load_state_dict(original_vae_config.state_dict())
        
        # 2. Physics Branch (Same as Option 1, but maybe lighter or no refinement)
        self.num_nodes = num_nodes
        self.latent_dim = original_vae_config.embed_dim
        
        self.z_to_omega = nn.Linear(self.latent_dim, num_nodes)
        self.z_to_k = nn.Linear(self.latent_dim, 1) # Simple linear map
        self.z_to_theta0 = nn.Linear(self.latent_dim, num_nodes)
        
        self.kuramoto = KuramotoLayer(num_nodes, dt=dt, duration=simulation_duration)
        self.balloon = BalloonWindkessel(dt=dt)
        
        # Alpha (Learnable scalar weight for physics contribution)
        self.alpha_physics = nn.Parameter(torch.tensor(0.1))
        
        # Projector to match dimensions if needed
        # Physics outputs [B, T, N]. Neural Decoder outputs [B, N] typically?
        # We need to adapt the physics output to match Neural Decoder's shape.
        self.physics_adapter = nn.Sequential(
            nn.Conv1d(num_nodes, num_nodes, kernel_size=1), # Mixing channel
            nn.AdaptiveAvgPool1d(1) # Collapse time if Neural Decoder is static
        )

    def decode_physics(self, z):
        # ... Similar physics logic ...
        if z.dim() == 3: z_flat = z.mean(dim=1)
        else: z_flat = z
            
        omega = self.z_to_omega(z_flat)
        k_global = torch.softplus(self.z_to_k(z_flat))
        theta0 = self.z_to_theta0(z_flat)
        
        phases = self.kuramoto(theta0, k_global, omega)
        neural = torch.relu(torch.cos(phases) + 1.0)
        bold = self.balloon(neural) # [B, T, N]
        
        return bold

    def forward(self, x):
        # 1. Base VAE Forward
        # We might need to call components manually to access z
        # or rely on base_vae returning z.
        # Let's use base_vae.encode() structure if available or copy logic.
        
        # Recopying forward logic from BrainVAE for granular control:
        posterior = self.base_vae.encode(x)
        z = posterior.sample()
        
        # Neural Recon
        recon_neural = self.base_vae.decode(z, target_length=x.shape[2])
        
        # 2. Physics Forward
        bold_physics = self.decode_physics(z) # [B, T, N]
        
        # Adapt Physics to Neural Shape
        # Neural Recon is [B, Channels, Length] or similar.
        # Looking at module.py, NeuroDecoder outputs [B, 4*2, 1024]? Or [B, 1, 15724]?
        # Assuming we align shapes:
        bold_physics_t = bold_physics.permute(0, 2, 1) # [B, N, T]
        
        # If shapes mismatch, we rely on adapter
        # But for this implementation to be generic, we assume user handles dimension matching
        # via the adapter or interpolation.
        physics_contrib = self.physics_adapter(bold_physics_t) 
        
        # 3. Combine
        # x_out = Neural + alpha * Physics
        # Broadcasting might be needed
        recon_combined = recon_neural + self.alpha_physics * physics_contrib
        
        # Recalculate losses
        recon_loss = self.base_vae.mse_loss(recon_combined, x)
        kl_loss = self.base_vae.kl_loss(posterior, x)
        
        return z, recon_combined, recon_loss, kl_loss
