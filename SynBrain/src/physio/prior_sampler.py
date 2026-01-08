
import torch
import numpy as np
import torch.nn as nn

class BioLatentSampler:
    """
    Sampler that initializes the diffusion process from a biological prior 
    (latent state encoded from Kuramoto simulations) instead of pure noise.
    
    This technique is known as SDEEdit or Image-to-Image diffusion.
    We start the reverse process at time t_start < T.
    
    x_start = alpha_t * x_prior + sigma_t * noise
    """
    def __init__(self, model, diffusion_steps=1000, scheduler=None):
        self.model = model
        self.num_steps = diffusion_steps
        
        # We assume a simple linear schedule for demonstration if scheduler is None
        #Ideally this should match the scheduler used in training (e.g., DDIM, PNDM)
        self.scheduler = scheduler
        
    def add_noise(self, x_start, t):
        """
        Forward diffusion process: q(x_t | x_0)
        Using simple variance preserving (VP) or similar schedule
        """
        # Note: This needs to align with the specific training schedule of SiT/Diffusion model
        # For SiT (Flow Matching), the path is usually straight line: x_t = (1-t)x_0 + t*epsilon (if t in [0,1])
        # Or standard diffusion: x_t = sqrt(alpha_bar)x_0 + sqrt(1-alpha_bar)epsilon
        
        # Assuming SiT is trained with EDM or similar continuous time
        # Let's interpret t as noise level [0, 1] for simplicity in this demo
        noise = torch.randn_like(x_start)
        
        # Linear interpolation (Flow Matching style)
        # t=0 -> x_start (clean)
        # t=1 -> noise
        t_batch = torch.ones(x_start.shape[0], device=x_start.device) * t
        
        # x_t = (1-t) * x_0 + t * noise
        # This is strictly optimal transport path often used in recent DiTs
        x_t = (1 - t) * x_start + t * noise
        return x_t

    def sample_from_prior(self, prior_latent, t_start=0.5, steps=50):
        """
        run reverse diffusion starting from t_start.
        
        Args:
            prior_latent: The 'clean' latent guess from Kuramoto [B, Dim]
            t_start: Starting noise level (0.0 to 1.0). 
                     1.0 = Pure noise (standard generation)
                     0.0 = Returns prior exactly
                     0.5 = Mix of prior and noise
            steps: Number of denoising steps
        """
        batch_size = prior_latent.shape[0]
        device = prior_latent.device
        
        # 1. Noisify the prior to t_start
        # This creates our starting point x_{t_start}
        # It allows the model to hallucinate details (noise) while keeping structure (prior)
        curr_x = self.add_noise(prior_latent, t_start)
        
        # 2. Reverse Loop (Denoising)
        # We integrate from t_start down to 0
        time_steps = torch.linspace(t_start, 0, steps + 1, device=device)
        
        for i in range(steps):
            t_curr = time_steps[i]
            t_next = time_steps[i+1]
            dt = t_next - t_curr # Negative step
            
            # Predict velocity/score using model
            # SiT takes (x, t) -> v (velocity)
            # t needs to be broadcast to batch
            t_input = torch.ones(batch_size, device=device) * t_curr
            
            # Forward pass of the model
            # model(x, t)
            velocity = self.model(curr_x, t_input)
            
            # Euler step: x_{t+dt} = x_t + v * dt
            # Since dt is negative, we move towards clean data
            curr_x = curr_x + velocity * dt
            
        return curr_x
