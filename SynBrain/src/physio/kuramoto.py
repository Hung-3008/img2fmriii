
import torch
import torch.nn as nn
import math

class KuramotoODE(nn.Module):
    """
    Differentiable Kuramoto Model for fMRI simulation.
    
    Equation: dθ_i/dt = ω_i + (K/N) * Σ_j A_ij * sin(θ_j - θ_i - delay_ij)
    
    This module is designed to be used with torchdiffeq.odeint.
    """
    def __init__(self, num_nodes, coupling_matrix=None, mean_delay=0.0):
        super().__init__()
        self.num_nodes = num_nodes
        
        # Structure Connectivity Matrix (A_ij)
        # If None, assumes all-to-all connectivity (not recommended for brain)
        if coupling_matrix is None:
            self.register_buffer('adj_mat', torch.ones(num_nodes, num_nodes))
        else:
             # Ensure zero diagonal
            coupling_matrix = coupling_matrix.fill_diagonal_(0)
            self.register_buffer('adj_mat', coupling_matrix)
            
        self.mean_delay = mean_delay

    def forward(self, t, theta, k_global, omega):
        """
        Compute dtheta/dt given current state theta.
        
        Args:
            t: Current time (scalar).
            theta: Phase of oscillators [batch_size, num_nodes].
            k_global: Global coupling strength [batch_size, 1].
            omega: Natural frequencies [batch_size, num_nodes].
             
        Returns:
            dtheta_dt: Derivative of phase [batch_size, num_nodes].
        """
        # theta shape: [B, N]
        # Calculate phase differences: sin(θ_j - θ_i)
        # Expansion for broadcasting:
        # theta_i: [B, N, 1]
        # theta_j: [B, 1, N]
        
        theta_i = theta.unsqueeze(2)
        theta_j = theta.unsqueeze(1)
        
        delta_theta = theta_j - theta_i 
        
        # Interaction term: sin(θ_j - θ_i)
        interaction = torch.sin(delta_theta)
        
        # Weighted by Adjacency Matrix
        # adj_mat: [N, N] -> broadcast to [B, N, N]
        weighted_interaction = self.adj_mat * interaction
        
        # Sum over j: Σ_j A_ij * sin(...)
        # Shape: [B, N]
        sum_interaction = weighted_interaction.sum(dim=2)
        
        # Final equation: ω + (K/N) * sum
        # K_global: [B, 1] -> broadcast to [B, N]
        dtheta_dt = omega + (k_global / self.num_nodes) * sum_interaction
        
        return dtheta_dt

class KuramotoLayer(nn.Module):
    """
    Wrapper to solve Kuramoto equations over a time horizon.
    """
    def __init__(self, num_nodes, dt=0.1, duration=100):
        super().__init__()
        self.kuramoto = KuramotoODE(num_nodes)
        self.dt = dt
        self.duration = duration
        
    def forward(self, initial_phase, k_global, omega, solver='euler'):
        """
        Solve Kuramoto equation.
        
        Args:
            initial_phase: [B, N]
            k_global: [B, 1]
            omega: [B, N]
            solver: 'euler' or 'rk4' (custom implementation for simplicity/control)
            
        Returns:
            phases: [B, Time, N]
        """
        batch_size = initial_phase.shape[0]
        num_steps = int(self.duration / self.dt)
        
        phases = [initial_phase]
        curr_theta = initial_phase
        
        # Simple differentiable Euler solver loop
        # We manually unroll or loop to keep gradients flowing
        for _ in range(num_steps):
            dtheta = self.kuramoto(0, curr_theta, k_global, omega)
            curr_theta = curr_theta + dtheta * self.dt
            # Wrap to -pi, pi? For ODE solver usually better to let it run unbounded 
            # and wrap only for sin/cos metric, but sin() handles unbounded fine.
            phases.append(curr_theta)
            
        return torch.stack(phases, dim=1)
