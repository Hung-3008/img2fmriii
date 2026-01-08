
import torch
import torch.nn as nn

class BalloonWindkessel(nn.Module):
    """
    Balloon-Windkessel Model (Friston et al., 2000).
    Converts Neural Activity (z) -> BOLD Signal (y).
    
    States:
    s: vasodilatory signal
    f: blood inflow
    v: blood volume
    q: deoxyhemoglobin content
    """
    def __init__(self, tr=2.0, dt=0.1):
        super().__init__()
        self.tr = tr
        self.dt = dt # Integration step size
        
        # Hemodynamic Parameters (standard priors)
        self.epsilon = 0.54  # Neuronal efficacy
        self.tau_s = 1.54    # Signal decay
        self.tau_f = 2.46    # Autoregulation
        self.tau_0 = 0.98    # Transit time
        self.alpha = 0.33    # Stiffness
        self.E0 = 0.34       # Resting oxygen extraction fraction
        self.V0 = 100.0      # Resting blood volume fraction (custom scale)
        
        # BOLD signal coefficients
        # k1 = 7 * E0
        # k2 = 2
        # k3 = 2 * E0 - 0.2
        self.k1 = 7 * self.E0
        self.k2 = 2.0
        self.k3 = 2 * self.E0 - 0.2

    def derivatives(self, state, u):
        """
        Calculate derivatives [ds, df, dv, dq] / dt.
        
        state: [s, f, v, q] (Batch, Nodes, 4)
        u: Neural input (Batch, Nodes)
        """
        s = state[..., 0]
        f = state[..., 1]
        v = state[..., 2]
        q = state[..., 3]
        
        # 1. Vasodilatory signal
        ds = self.epsilon * u - s / self.tau_s - (f - 1) / self.tau_f
        
        # 2. Blood Inflow 
        df = s / self.tau_s
        
        # 3. Blood Volume
        dv = (f - v**(1/self.alpha)) / self.tau_0
        
        # 4. Deoxyhemoglobin Content
        # f_out = v^(1/alpha)
        # E(f) = 1 - (1 - E0)^(1/f)
        f_out = v**(1/self.alpha)
        
        # Numerical stability clip for f
        f_safe = torch.clamp(f, min=1e-6)
        extraction = 1 - (1 - self.E0)**(1 / f_safe)
        
        dq = (f * extraction / self.E0 - f_out * (q / v)) / self.tau_0
        
        return torch.stack([ds, df, dv, dq], dim=-1)

    def forward(self, neural_activity):
        """
        Simulate BOLD response for a time series of neural activity.
        
        Args:
            neural_activity: [Batch, Time, Nodes] (High temporal resolution, e.g. 10Hz)
            
        Returns:
            bold_signal: [Batch, Time, Nodes]
        """
        batch_size, time_steps, num_nodes = neural_activity.shape
        device = neural_activity.device
        
        # Initial State: s=0, f=1, v=1, q=1
        curr_state = torch.ones(batch_size, num_nodes, 4, device=device)
        curr_state[..., 0] = 0.0 # s=0
        
        bold_signals = []
        
        # Euler Integration
        for t in range(time_steps):
            u = neural_activity[:, t, :]
            
            grads = self.derivatives(curr_state, u)
            curr_state = curr_state + grads * self.dt
            
            # Calculate BOLD from v and q
            v = curr_state[..., 2]
            q = curr_state[..., 3]
            
            # BOLD equation
            y = self.V0 * (self.k1 * (1 - q) + self.k2 * (1 - q/v) + self.k3 * (1 - v))
            bold_signals.append(y)
            
        return torch.stack(bold_signals, dim=1)
