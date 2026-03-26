"""CSFM ODE integrator — adapted for BrainFlow.

Copied from CSFM. Key: time goes from t=1 → t=0 (source → target).
self.t = 1 - linspace(t0, t1, N) then shifted by time_dist_shift.
"""

import torch as th
from torchdiffeq import odeint


class ode:
    """ODE solver class (from CSFM)."""
    def __init__(self, drift, *, t0, t1, sampler_type, num_steps, atol, rtol, time_dist_shift):
        assert t0 < t1
        self.drift = drift
        self.t = 1 - th.linspace(t0, t1, num_steps)
        self.t = time_dist_shift * self.t / (1 + (time_dist_shift - 1) * self.t)
        self.atol = atol
        self.rtol = rtol
        self.sampler_type = sampler_type

    def sample(self, x, model, **model_kwargs):
        device = x[0].device if isinstance(x, tuple) else x.device

        def _fn(t, x):
            t_vec = th.ones(x[0].size(0)).to(device) * t if isinstance(x, tuple) \
                else th.ones(x.size(0)).to(device) * t
            return self.drift(x, t_vec, model, **model_kwargs)

        t = self.t.to(device)
        atol = [self.atol] * len(x) if isinstance(x, tuple) else [self.atol]
        rtol = [self.rtol] * len(x) if isinstance(x, tuple) else [self.rtol]
        samples = odeint(_fn, x, t, method=self.sampler_type, atol=atol, rtol=rtol)
        return samples
