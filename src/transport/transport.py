"""CSFM Transport core — adapted for BrainFlow.

Copied from CSFM with modifications:
- Removed training_losses() / training_losses_textve() (we do this in the model)
- Kept: sample_timestep(), get_drift(), get_score(), Sampler (ODE/SDE)
- Kept: truncated_logitnormal_sample for time distribution
"""

import torch as th
import numpy as np
import enum

from . import path
from .utils import mean_flat
from .integrators import ode


class ModelType(enum.Enum):
    NOISE = enum.auto()
    SCORE = enum.auto()
    VELOCITY = enum.auto()


class PathType(enum.Enum):
    LINEAR = enum.auto()
    GVP = enum.auto()
    VP = enum.auto()


class WeightType(enum.Enum):
    NONE = enum.auto()
    VELOCITY = enum.auto()
    LIKELIHOOD = enum.auto()


def truncated_logitnormal_sample(shape, mu, sigma, low=0.0, high=1.0):
    mu = th.as_tensor(mu)
    sigma = th.as_tensor(sigma)
    low = th.as_tensor(low)
    high = th.as_tensor(high)

    z_low = th.logit(low)
    z_high = th.logit(high)

    base = th.distributions.Normal(th.zeros_like(mu), th.ones_like(sigma))
    alpha = (z_low - mu) / sigma
    beta = (z_high - mu) / sigma

    cdf_alpha = base.cdf(alpha)
    cdf_beta = base.cdf(beta)

    out_shape = th.broadcast_shapes(shape, mu.shape, sigma.shape, low.shape, high.shape)
    U = th.rand(out_shape, device=mu.device, dtype=mu.dtype)
    U = cdf_alpha + (cdf_beta - cdf_alpha) * U.clamp_(0, 1)

    Z = mu + sigma * base.icdf(U)
    X = th.sigmoid(Z)
    return X.clamp(low, high)


class Transport:
    def __init__(
        self,
        *,
        model_type,
        path_type,
        loss_type,
        time_dist_type,
        time_dist_shift,
        train_eps,
        sample_eps,
    ):
        path_options = {
            PathType.LINEAR: path.ICPlan,
            PathType.GVP: path.GVPCPlan,
            PathType.VP: path.VPCPlan,
        }
        self.loss_type = loss_type
        self.model_type = model_type
        self.time_dist_type = time_dist_type
        self.time_dist_shift = time_dist_shift
        assert self.time_dist_shift >= 1.0
        self.path_sampler = path_options[path_type]()
        self.train_eps = train_eps
        self.sample_eps = sample_eps

    def check_interval(self, train_eps, sample_eps, *, diffusion_form="SBDM",
                       sde=False, reverse=False, eval=False, last_step_size=0.0):
        t0 = 0
        t1 = 1 - 1 / 1000
        eps = train_eps if not eval else sample_eps
        if type(self.path_sampler) in [path.VPCPlan]:
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size
        elif (type(self.path_sampler) in [path.ICPlan, path.GVPCPlan]) \
                and (self.model_type != ModelType.VELOCITY or sde):
            t0 = eps if (diffusion_form == "SBDM" and sde) or self.model_type != ModelType.VELOCITY else 0
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size
        if reverse:
            t0, t1 = 1 - t0, 1 - t1
        return t0, t1

    def sample_timestep(self, x1):
        """Sample timestep t based on shape of x1."""
        dist_options = self.time_dist_type.split("_")
        t0, t1 = self.check_interval(self.train_eps, self.sample_eps)
        if dist_options[0] == "uniform":
            t = th.rand((x1.shape[0],)) * (t1 - t0) + t0
        elif dist_options[0] == "logit-normal":
            assert len(dist_options) == 3
            mu, sigma = float(dist_options[1]), float(dist_options[2])
            t = truncated_logitnormal_sample((x1.shape[0],), mu=mu, sigma=sigma, low=t0, high=t1)
        else:
            raise NotImplementedError(f"Unknown time distribution type {self.time_dist_type}")
        t = t.to(x1)
        t = self.time_dist_shift * t / (1 + (self.time_dist_shift - 1) * t)
        return t

    def get_drift(self):
        def velocity_ode(x, t, model, **model_kwargs):
            model_output = model(x, t, **model_kwargs)
            return model_output

        if self.model_type == ModelType.VELOCITY:
            drift_fn = velocity_ode
        else:
            raise NotImplementedError("Only velocity model supported in BrainFlow")

        def body_fn(x, t, model, **model_kwargs):
            model_output = drift_fn(x, t, model, **model_kwargs)
            assert model_output.shape == x.shape
            return model_output

        return body_fn

    def get_score(self):
        if self.model_type == ModelType.VELOCITY:
            score_fn = lambda x, t, model, **kwargs: \
                self.path_sampler.get_score_from_velocity(model(x, t, **kwargs), x, t)
        else:
            raise NotImplementedError()
        return score_fn


class Sampler:
    """Sampler class for the transport model."""
    def __init__(self, transport):
        self.transport = transport
        self.drift = self.transport.get_drift()
        self.score = self.transport.get_score()

    def sample_ode(self, *, sampling_method="dopri5", num_steps=50,
                   atol=1e-6, rtol=1e-3, reverse=False):
        if reverse:
            drift = lambda x, t, model, **kwargs: self.drift(x, th.ones_like(t) * (1 - t), model, **kwargs)
        else:
            drift = self.drift

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps, self.transport.sample_eps,
            sde=False, eval=True, reverse=reverse, last_step_size=0.0,
        )

        _ode = ode(
            drift=drift, t0=t0, t1=t1,
            sampler_type=sampling_method, num_steps=num_steps,
            atol=atol, rtol=rtol,
            time_dist_shift=self.transport.time_dist_shift,
        )
        return _ode.sample
