# -*- coding: utf-8 -*-
"""
DDPM (Denoising Diffusion Probabilistic Model) training + sampling.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import DiffusionSchedule, extract


@dataclass
class DDPMConfig:
    timesteps: int = 1000
    schedule: str = "cosine"  # "cosine" or "linear"


class DDPM(nn.Module):
    def __init__(self, model: nn.Module, cfg: DDPMConfig):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.sched = DiffusionSchedule.make(cfg.timesteps, cfg.schedule)

        # Register schedule buffers so they move with .to(device)
        for name, tensor in self.sched.__dict__.items():
            self.register_buffer(name, tensor)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Diffuse x0 -> xt at timestep t.
        x0: (B,1,H,W) in [-1,1]
        t: (B,)
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_acp = extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_om = extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_acp * x0 + sqrt_om * noise

    def p_losses(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Training loss: MSE between true noise and predicted noise.
        """
        if noise is None:
            noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise=noise)
        pred = self.model(xt, t)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Sample x_{t-1} from x_t.
        """
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_acp_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        # model predicts noise
        eps = self.model(x, t)
        model_mean = sqrt_recip_alphas_t * (x - betas_t * eps / sqrt_one_minus_acp_t)

        if (t == 0).all():
            return model_mean

        posterior_var_t = extract(self.posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_var_t) * noise

    @torch.no_grad()
    def sample(self, batch_size: int, shape: Tuple[int, int, int], device: torch.device) -> torch.Tensor:
        """
        shape: (C,H,W)
        returns: (B,C,H,W) in [-1,1]
        """
        c, h, w = shape
        x = torch.randn((batch_size, c, h, w), device=device)
        for i in reversed(range(self.cfg.timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t)
        return x
