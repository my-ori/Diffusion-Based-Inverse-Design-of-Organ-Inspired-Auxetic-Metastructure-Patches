# -*- coding: utf-8 -*-
"""
Utility helpers for diffusion training.

This project assumes your images are 1x50x50 tensors with values in {0,1}.
We scale them to [-1, 1] for diffusion training.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import math
import torch
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_neg_one_to_one(x01: torch.Tensor) -> torch.Tensor:
    """x in [0,1] -> [-1,1]."""
    return x01 * 2.0 - 1.0


def to_zero_one(xm11: torch.Tensor) -> torch.Tensor:
    """x in [-1,1] -> [0,1]."""
    return (xm11 + 1.0) * 0.5


def center_crop_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    Center-crop (or pad) tensor x spatially to match ref's HxW.
    x, ref: (B,C,H,W)
    """
    _, _, h, w = x.shape
    _, _, hr, wr = ref.shape

    # If larger, crop
    if h > hr:
        dh = h - hr
        top = dh // 2
        x = x[:, :, top:top+hr, :]
    if w > wr:
        dw = w - wr
        left = dw // 2
        x = x[:, :, :, left:left+wr]

    # If smaller, pad
    _, _, h2, w2 = x.shape
    pad_h = max(0, hr - h2)
    pad_w = max(0, wr - w2)
    if pad_h > 0 or pad_w > 0:
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))

    return x


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
    """
    Extract t-indexed values from a 1D tensor a, and reshape for broadcasting to x.
    a: (T,)
    t: (B,)
    returns: (B,1,1,1) for image-shaped x
    """
    out = a.gather(0, t)
    return out.reshape((t.shape[0],) + (1,) * (len(x_shape) - 1))


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine schedule from:
      Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models"
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 1e-8, 0.999).to(torch.float32)


@dataclass
class DiffusionSchedule:
    betas: torch.Tensor              # (T,)
    alphas: torch.Tensor             # (T,)
    alphas_cumprod: torch.Tensor     # (T,)
    alphas_cumprod_prev: torch.Tensor# (T,)
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor
    sqrt_recip_alphas: torch.Tensor
    posterior_variance: torch.Tensor

    @staticmethod
    def make(timesteps: int, schedule: str = "cosine") -> "DiffusionSchedule":
        if schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0], dtype=torch.float32), alphas_cumprod[:-1]], dim=0)

        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        return DiffusionSchedule(
            betas=betas,
            alphas=alphas,
            alphas_cumprod=alphas_cumprod,
            alphas_cumprod_prev=alphas_cumprod_prev,
            sqrt_alphas_cumprod=sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
            sqrt_recip_alphas=sqrt_recip_alphas,
            posterior_variance=posterior_variance,
        )
