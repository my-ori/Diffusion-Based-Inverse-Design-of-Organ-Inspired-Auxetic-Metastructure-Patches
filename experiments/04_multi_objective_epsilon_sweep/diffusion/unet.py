# -*- coding: utf-8 -*-
"""
A compact UNet for 1-channel 2D images (e.g., 50x50 binary unit cells).

- Predicts noise epsilon for DDPM.
- Works for arbitrary HxW via crop/pad matching of skip connections.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import center_crop_like


def _gn_groups(ch: int, max_groups: int = 8) -> int:
    """Pick a GroupNorm group count that divides channels."""
    g = min(max_groups, ch)
    while ch % g != 0 and g > 1:
        g -= 1
    return g


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        emb = math.log(10000) / max(1, (half - 1))
        emb = torch.exp(torch.arange(half, device=device, dtype=torch.float32) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=_gn_groups(in_ch), num_channels=in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch),
        )

        self.norm2 = nn.GroupNorm(num_groups=_gn_groups(out_ch), num_channels=out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


@dataclass
class UNetConfig:
    in_channels: int = 1
    out_channels: int = 1
    base_channels: int = 64
    channel_mults: Tuple[int, ...] = (1, 2, 4)
    num_res_blocks: int = 2
    dropout: float = 0.0
    time_emb_dim: int = 256


class UNet(nn.Module):
    def __init__(self, cfg: UNetConfig):
        super().__init__()
        self.cfg = cfg

        time_dim = cfg.time_emb_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        self.in_conv = nn.Conv2d(cfg.in_channels, cfg.base_channels, kernel_size=3, padding=1)

        # -------- Down path --------
        ch = cfg.base_channels
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        # record skip channel sizes: one per ResBlock (NOT per downsample)
        self.skip_channels: List[int] = []

        for si, mult in enumerate(cfg.channel_mults):
            out_ch = cfg.base_channels * mult
            for _ in range(cfg.num_res_blocks):
                self.down_blocks.append(ResBlock(ch, out_ch, time_dim, dropout=cfg.dropout))
                ch = out_ch
                self.skip_channels.append(ch)
            if si != (len(cfg.channel_mults) - 1):
                self.downsamples.append(Downsample(ch))
            else:
                self.downsamples.append(nn.Identity())

        # -------- Middle --------
        self.mid1 = ResBlock(ch, ch, time_dim, dropout=cfg.dropout)
        self.mid2 = ResBlock(ch, ch, time_dim, dropout=cfg.dropout)

        # -------- Up path --------
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        # We will pop exactly len(channel_mults)*num_res_blocks skips.
        skip_ch_list = list(self.skip_channels)

        for si, mult in reversed(list(enumerate(cfg.channel_mults))):
            out_ch = cfg.base_channels * mult
            for _ in range(cfg.num_res_blocks):
                skip_ch = skip_ch_list.pop()
                self.up_blocks.append(ResBlock(ch + skip_ch, out_ch, time_dim, dropout=cfg.dropout))
                ch = out_ch
            if si != 0:
                self.upsamples.append(Upsample(ch))
            else:
                self.upsamples.append(nn.Identity())

        assert len(skip_ch_list) == 0, "Internal error: skip channel accounting mismatch."

        self.out_norm = nn.GroupNorm(num_groups=_gn_groups(ch), num_channels=ch)
        self.out_conv = nn.Conv2d(ch, cfg.out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: (B,1,H,W) in [-1,1]
        t: (B,) int timesteps
        """
        t_emb = self.time_mlp(t)

        h = self.in_conv(x)

        # Down: store skip after each ResBlock
        skips: List[torch.Tensor] = []
        bptr = 0
        for si in range(len(self.cfg.channel_mults)):
            for _ in range(self.cfg.num_res_blocks):
                h = self.down_blocks[bptr](h, t_emb)
                bptr += 1
                skips.append(h)
            h = self.downsamples[si](h)

        # Mid
        h = self.mid2(self.mid1(h, t_emb), t_emb)

        # Up: for each stage, run resblocks consuming skips, then upsample
        up_bptr = 0
        for si in range(len(self.cfg.channel_mults)):
            for _ in range(self.cfg.num_res_blocks):
                skip = skips.pop()
                h = center_crop_like(h, skip)
                h = torch.cat([h, skip], dim=1)
                h = self.up_blocks[up_bptr](h, t_emb)
                up_bptr += 1
            h = self.upsamples[si](h)

        return self.out_conv(F.silu(self.out_norm(h)))
