# models/cnn_surrogate.py
# -*- coding: utf-8 -*-
"""
CNN surrogate for auxetic unit-cell images (binary masks).
Input:  x  -> (B, 1, 50, 50)
Output: stress_pred -> (B, 30)
        nu_pred     -> (B, 30)

Two-head regression (shared CNN backbone + 2 MLP heads).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Building blocks
# -----------------------------
class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int = 3,
        s: int = 1,
        p: Optional[int] = None,
        act: str = "relu",
    ):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        if self.act == "relu":
            return F.relu(x, inplace=True)
        if self.act == "silu":
            return F.silu(x, inplace=True)
        raise ValueError(f"Unknown activation: {self.act}")


class ResidualBlock(nn.Module):
    """
    Basic ResNet-like block:
      x -> conv3 -> bn -> act -> conv3 -> bn -> +skip -> act
    Downsample when stride=2 or channel mismatch.
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, act: str = "relu", dropout: float = 0.0):
        super().__init__()
        self.act = act
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

        self.conv1 = ConvBNAct(in_ch, out_ch, k=3, s=stride, act=act)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        if self.act == "relu":
            return F.relu(out, inplace=True)
        if self.act == "silu":
            return F.silu(out, inplace=True)
        raise ValueError(f"Unknown activation: {self.act}")


class MLPHead(nn.Module):
    """
    Simple MLP head for regression from pooled features.
    """
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 256, dropout: float = 0.1, act: str = "relu"):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        if self.act == "relu":
            x = F.relu(x, inplace=True)
        elif self.act == "silu":
            x = F.silu(x, inplace=True)
        else:
            raise ValueError(f"Unknown activation: {self.act}")
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# -----------------------------
# Model config
# -----------------------------
@dataclass
class SurrogateConfig:
    in_channels: int = 1
    base_channels: int = 32          # width
    blocks_per_stage: Tuple[int, int, int] = (2, 2, 2)  # depth
    act: str = "relu"
    dropout_backbone: float = 0.0
    dropout_head: float = 0.1
    head_hidden_dim: int = 256
    out_stress_dim: int = 30
    out_nu_dim: int = 30


# -----------------------------
# Main model
# -----------------------------
class AuxeticCNNSurrogate(nn.Module):
    """
    Shared CNN backbone + two regression heads.
    Forward returns:
      (stress_pred, nu_pred) by default
    or a dict if return_dict=True.
    """
    def __init__(self, cfg: SurrogateConfig = SurrogateConfig()):
        super().__init__()
        self.cfg = cfg

        C0 = cfg.base_channels
        act = cfg.act

        # Stem: (B,1,50,50) -> (B,C0,50,50)
        self.stem = nn.Sequential(
            ConvBNAct(cfg.in_channels, C0, k=3, s=1, act=act),
            ConvBNAct(C0, C0, k=3, s=1, act=act),
        )

        # Stages: downsample by stride=2 at stage starts
        # 50 -> 25 -> 13 (approx) -> 7 (approx) depending on padding/stride
        self.stage1 = self._make_stage(C0, C0, n_blocks=cfg.blocks_per_stage[0], stride_first=1, act=act, dropout=cfg.dropout_backbone)
        self.stage2 = self._make_stage(C0, C0 * 2, n_blocks=cfg.blocks_per_stage[1], stride_first=2, act=act, dropout=cfg.dropout_backbone)
        self.stage3 = self._make_stage(C0 * 2, C0 * 4, n_blocks=cfg.blocks_per_stage[2], stride_first=2, act=act, dropout=cfg.dropout_backbone)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        feat_dim = C0 * 4

        # Two heads
        self.head_stress = MLPHead(
            in_dim=feat_dim,
            out_dim=cfg.out_stress_dim,
            hidden_dim=cfg.head_hidden_dim,
            dropout=cfg.dropout_head,
            act=act,
        )
        self.head_nu = MLPHead(
            in_dim=feat_dim,
            out_dim=cfg.out_nu_dim,
            hidden_dim=cfg.head_hidden_dim,
            dropout=cfg.dropout_head,
            act=act,
        )

        self._init_weights()

    def _make_stage(self, in_ch: int, out_ch: int, n_blocks: int, stride_first: int, act: str, dropout: float) -> nn.Sequential:
        blocks = []
        blocks.append(ResidualBlock(in_ch, out_ch, stride=stride_first, act=act, dropout=dropout))
        for _ in range(n_blocks - 1):
            blocks.append(ResidualBlock(out_ch, out_ch, stride=1, act=act, dropout=dropout))
        return nn.Sequential(*blocks)

    def _init_weights(self) -> None:
        # Good defaults for Conv/Linear
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)

    @torch.no_grad()
    def num_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        x: torch.Tensor,
        return_dict: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        x: (B,1,50,50) float32 in [0,1]
        """
        h = self.stem(x)
        h = self.stage1(h)
        h = self.stage2(h)
        h = self.stage3(h)

        h = self.pool(h).flatten(1)  # (B, feat_dim)

        stress = self.head_stress(h)  # (B,30)
        nu = self.head_nu(h)          # (B,30)

        if return_dict:
            return {"stress": stress, "nu": nu}
        return stress, nu

