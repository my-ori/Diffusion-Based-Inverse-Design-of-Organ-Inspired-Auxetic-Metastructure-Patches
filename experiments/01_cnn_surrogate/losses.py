# losses.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossConfig:
    # Base pointwise MSE (normalized space)
    w_stress: float = 1.0
    w_nu: float = 1.0

    # Shape-aware (computed in PHYSICAL space after denormalization)
    w_d1_stress: float = 0.0   # slope match
    w_d2_stress: float = 0.0   # curvature match
    w_d1_nu: float = 0.0
    w_d2_nu: float = 0.0

    # Physics-ish constraints (PHYSICAL space)
    w_mono_stress: float = 0.0  # penalize negative stress slope

    # Optional bounds on nu (PHYSICAL space)
    w_bound_nu: float = 0.0
    nu_min: float = -1.2
    nu_max: float = 1.2


class SurrogateLoss(nn.Module):
    """
    If you pass norm stats (mean/std per strain point), we can denormalize inside loss.
    - stress_mean/std: (30,)
    - nu_mean/std: (30,)
    """
    def __init__(
        self,
        cfg: LossConfig = LossConfig(),
        stress_mean: Optional[torch.Tensor] = None,
        stress_std: Optional[torch.Tensor] = None,
        nu_mean: Optional[torch.Tensor] = None,
        nu_std: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.cfg = cfg

        # Register as buffers so they move with .to(device)
        if stress_mean is not None:
            self.register_buffer("stress_mean", stress_mean.float())
            self.register_buffer("stress_std", stress_std.float())
            self.register_buffer("nu_mean", nu_mean.float())
            self.register_buffer("nu_std", nu_std.float())
        else:
            self.stress_mean = None
            self.stress_std = None
            self.nu_mean = None
            self.nu_std = None

    def _denorm(self, y_norm: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return y_norm * std.view(1, -1) + mean.view(1, -1)

    def _d1(self, y: torch.Tensor) -> torch.Tensor:
        return y[:, 1:] - y[:, :-1]  # (B,29)

    def _d2(self, y: torch.Tensor) -> torch.Tensor:
        return y[:, 2:] - 2.0 * y[:, 1:-1] + y[:, :-2]  # (B,28)

    def forward(
        self,
        stress_pred: torch.Tensor,  # (B,30) normalized
        nu_pred: torch.Tensor,      # (B,30) normalized
        y_stress: torch.Tensor,     # (B,30) normalized
        y_nu: torch.Tensor,         # (B,30) normalized
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        cfg = self.cfg

        # --- Base losses in normalized space ---
        L_stress = F.mse_loss(stress_pred, y_stress, reduction="mean")
        L_nu = F.mse_loss(nu_pred, y_nu, reduction="mean")
        total = cfg.w_stress * L_stress + cfg.w_nu * L_nu

        logs: Dict[str, torch.Tensor] = {
            "loss_total": total.detach(),
            "loss_stress": L_stress.detach(),
            "loss_nu": L_nu.detach(),
        }

        # If no norm stats, we cannot do physical-space shape penalties safely
        if self.stress_mean is None:
            return total, logs

        # --- Denormalize to physical space for shape penalties ---
        stress_pred_p = self._denorm(stress_pred, self.stress_mean, self.stress_std)
        stress_true_p = self._denorm(y_stress, self.stress_mean, self.stress_std)
        nu_pred_p = self._denorm(nu_pred, self.nu_mean, self.nu_std)
        nu_true_p = self._denorm(y_nu, self.nu_mean, self.nu_std)

        # Derivative matching
        L_d1_stress = F.mse_loss(self._d1(stress_pred_p), self._d1(stress_true_p), reduction="mean") if cfg.w_d1_stress > 0 else stress_pred.new_tensor(0.0)
        L_d2_stress = F.mse_loss(self._d2(stress_pred_p), self._d2(stress_true_p), reduction="mean") if cfg.w_d2_stress > 0 else stress_pred.new_tensor(0.0)
        L_d1_nu = F.mse_loss(self._d1(nu_pred_p), self._d1(nu_true_p), reduction="mean") if cfg.w_d1_nu > 0 else stress_pred.new_tensor(0.0)
        L_d2_nu = F.mse_loss(self._d2(nu_pred_p), self._d2(nu_true_p), reduction="mean") if cfg.w_d2_nu > 0 else stress_pred.new_tensor(0.0)

        # Monotonicity penalty for stress (penalize negative slope)
        if cfg.w_mono_stress > 0:
            d1p = self._d1(stress_pred_p)
            L_mono = (F.relu(-d1p) ** 2).mean()
        else:
            L_mono = stress_pred.new_tensor(0.0)

        # Bounds on nu
        if cfg.w_bound_nu > 0:
            over = F.relu(nu_pred_p - cfg.nu_max)
            under = F.relu(cfg.nu_min - nu_pred_p)
            L_bound = (over ** 2 + under ** 2).mean()
        else:
            L_bound = stress_pred.new_tensor(0.0)

        total = total + (
            cfg.w_d1_stress * L_d1_stress +
            cfg.w_d2_stress * L_d2_stress +
            cfg.w_d1_nu * L_d1_nu +
            cfg.w_d2_nu * L_d2_nu +
            cfg.w_mono_stress * L_mono +
            cfg.w_bound_nu * L_bound
        )

        logs.update({
            "loss_total": total.detach(),
            "loss_d1_stress": L_d1_stress.detach(),
            "loss_d2_stress": L_d2_stress.detach(),
            "loss_d1_nu": L_d1_nu.detach(),
            "loss_d2_nu": L_d2_nu.detach(),
            "loss_mono_stress": L_mono.detach(),
            "loss_bound_nu": L_bound.detach(),
        })
        return total, logs

