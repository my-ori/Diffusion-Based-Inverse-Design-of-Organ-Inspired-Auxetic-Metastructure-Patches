# -*- coding: utf-8 -*-
"""
Exponential Moving Average (EMA) for model weights (often improves sampling quality).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class EMA:
    decay: float = 0.9999
    model: Optional[torch.nn.Module] = None
    shadow: Optional[dict] = None

    def register(self, model: torch.nn.Module) -> None:
        self.model = model
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        if self.shadow is None:
            self.register(model)
            return
        for k, v in model.state_dict().items():
            if k not in self.shadow:
                self.shadow[k] = v.detach().clone()
            else:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=(1.0 - self.decay))

    def copy_to(self, model: torch.nn.Module) -> None:
        if self.shadow is None:
            raise RuntimeError("EMA not initialized. Call register() first.")
        model.load_state_dict(self.shadow, strict=True)
