# -*- coding: utf-8 -*-
"""
Simple image saving helpers without requiring torchvision.
"""
from __future__ import annotations

from typing import Tuple, Optional
import math
import numpy as np
from PIL import Image

import torch

from .utils import to_zero_one


def _to_uint8(x01: torch.Tensor) -> np.ndarray:
    x01 = x01.clamp(0, 1)
    x_u8 = (x01 * 255.0).round().to(torch.uint8)
    return x_u8.cpu().numpy()


def save_grid(
    xm11: torch.Tensor,
    path: str,
    nrow: int = 8,
    binarize: bool = True,
    threshold: float = 0.0,
    upscale: int = 6,
) -> None:
    """
    Save a grid PNG.
      xm11: (B,1,H,W) in [-1,1]
      threshold: threshold in [-1,1] space, default 0.0 corresponds to 0.5 in [0,1]
    """
    b, c, h, w = xm11.shape
    assert c == 1, "Expected 1-channel images."
    x01 = to_zero_one(xm11)

    if binarize:
        xm = (xm11 > threshold).float()
        x01 = to_zero_one(xm)

    nrow = max(1, int(nrow))
    ncol = int(math.ceil(b / nrow))
    canvas = np.zeros((ncol * h, nrow * w), dtype=np.uint8)

    imgs = _to_uint8(x01[:, 0])  # (B,H,W)

    for i in range(b):
        r = i // nrow
        c0 = i % nrow
        canvas[r*h:(r+1)*h, c0*w:(c0+1)*w] = imgs[i]

    im = Image.fromarray(canvas, mode="L")
    if upscale > 1:
        im = im.resize((canvas.shape[1] * upscale, canvas.shape[0] * upscale), resample=Image.NEAREST)
    im.save(path)
