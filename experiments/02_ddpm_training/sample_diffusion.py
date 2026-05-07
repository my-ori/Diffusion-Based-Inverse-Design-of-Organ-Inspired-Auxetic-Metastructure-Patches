# -*- coding: utf-8 -*-
"""
Generate new unit-cell images using a trained DDPM checkpoint.

"""
from __future__ import annotations

import os
import json
import argparse
from typing import List

import torch

from diffusion.unet import UNet, UNetConfig
from diffusion.ddpm import DDPM, DDPMConfig
from diffusion.ema import EMA
from diffusion.image_io import save_grid
from diffusion.utils import to_zero_one


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="Checkpoint .pt saved by train_diffusion.py")
    ap.add_argument("--out_dir", type=str, default="runs/ddpm/gen")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--num", type=int, default=200, help="How many valid samples to generate.")
    ap.add_argument("--num_batch", type=int, default=64, help="How many samples per diffusion sampling batch.")
    ap.add_argument("--timesteps", type=int, default=1000, help="Must match training.")
    ap.add_argument("--schedule", type=str, default="cosine", choices=["cosine", "linear"], help="Must match training.")
    ap.add_argument("--volfrac_min", type=float, default=0.30)
    ap.add_argument("--save_every", type=int, default=128, help="Write a grid PNG every N accepted samples.")
    return ap.parse_args()


def volfrac_from_xm11(xm11: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    """
    xm11: (B,1,H,W) in [-1,1]
    returns volfrac in [0,1] per sample: (B,)
    """
    solid = (xm11 > threshold).float()
    return solid.mean(dim=(1, 2, 3))


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    train_args = ckpt.get("args", {})
    base_ch = int(train_args.get("base_channels", 64))
    dropout = float(train_args.get("dropout", 0.0))

    unet = UNet(UNetConfig(
        in_channels=1,
        out_channels=1,
        base_channels=base_ch,
        channel_mults=(1, 2, 4),
        num_res_blocks=2,
        dropout=dropout,
        time_emb_dim=256,
    )).to(device)

    ddpm = DDPM(unet, DDPMConfig(timesteps=args.timesteps, schedule=args.schedule)).to(device)
    ddpm.load_state_dict(ckpt["ddpm"], strict=True)

    # Prefer EMA weights if available
    if "ema" in ckpt and ckpt["ema"] is not None:
        ema = EMA(decay=0.9999)
        ema.shadow = ckpt["ema"]
        ema.copy_to(ddpm)
        print("[EMA] using EMA weights for sampling")

    ddpm.eval()

    accepted: List[torch.Tensor] = []
    total_draws = 0

    while len(accepted) < args.num:
        with torch.no_grad():
            x = ddpm.sample(batch_size=args.num_batch, shape=(1, 50, 50), device=device)
        total_draws += x.shape[0]
        vf = volfrac_from_xm11(x, threshold=0.0)
        keep = vf >= args.volfrac_min
        if keep.any():
            accepted.append(x[keep].cpu())

        if sum(t.shape[0] for t in accepted) >= args.num:
            break

        if (sum(t.shape[0] for t in accepted) % args.save_every) < args.num_batch:
            # write an intermediate grid
            cur = torch.cat(accepted, dim=0)[:min(sum(t.shape[0] for t in accepted), 64)]
            save_grid(cur, os.path.join(args.out_dir, "preview.png"), nrow=8, binarize=True, threshold=0.0, upscale=6)
            print(f"[Progress] accepted={sum(t.shape[0] for t in accepted)} / {args.num}  (draws={total_draws})")

    samples = torch.cat(accepted, dim=0)[:args.num]  # (num,1,50,50)
    for i in range(0, args.num, 25):
        chunk = samples[i:i+25]
        save_grid(chunk, os.path.join(args.out_dir, f"grid_{i:04d}.png"), nrow=5, binarize=True, threshold=0.0, upscale=6)

    #save_grid(samples[:64], os.path.join(args.out_dir, "grid_64.png"), nrow=8, binarize=True, threshold=0.0, upscale=6)

    # Save the raw tensors too (so you can later label them using FEM)
    out_pt = os.path.join(args.out_dir, f"samples_{args.num}.pt")
    torch.save({"xm11": samples, "volfrac_min": args.volfrac_min}, out_pt)

    # Save binarized 0/1 as numpy
    x01 = (samples > 0.0).float()
    out_npy = os.path.join(args.out_dir, f"samples_{args.num}_bin01.npy")
    import numpy as np
    np.save(out_npy, x01.numpy())

    accept_rate = args.num / max(1, total_draws)
    meta = {
        "ckpt": args.ckpt,
        "num": args.num,
        "num_batch": args.num_batch,
        "timesteps": args.timesteps,
        "schedule": args.schedule,
        "volfrac_min": args.volfrac_min,
        "total_draws": total_draws,
        "accept_rate": accept_rate,
    }
    with open(os.path.join(args.out_dir, "sampling_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[Done] wrote {out_pt}, {out_npy}, and grid_64.png  | accept_rate={accept_rate:.3f}")


if __name__ == "__main__":
    main()
