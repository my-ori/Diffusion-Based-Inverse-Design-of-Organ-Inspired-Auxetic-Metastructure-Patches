# -*- coding: utf-8 -*-
"""
Train an unconditional DDPM diffusion model on your 50x50 unit-cell images.

"""
from __future__ import annotations
import numpy as np
import os
import time
import json
import argparse
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader

from diffusion.utils import set_seed
from diffusion.dataset import VolumeFractionFilteredDataset
from diffusion.unet import UNet, UNetConfig
from diffusion.ddpm import DDPM, DDPMConfig
from diffusion.ema import EMA
from diffusion.image_io import save_grid


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_path", type=str, required=True, help="Path to the .npz dataset (must contain X field).")
    ap.add_argument("--out_dir", type=str, default="runs/ddpm", help="Output directory for checkpoints and samples.")
    ap.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    ap.add_argument("--seed", type=int, default=0)

    # Data
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=12)
    ap.add_argument("--volfrac_min", type=float, default=0.30)
    ap.add_argument("--no_augment", action="store_true", help="Disable random flips.")
    ap.add_argument("--pin_memory", action="store_true", help="Enable DataLoader pin_memory.")

    # Diffusion
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--t_sampling", type=str, default="uniform", choices=["uniform","early"],
                    help="How to sample diffusion timesteps during training. early biases to small t (better for structure learning).")
    ap.add_argument("--schedule", type=str, default="cosine", choices=["cosine", "linear"])
    ap.add_argument("--base_channels", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.0)

    # Optim
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    # Train loop
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=1, help="Save checkpoint every N epochs.")
    ap.add_argument("--sample_every", type=int, default=50, help="Write sample grid every N epochs.")
    ap.add_argument("--num_sample", type=int, default=64)
    ap.add_argument("--sample_nrow", type=int, default=8)
    ap.add_argument(
        "--holdout_classes", "--exclude_classes",
        nargs="*", default=[],
        help="Class names to EXCLUDE from training (e.g. double_sin anti_chiral_iso lozenge_chiral)."
    )
    # Resume
    ap.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from.")
    return ap.parse_args()


def sample_timesteps(T: int, bsz: int, device: torch.device, method: str = "uniform") -> torch.Tensor:
    """
    Sample diffusion timesteps for training.

    - uniform: standard DDPM training
    - early: bias toward small t to avoid the "identity solution" where x_t≈noise for most t
    """
    if method == "uniform":
        return torch.randint(0, T, (bsz,), device=device, dtype=torch.long)
    if method == "early":
        # u^2 biases toward 0 (small t)
        u = torch.rand((bsz,), device=device)
        t = torch.floor((u ** 2) * T).to(torch.long)
        return torch.clamp(t, 0, T - 1)
    raise ValueError(f"Unknown t_sampling method: {method}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "samples"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "checkpoints"), exist_ok=True)

    # Dataset
    ds = VolumeFractionFilteredDataset(
        args.npz_path,
        volfrac_min=args.volfrac_min,
        augment_flip=(not args.no_augment),
        return_meta=False,
        holdout_class_names=args.holdout_classes,
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True,
    )

    # Model + diffusion
    unet = UNet(UNetConfig(
        in_channels=1,
        out_channels=1,
        base_channels=args.base_channels,
        channel_mults=(1, 2, 4),
        num_res_blocks=2,
        dropout=args.dropout,
        time_emb_dim=256,
    )).to(device)

    ddpm = DDPM(unet, DDPMConfig(timesteps=args.timesteps, schedule=args.schedule)).to(device)

    opt = torch.optim.AdamW(ddpm.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ema = EMA(decay=0.9999)
    ema.register(ddpm)

    start_epoch = 0
    global_step = 0

    # Save dataset + config info
    with open(os.path.join(args.out_dir, "dataset_info.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(ds.info), f, indent=2)

    cfg_dump = {"args": vars(args)}
    with open(os.path.join(args.out_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg_dump, f, indent=2)

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        ddpm.load_state_dict(ckpt["ddpm"], strict=True)
        opt.load_state_dict(ckpt["opt"])
        if "ema" in ckpt and ckpt["ema"] is not None:
            ema.shadow = ckpt["ema"]
        start_epoch = int(ckpt.get("epoch", 0) + 1)
        global_step = int(ckpt.get("global_step", 0))
        print(f"[Resume] epoch={start_epoch}, global_step={global_step} from {args.resume}")

    print(f"[Device] {device}")
    print(f"[Data] kept {ds.info.num_kept}/{ds.info.num_total} samples with volfrac >= {ds.info.volfrac_min:.2f}")
    print(f"       kept volfrac: mean={ds.info.volfrac_mean_kept:.3f}, min={ds.info.volfrac_min_kept:.3f}, max={ds.info.volfrac_max_kept:.3f}")
    # ---- Per-class breakdown (after filtering) ----
    try:
        class_ids_all = np.asarray(ds.base.class_ids, dtype=np.int64)   # (N,)
        class_names = list(ds.base.class_names)                         # (C,)
        kept_idx = np.asarray(ds.keep_idx, dtype=np.int64)              # (K,)
        kept_class_ids = class_ids_all[kept_idx]                        # (K,)

        # bincount assumes ids are >= 0; ensure minlength covers all classes
        minlength = max(len(class_names), int(class_ids_all.max()) + 1)
        counts = np.bincount(kept_class_ids, minlength=minlength)

        print("[Data] kept samples per class (after all filters):")
        for cid, name in enumerate(class_names):
            cnt = int(counts[cid]) if cid < counts.size else 0
            print(f"   - {name} (id={cid}): {cnt}")

        # If you implemented holdout info in ds.info, print it too (optional)
        hold_names = getattr(ds.info, "holdout_class_names", None)
        if hold_names:
            print(f"[Data] held-out classes (excluded): {', '.join(map(str, hold_names))}")

    except Exception as e:
        print(f"[Data] per-class breakdown unavailable: {e}")
    # ----------------------------------------------

    for epoch in range(start_epoch, args.epochs):
        ddpm.train()
        t0 = time.time()
        running = 0.0

        for it, batch in enumerate(dl):
            x = batch["x"].to(device)  # (B,1,50,50) in [-1,1]
            bsz = x.shape[0]
            t = sample_timesteps(args.timesteps, bsz, device, method=args.t_sampling)

            loss = ddpm.p_losses(x, t)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(ddpm.parameters(), args.grad_clip)
            opt.step()
            ema.update(ddpm)

            running += float(loss.item())
            global_step += 1

            if (global_step % args.log_every) == 0:
                avg = running / args.log_every
                running = 0.0
                print(f"epoch {epoch:03d} step {global_step:07d} loss {avg:.6f}")

        dt = time.time() - t0
        print(f"[Epoch {epoch}] done in {dt:.1f}s")

        # Sampling (use EMA weights)
        if (epoch % args.sample_every) == 0:
            ddpm_eval = DDPM(unet.__class__(unet.cfg), DDPMConfig(timesteps=args.timesteps, schedule=args.schedule)).to(device)
            # copy EMA weights into ddpm_eval
            ema.copy_to(ddpm_eval)
            ddpm_eval.eval()

            with torch.no_grad():
                samples = ddpm_eval.sample(
                    batch_size=args.num_sample,
                    shape=(1, 50, 50),
                    device=device,
                )
                        # Save grayscale AND binarized previews (binarized can look like noise early on)
            out_png_gray = os.path.join(args.out_dir, "samples", f"epoch_{epoch:03d}_gray.png")
            out_png_bin  = os.path.join(args.out_dir, "samples", f"epoch_{epoch:03d}_bin.png")
            # grayscale
            save_grid(samples, out_png_gray, nrow=args.sample_nrow, binarize=False, threshold=0.0, upscale=6)
            # binarized at 0.0 (equivalent to 0.5 in [0,1])
            save_grid(samples, out_png_bin, nrow=args.sample_nrow, binarize=True, threshold=0.0, upscale=6)
            # quick stats
            vf = (samples > 0.0).float().mean(dim=(1,2,3)).detach().cpu()
            print(f"[Sample] wrote {out_png_gray} and {out_png_bin} | vf(mean/min/max)={vf.mean():.3f}/{vf.min():.3f}/{vf.max():.3f}")

        # Checkpoint
        if (epoch % args.save_every) == 0:
            ckpt_path = os.path.join(args.out_dir, "checkpoints", f"ddpm_epoch_{epoch:03d}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "ddpm": ddpm.state_dict(),
                    "opt": opt.state_dict(),
                    "ema": ema.shadow,
                    "args": vars(args),
                    "dataset_info": asdict(ds.info),
                },
                ckpt_path,
            )
            print(f"[Checkpoint] saved {ckpt_path}")

    print("Training finished.")


if __name__ == "__main__":
    main()
