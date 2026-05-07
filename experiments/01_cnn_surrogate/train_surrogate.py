#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_surrogate.py

Train a two-head CNN surrogate:
  input : (B,1,50,50) unit-cell binary mask
  output: stress curve (B,30) + poisson curve (B,30)

Example:
  python train_surrogate.py \
    --npz /mnt/c/Ori/data_collect/auxetic_unitcell_dataset.npz \
    --out_dir /mnt/c/Ori/data_collect/runs/surrogate_v1 \
    --epochs 200 --batch_size 128 --lr 3e-4 --num_workers 6
"""

from __future__ import annotations

import os
import json
import time
import math
import argparse
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

from data_loader_npz import make_dataloaders
from cnn_surrogate import AuxeticCNNSurrogate, SurrogateConfig
from losses import SurrogateLoss, LossConfig


# -------------------------
# Utils
# -------------------------
def seed_all(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_device(batch, device: torch.device):
    # batch is a dict from our Dataset
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _compute_r2_global(pred: torch.Tensor, true: torch.Tensor) -> float:
    """R² on flattened vector."""
    eps = 1e-12
    y = true.reshape(-1)
    yhat = pred.reshape(-1)
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return float((1.0 - ss_res / (ss_tot + eps)).item())


def _compute_r2_macro(pred: torch.Tensor, true: torch.Tensor) -> float:
    """Mean R² over dimensions (per strain-point R² averaged)."""
    eps = 1e-12
    y_mean = true.mean(dim=0, keepdim=True)
    ss_res_d = ((true - pred) ** 2).sum(dim=0)
    ss_tot_d = ((true - y_mean) ** 2).sum(dim=0)
    r2_dim = 1.0 - ss_res_d / (ss_tot_d + eps)
    return float(r2_dim.mean().item())


@torch.no_grad()
def evaluate_r2_physical(model: nn.Module, loader, stats, device: torch.device, amp: bool = True):
    """
    Compute R² in PHYSICAL space for one loader.
    Training uses normalized labels; we denormalize both pred and true here.
    """
    if loader is None:
        return None

    model.eval()
    stats_d = stats.to(device)

    stress_true_all = []
    stress_pred_all = []
    nu_true_all = []
    nu_pred_all = []

    for batch in loader:
        batch = to_device(batch, device)
        x = batch["x"]
        y_stress_n = batch["y_stress"]  # normalized
        y_nu_n = batch["y_nu"]          # normalized

        with autocast(enabled=amp):
            stress_pred_n, nu_pred_n = model(x)

        # Denormalize to physical space
        stress_true = y_stress_n * stats_d.stress_std.view(1, -1) + stats_d.stress_mean.view(1, -1)
        nu_true = y_nu_n * stats_d.nu_std.view(1, -1) + stats_d.nu_mean.view(1, -1)
        stress_pred = stress_pred_n * stats_d.stress_std.view(1, -1) + stats_d.stress_mean.view(1, -1)
        nu_pred = nu_pred_n * stats_d.nu_std.view(1, -1) + stats_d.nu_mean.view(1, -1)

        stress_true_all.append(stress_true.detach().cpu())
        stress_pred_all.append(stress_pred.detach().cpu())
        nu_true_all.append(nu_true.detach().cpu())
        nu_pred_all.append(nu_pred.detach().cpu())

    if len(stress_true_all) == 0:
        return None

    stress_true = torch.cat(stress_true_all, dim=0)
    stress_pred = torch.cat(stress_pred_all, dim=0)
    nu_true = torch.cat(nu_true_all, dim=0)
    nu_pred = torch.cat(nu_pred_all, dim=0)

    return {
        "stress": {
            "r2_global": _compute_r2_global(stress_pred, stress_true),
            "r2_macro": _compute_r2_macro(stress_pred, stress_true),
        },
        "nu": {
            "r2_global": _compute_r2_global(nu_pred, nu_true),
            "r2_macro": _compute_r2_macro(nu_pred, nu_true),
        },
        "n": int(stress_true.shape[0]),
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion: nn.Module, device: torch.device, amp: bool = True):
    model.eval()
    total_loss = 0.0
    total_stress = 0.0
    total_nu = 0.0
    total_d1_stress = 0.0
    total_d1_nu = 0.0
    n = 0

    for batch in loader:
        batch = to_device(batch, device)
        x = batch["x"]
        y_stress = batch["y_stress"]
        y_nu = batch["y_nu"]

        with autocast(enabled=amp):
            stress_pred, nu_pred = model(x)
            loss, logs = criterion(stress_pred, nu_pred, y_stress, y_nu)

        bs = x.shape[0]
        n += bs
        total_loss += float(logs["loss_total"]) * bs
        total_stress += float(logs["loss_stress"]) * bs
        total_nu += float(logs["loss_nu"]) * bs
        total_d1_stress += float(logs.get("loss_d1_stress", 0.0)) * bs
        total_d1_nu += float(logs.get("loss_d1_nu", 0.0)) * bs

    return {
        "loss_total": total_loss / max(n, 1),
        "loss_stress": total_stress / max(n, 1),
        "loss_nu": total_nu / max(n, 1),
        "loss_d1_stress": total_d1_stress / max(n, 1),
        "loss_d1_nu": total_d1_nu / max(n, 1),
    }



def save_checkpoint(path: str,
                    model: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    scaler: GradScaler,
                    epoch: int,
                    best_val: float,
                    info: dict):
    ckpt = {
        "epoch": epoch,
        "best_val": best_val,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "info": info,
    }
    torch.save(ckpt, path)


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, required=True, help="Path to auxetic_unitcell_dataset.npz")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory for logs/checkpoints")

    # Data
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--val_ratio", type=float, default=0.05)
    ap.add_argument("--test_ratio", type=float, default=0.10,
                    help="Within the *seen* classes, hold out this fraction as an extra test set (testset2). "
                         "Set to 0 to disable.")
    ap.add_argument("--holdout_classes", type=str, default="sinusoidal,anti_chiral_iso",
                    help="Comma-separated class names to HOLD OUT as unseen test set. "
                         "Example: sinusoidal,anti_chiral_iso. Use empty string for none.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=12)

    ap.add_argument("--volfrac_min", type=float, default=None,
                help="Filter dataset by volume fraction (solid pixel ratio): keep samples with vf >= volfrac_min.")
    ap.add_argument("--volfrac_max", type=float, default=None,
                help="Filter dataset by volume fraction (solid pixel ratio): keep samples with vf <= volfrac_max.")


    # Model
    ap.add_argument("--base_channels", type=int, default=32)
    ap.add_argument("--blocks", type=str, default="2,2,2", help="blocks_per_stage, e.g. 2,2,2")
    ap.add_argument("--head_hidden_dim", type=int, default=256)
    ap.add_argument("--dropout_head", type=float, default=0.1)

    # Optim
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    # Scheduler
    ap.add_argument("--scheduler", type=str, default="cosine", choices=["none", "cosine", "step"])
    ap.add_argument("--warmup_epochs", type=int, default=5)
    ap.add_argument("--step_size", type=int, default=50)
    ap.add_argument("--gamma", type=float, default=0.5)

    # Mixed precision
    ap.add_argument("--no_amp", action="store_true")

    # Save freq
    ap.add_argument("--save_every", type=int, default=200)

    args = ap.parse_args()
    # Hold out some classes completely for "unseen class" testing
    holdout_classes = [s.strip() for s in (args.holdout_classes or "").split(",") if s.strip()]
    ensure_dir(args.out_dir)

    # Repro
    seed_all(args.seed)
    torch.backends.cudnn.benchmark = True  # speed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = (not args.no_amp) and (device.type == "cuda")

    # Dataloaders (normalized y)
    train_loader, val_loader, test1_loader, test2_loader, stats, info = make_dataloaders(
        npz_path=args.npz,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        test_ratio_seen=args.test_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        normalize_y=True,          # you chose normalized training
        return_concat_y=False,
        holdout_class_names=holdout_classes if len(holdout_classes) else None,
        return_test_loader=True,
        return_seen_test_loader=True,
        volfrac_min=args.volfrac_min,
        volfrac_max=args.volfrac_max,
    )

    # Split summary
    if info.get("holdout_class_names"):
        print("Holdout (unseen testset1) classes:", info["holdout_class_names"], "| num_test1:", info.get("num_test_holdout", info.get("num_test", 0)))
    print("Train/Val/Test2 sizes (seen classes):",
          info.get("num_train", 0), "/", info.get("num_val", 0), "/", info.get("num_test_seen", 0),
          "| test_ratio:", info.get("test_ratio_seen", args.test_ratio))
    if info.get("volfrac_filter") is not None:
        print("Volume-fraction filter:", info.get("volfrac_filter"), "| kept:", info.get("num_kept_after_volfrac_filter"), "| stats:", info.get("volfrac_stats_kept"))

    # Model
    blocks = tuple(int(x.strip()) for x in args.blocks.split(","))
    cfg = SurrogateConfig(
        in_channels=1,
        base_channels=args.base_channels,
        blocks_per_stage=blocks,
        act="relu",
        dropout_backbone=0.0,
        dropout_head=args.dropout_head,
        head_hidden_dim=args.head_hidden_dim,
        out_stress_dim=30,
        out_nu_dim=30,
    )
    model = AuxeticCNNSurrogate(cfg).to(device)

    # Loss
    loss_cfg = LossConfig(w_stress=1.0, w_nu=3.0, w_d1_stress=0.2, w_d1_nu=0.1)
    criterion = SurrogateLoss(
        loss_cfg,
        stress_mean=stats.stress_mean,
        stress_std=stats.stress_std,
        nu_mean=stats.nu_mean,
        nu_std=stats.nu_std,
    ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Scheduler
    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)

    def lr_lambda(step: int):
        if args.scheduler == "none":
            return 1.0
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        # cosine decay to 0.1 of initial lr
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        if args.scheduler == "cosine":
            return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
        return 1.0

    if args.scheduler in ["cosine", "none"]:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # AMP
    scaler = GradScaler(enabled=amp)

    # Save config
    run_cfg = {
        "args": vars(args),
        "device": str(device),
        "amp": amp,
        "model_params": count_params(model),
        "data_info": info,
        "model_cfg": asdict(cfg),
        "loss_cfg": asdict(loss_cfg),
        # Save normalization stats for later denormalization during evaluation/plotting
        "norm_stats": {
            "stress_mean": stats.stress_mean.cpu().numpy().tolist(),
            "stress_std": stats.stress_std.cpu().numpy().tolist(),
            "nu_mean": stats.nu_mean.cpu().numpy().tolist(),
            "nu_std": stats.nu_std.cpu().numpy().tolist(),
        },
    }
    save_json(os.path.join(args.out_dir, "config.json"), run_cfg)

    # Training loop
    best_val = float("inf")
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_stress = 0.0
        epoch_nu = 0.0
        epoch_d1_stress = 0.0
        epoch_d1_nu = 0.0

        n = 0

        for step, batch in enumerate(train_loader, start=1):
            batch = to_device(batch, device)
            x = batch["x"]
            y_stress = batch["y_stress"]
            y_nu = batch["y_nu"]

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=amp):
                stress_pred, nu_pred = model(x)
                loss, logs = criterion(stress_pred, nu_pred, y_stress, y_nu)

            scaler.scale(loss).backward()

            if args.grad_clip and args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            if args.scheduler in ["cosine", "none"]:
                scheduler.step()

            bs = x.shape[0]
            n += bs
            epoch_loss += float(logs["loss_total"]) * bs
            epoch_stress += float(logs["loss_stress"]) * bs
            epoch_nu += float(logs["loss_nu"]) * bs
            epoch_d1_stress += float(logs.get("loss_d1_stress", 0.0)) * bs
            epoch_d1_nu += float(logs.get("loss_d1_nu", 0.0)) * bs



        if args.scheduler == "step":
            scheduler.step()

        train_metrics = {
            "loss_total": epoch_loss / max(n, 1),
            "loss_stress": epoch_stress / max(n, 1),
            "loss_nu": epoch_nu / max(n, 1),
            "loss_d1_stress": epoch_d1_stress / max(n, 1),
            "loss_d1_nu": epoch_d1_nu / max(n, 1),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }

        val_metrics = evaluate(model, val_loader, criterion, device, amp=amp)

        # Logging to stdout
        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train: {train_metrics['loss_total']:.7f} "
            f"(s:{train_metrics['loss_stress']:.7f}, n:{train_metrics['loss_nu']:.7f}, "
            f"d1s:{train_metrics['loss_d1_stress']:.7f}, d1n:{train_metrics['loss_d1_nu']:.7f}) | "
            f"val: {val_metrics['loss_total']:.7f} "
            f"(s:{val_metrics['loss_stress']:.7f}, n:{val_metrics['loss_nu']:.7f}, "
            f"d1s:{val_metrics['loss_d1_stress']:.7f}, d1n:{val_metrics['loss_d1_nu']:.7f}) | "
            f"lr {train_metrics['lr']:.2e} | "
            f"time {elapsed/60.0:.1f} min"
        )

        # Append metrics to a jsonl log
        log_row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        with open(os.path.join(args.out_dir, "metrics.jsonl"), "a") as f:
            f.write(json.dumps(log_row) + "\n")

        # Checkpointing
        if val_metrics["loss_total"] < best_val:
            best_val = val_metrics["loss_total"]
            save_checkpoint(
                os.path.join(args.out_dir, "best.pt"),
                model, optimizer, scaler, epoch, best_val,
                info={"train": train_metrics, "val": val_metrics}
            )

        if (epoch % args.save_every) == 0 or epoch == args.epochs:
            save_checkpoint(
                os.path.join(args.out_dir, f"epoch_{epoch:03d}.pt"),
                model, optimizer, scaler, epoch, best_val,
                info={"train": train_metrics, "val": val_metrics}
            )

    print("Done. Best val loss:", best_val)

    # --- R² report on testset2 (seen classes) and testset1 (holdout classes) using BEST checkpoint ---
    best_path = os.path.join(args.out_dir, "best.pt")
    if os.path.isfile(best_path):
        ckpt = torch.load(best_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"], strict=True)
        model.to(device)
        model.eval()

        r2_val = evaluate_r2_physical(model, val_loader, stats, device, amp=amp)
        r2_test2 = evaluate_r2_physical(model, test2_loader, stats, device, amp=amp) if (args.test_ratio and args.test_ratio > 0) else None
        r2_test1 = evaluate_r2_physical(model, test1_loader, stats, device, amp=amp) if (test1_loader is not None) else None

        report = {
            "val_seen": r2_val,
            "testset2_seen": r2_test2,
            "testset1_holdout": r2_test1,
        }
        save_json(os.path.join(args.out_dir, "r2_report_best.json"), report)

        if r2_test2 is not None:
            print("\n=== BEST checkpoint | Testset2 (seen classes) | PHYSICAL-space R² ===")
            print("stress:", r2_test2["stress"], "| nu:", r2_test2["nu"], "| n:", r2_test2["n"])
        if r2_test1 is not None and (info.get("num_test_holdout", 0) > 0):
            print("\n=== BEST checkpoint | Testset1 (holdout classes) | PHYSICAL-space R² ===")
            print("stress:", r2_test1["stress"], "| nu:", r2_test1["nu"], "| n:", r2_test1["n"])
        print("\nSaved R² report:", os.path.join(args.out_dir, "r2_report_best.json"))


if __name__ == "__main__":
    main()
