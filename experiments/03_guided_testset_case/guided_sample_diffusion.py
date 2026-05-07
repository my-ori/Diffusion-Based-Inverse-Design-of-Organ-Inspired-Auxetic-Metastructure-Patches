# -*- coding: utf-8 -*-
"""guided_sample_diffusion.py

Surrogate-guided sampling (gradient guidance) for your *unconditional* DDPM.


Key idea:
  - Keep unconditional diffusion as a strong prior (realistic designs)
  - Use CNN surrogate as an energy model during sampling

Target JSON formats supported:
  - Stress + nu (preferred):
      {"stress": [...30...], "nu": [...30...]}
  - Stress only:
      {"stress": [...30...]}
  - Nu only:
      {"nu": [...30...]}
  - Legacy keys also accepted:
      {"y_stress": [...], "y_nu": [...]}, or only one of them.

When only one target is provided, guidance/loss automatically ignores the missing one.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional

import torch
import torch.nn.functional as F

# -------------------------
# Robust imports (project vs flat)
# -------------------------
try:
    from diffusion.unet import UNet, UNetConfig
    from diffusion.ddpm import DDPM, DDPMConfig
    from diffusion.ema import EMA
    from diffusion.utils import extract, to_zero_one
    from diffusion.image_io import save_grid
except Exception:  # pragma: no cover
    from unet import UNet, UNetConfig
    from ddpm import DDPM, DDPMConfig
    from ema import EMA
    from utils import extract, to_zero_one
    from image_io import save_grid

try:
    from models.cnn_surrogate import AuxeticCNNSurrogate, SurrogateConfig
except Exception:  # pragma: no cover
    from cnn_surrogate import AuxeticCNNSurrogate, SurrogateConfig


# -------------------------
# Normalization stats
# -------------------------
@dataclass
class NormStats:
    stress_mean: torch.Tensor  # (30,)
    stress_std: torch.Tensor   # (30,)
    nu_mean: torch.Tensor      # (30,)
    nu_std: torch.Tensor       # (30,)

    @staticmethod
    def from_config_json(cfg: Dict[str, Any], device: torch.device) -> "NormStats":
        ns = cfg["norm_stats"]

        def _t(key: str) -> torch.Tensor:
            return torch.tensor(ns[key], dtype=torch.float32, device=device)

        return NormStats(
            stress_mean=_t("stress_mean"),
            stress_std=_t("stress_std"),
            nu_mean=_t("nu_mean"),
            nu_std=_t("nu_std"),
        )

    def normalize_stress(self, stress: torch.Tensor) -> torch.Tensor:
        return (stress - self.stress_mean) / self.stress_std

    def normalize_nu(self, nu: torch.Tensor) -> torch.Tensor:
        return (nu - self.nu_mean) / self.nu_std

    def denormalize(self, stress_n: torch.Tensor, nu_n: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return stress_n * self.stress_std + self.stress_mean, nu_n * self.nu_std + self.nu_mean


# -------------------------
# CLI
# -------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    # Required
    ap.add_argument("--ddpm_ckpt", type=str, required=True, help="DDPM checkpoint, e.g., ddpm_epoch_2950.pt")
    ap.add_argument("--cnn_ckpt", type=str, required=True, help="CNN surrogate checkpoint, e.g., best.pt")
    ap.add_argument("--cnn_cfg", type=str, required=True, help="CNN surrogate config.json (contains norm_stats)")
    ap.add_argument("--target_json", type=str, required=True, help="Target property JSON with stress/nu arrays (len 30)")

    # Output
    ap.add_argument("--out_dir", type=str, default="runs/ddpm/guided")

    # Sampling
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num", type=int, default=16, help="How many final designs to output (best-by-loss among the pool).")
    ap.add_argument("--num_batch", type=int, default=32, help="Batch size for diffusion sampling.")
    ap.add_argument("--pool_mult", type=int, default=3, help="Oversample factor; keep top-N by property loss.")
    ap.add_argument("--volfrac_min", type=float, default=0.30, help="Hard filter after binarization.")
    ap.add_argument("--binarize_threshold", type=float, default=0.0, help="Threshold in [-1,1] space (0.0 == 0.5 in [0,1])." )

    # Use training values by default (recommended)
    ap.add_argument("--timesteps", type=int, default=-1, help="Override timesteps (must match training). -1=use checkpoint args")
    ap.add_argument("--schedule", type=str, default="", choices=["", "cosine", "linear"], help="Override schedule. empty=use checkpoint args")

    # Guidance
    ap.add_argument("--target_mode", type=str, default="physical", choices=["physical", "normalized"],
                    help="If physical, target is in original units and will be normalized using cnn_cfg norm_stats.")
    ap.add_argument("--guidance_scale", type=float, default=2.0, help="Overall strength of gradient guidance.")
    ap.add_argument("--guidance_start_frac", type=float, default=0.30,
                    help="Apply guidance only for the last this-fraction of timesteps (e.g., 0.30 => last 30%).")
    ap.add_argument("--guidance_power", type=float, default=2.0,
                    help="Ramp schedule: scale_t = scale * w^power where w goes 0->1 as t goes start->0")
    ap.add_argument("--guidance_every", type=int, default=1, help="Apply guidance every N steps (>=1).")
    ap.add_argument("--grad_clip", type=float, default=1.0, help="Clip guidance gradient by global L2 norm (0 disables).")

    # Loss weights (only used if the corresponding target exists)
    ap.add_argument("--w_stress", type=float, default=1.0)
    ap.add_argument("--w_nu", type=float, default=1.0)
    ap.add_argument("--volfrac_lambda", type=float, default=0.0,
                    help="Optional soft penalty weight to encourage volfrac >= volfrac_min during guidance.")

    # Saving
    ap.add_argument("--save_every", type=int, default=16, help="Write a grid PNG every N kept samples.")
    ap.add_argument("--save_nrow", type=int, default=4)
    return ap.parse_args()


def set_seed(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# Loading helpers
# -------------------------
def load_ddpm(ckpt_path: str, device: torch.device, timesteps: int, schedule: str, use_ema: bool = True) -> Tuple[DDPM, Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
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

    ddpm = DDPM(unet, DDPMConfig(timesteps=timesteps, schedule=schedule)).to(device)
    ddpm.load_state_dict(ckpt["ddpm"], strict=True)

    if use_ema and ("ema" in ckpt) and (ckpt["ema"] is not None):
        ema = EMA(decay=0.9999)
        ema.shadow = ckpt["ema"]
        ema.copy_to(ddpm)
        print("[EMA] using EMA weights")

    ddpm.eval()
    for p in ddpm.parameters():
        p.requires_grad_(False)
    return ddpm, ckpt


def load_cnn(cnn_ckpt: str, cnn_cfg_json: str, device: torch.device) -> Tuple[AuxeticCNNSurrogate, NormStats, Dict[str, Any]]:
    with open(cnn_cfg_json, "r", encoding="utf-8") as f:
        cfgj = json.load(f)

    model_cfg = dict(cfgj.get("model_cfg", {}))
    if "blocks_per_stage" in model_cfg and isinstance(model_cfg["blocks_per_stage"], list):
        model_cfg["blocks_per_stage"] = tuple(int(x) for x in model_cfg["blocks_per_stage"])

    cfg = SurrogateConfig(**model_cfg)
    cnn = AuxeticCNNSurrogate(cfg).to(device)

    ckpt = torch.load(cnn_ckpt, map_location="cpu")
    cnn.load_state_dict(ckpt["model_state"], strict=True)
    cnn.eval()
    for p in cnn.parameters():
        p.requires_grad_(False)

    ns = NormStats.from_config_json(cfgj, device=device)
    return cnn, ns, cfgj


def _get_optional_key(d: Dict[str, Any], keys: Tuple[str, str]) -> Optional[Any]:
    if keys[0] in d:
        return d[keys[0]]
    if keys[1] in d:
        return d[keys[1]]
    return None


def load_target(target_json: str, device: torch.device) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, Any]]:
    with open(target_json, "r", encoding="utf-8") as f:
        d = json.load(f)

    stress = _get_optional_key(d, ("stress", "y_stress"))
    nu = _get_optional_key(d, ("nu", "y_nu"))

    stress_t = None
    nu_t = None

    if stress is not None:
        stress_t = torch.tensor(stress, dtype=torch.float32, device=device)
        if stress_t.numel() != 30:
            raise ValueError(f"stress target must have length 30. Got {stress_t.numel()}")

    if nu is not None:
        nu_t = torch.tensor(nu, dtype=torch.float32, device=device)
        if nu_t.numel() != 30:
            raise ValueError(f"nu target must have length 30. Got {nu_t.numel()}")

    if stress_t is None and nu_t is None:
        raise ValueError("target_json must contain at least one of: 'stress'/'y_stress' or 'nu'/'y_nu'.")

    return stress_t, nu_t, d


# -------------------------
# Guidance / sampling
# -------------------------
def volfrac_soft_from_x01(x01: torch.Tensor) -> torch.Tensor:
    return x01.mean(dim=(1, 2, 3))


def volfrac_hard_from_xm11(xm11: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    return (xm11 > threshold).float().mean(dim=(1, 2, 3))


def guidance_scale_for_step(i: int, start_t: int, base_scale: float, power: float) -> float:
    if base_scale <= 0:
        return 0.0
    if i > start_t:
        return 0.0
    w = float(start_t - i) / float(max(1, start_t))
    return float(base_scale) * (w ** float(power))


def guided_p_sample(
    ddpm: DDPM,
    x: torch.Tensor,
    t: torch.Tensor,
    *,
    cnn: AuxeticCNNSurrogate,
    target_stress_n: Optional[torch.Tensor],
    target_nu_n: Optional[torch.Tensor],
    scale: float,
    w_stress: float,
    w_nu: float,
    volfrac_min: float,
    volfrac_lambda: float,
    grad_clip: float,
    binarize_threshold: float,
) -> torch.Tensor:
    # ----- Standard DDPM mean (no grad through UNet) -----
    with torch.no_grad():
        eps = ddpm.model(x, t)
        betas_t = extract(ddpm.betas, t, x.shape)
        sqrt_one_minus_acp_t = extract(ddpm.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = extract(ddpm.sqrt_recip_alphas, t, x.shape)
        model_mean = sqrt_recip_alphas_t * (x - betas_t * eps / sqrt_one_minus_acp_t)

    # ----- Guidance term (grad wrt x only) -----
    if scale > 0.0 and (w_stress > 0.0 or w_nu > 0.0):
        x_in = x.detach().requires_grad_(True)

        sqrt_acp_t = extract(ddpm.sqrt_alphas_cumprod, t, x.shape)
        sqrt_om_t = extract(ddpm.sqrt_one_minus_alphas_cumprod, t, x.shape)
        x0_hat = (x_in - sqrt_om_t * eps.detach()) / sqrt_acp_t

        x01_hat = to_zero_one(x0_hat).clamp(0.0, 1.0)
        stress_pred_n, nu_pred_n = cnn(x01_hat)

        loss_terms = []
        if target_stress_n is not None and w_stress > 0.0:
            loss_stress = F.mse_loss(
                stress_pred_n,
                target_stress_n.view(1, -1).expand_as(stress_pred_n),
                reduction="mean",
            )
            loss_terms.append(float(w_stress) * loss_stress)

        if target_nu_n is not None and w_nu > 0.0:
            loss_nu = F.mse_loss(
                nu_pred_n,
                target_nu_n.view(1, -1).expand_as(nu_pred_n),
                reduction="mean",
            )
            loss_terms.append(float(w_nu) * loss_nu)

        if len(loss_terms) > 0:
            loss = sum(loss_terms)

            if volfrac_lambda > 0.0:
                vf_soft = volfrac_soft_from_x01(x01_hat)
                loss_vf = F.relu(torch.tensor(volfrac_min, device=vf_soft.device) - vf_soft).pow(2).mean()
                loss = loss + float(volfrac_lambda) * loss_vf

            grad = torch.autograd.grad(loss, x_in, retain_graph=False, create_graph=False)[0]

            if grad_clip and grad_clip > 0:
                g = grad.flatten(1)
                gn = torch.norm(g, dim=1, keepdim=True).clamp(min=1e-12)
                clip = float(grad_clip)
                factor = torch.clamp(clip / gn, max=1.0)
                grad = grad * factor.view(-1, 1, 1, 1)

            model_mean = model_mean - float(scale) * grad.detach()

    if (t == 0).all():
        return model_mean.detach()

    with torch.no_grad():
        posterior_var_t = extract(ddpm.posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        x_prev = model_mean + torch.sqrt(posterior_var_t) * noise
        return x_prev.detach()


def sample_guided_batch(
    ddpm: DDPM,
    *,
    cnn: AuxeticCNNSurrogate,
    target_stress_n: Optional[torch.Tensor],
    target_nu_n: Optional[torch.Tensor],
    batch_size: int,
    device: torch.device,
    guidance_scale: float,
    guidance_start_frac: float,
    guidance_power: float,
    guidance_every: int,
    w_stress: float,
    w_nu: float,
    volfrac_min: float,
    volfrac_lambda: float,
    grad_clip: float,
    binarize_threshold: float,
) -> torch.Tensor:
    B = int(batch_size)
    x = torch.randn((B, 1, 50, 50), device=device)

    T = int(ddpm.cfg.timesteps)
    start_t = int(round((T - 1) * float(guidance_start_frac)))
    guidance_every = max(1, int(guidance_every))

    for i in reversed(range(T)):
        t = torch.full((B,), i, device=device, dtype=torch.long)
        if (i % guidance_every) == 0:
            scale_i = guidance_scale_for_step(i, start_t=start_t, base_scale=guidance_scale, power=guidance_power)
        else:
            scale_i = 0.0
        x = guided_p_sample(
            ddpm,
            x,
            t,
            cnn=cnn,
            target_stress_n=target_stress_n,
            target_nu_n=target_nu_n,
            scale=scale_i,
            w_stress=w_stress,
            w_nu=w_nu,
            volfrac_min=volfrac_min,
            volfrac_lambda=volfrac_lambda,
            grad_clip=grad_clip,
            binarize_threshold=binarize_threshold,
        )
    return x


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    cnn, ns, _cnn_cfgj = load_cnn(args.cnn_ckpt, args.cnn_cfg, device=device)

    tgt_stress, tgt_nu, tgt_raw = load_target(args.target_json, device=device)
    use_stress = tgt_stress is not None
    use_nu = tgt_nu is not None

    w_stress_eff = float(args.w_stress) if use_stress else 0.0
    w_nu_eff = float(args.w_nu) if use_nu else 0.0
    if w_stress_eff == 0.0 and w_nu_eff == 0.0:
        raise ValueError("Both effective weights are 0. Provide at least one target and set its weight > 0.")

    if args.target_mode == "physical":
        tgt_stress_n = ns.normalize_stress(tgt_stress) if use_stress else None
        tgt_nu_n = ns.normalize_nu(tgt_nu) if use_nu else None
    else:
        tgt_stress_n = tgt_stress if use_stress else None
        tgt_nu_n = tgt_nu if use_nu else None

    ckpt_tmp = torch.load(args.ddpm_ckpt, map_location="cpu")
    ckpt_args = ckpt_tmp.get("args", {})
    timesteps = int(ckpt_args.get("timesteps", 1000)) if args.timesteps < 0 else int(args.timesteps)
    schedule = str(ckpt_args.get("schedule", "cosine")) if args.schedule == "" else str(args.schedule)
    del ckpt_tmp

    ddpm, _ = load_ddpm(args.ddpm_ckpt, device=device, timesteps=timesteps, schedule=schedule, use_ema=True)
    print(f"[DDPM] timesteps={timesteps} schedule={schedule}")
    print(f"[Target] use_stress={use_stress} use_nu={use_nu} w_stress={w_stress_eff} w_nu={w_nu_eff}")

    want = int(args.num)
    pool_target = want * max(1, int(args.pool_mult))
    kept_x: List[torch.Tensor] = []
    kept_loss: List[torch.Tensor] = []
    kept_vf: List[torch.Tensor] = []
    kept_pred_stress_n: List[torch.Tensor] = []
    kept_pred_nu_n: List[torch.Tensor] = []

    total_draws = 0
    while sum(t.shape[0] for t in kept_x) < pool_target:
        xm11 = sample_guided_batch(
            ddpm,
            cnn=cnn,
            target_stress_n=tgt_stress_n,
            target_nu_n=tgt_nu_n,
            batch_size=args.num_batch,
            device=device,
            guidance_scale=args.guidance_scale,
            guidance_start_frac=args.guidance_start_frac,
            guidance_power=args.guidance_power,
            guidance_every=args.guidance_every,
            w_stress=w_stress_eff,
            w_nu=w_nu_eff,
            volfrac_min=args.volfrac_min,
            volfrac_lambda=args.volfrac_lambda,
            grad_clip=args.grad_clip,
            binarize_threshold=args.binarize_threshold,
        )
        total_draws += int(xm11.shape[0])

        with torch.no_grad():
            x01 = to_zero_one(xm11).clamp(0.0, 1.0)
            stress_pred_n, nu_pred_n = cnn(x01)

            loss = torch.zeros((xm11.shape[0],), device=stress_pred_n.device, dtype=torch.float32)
            if tgt_stress_n is not None and w_stress_eff > 0.0:
                loss = loss + float(w_stress_eff) * (stress_pred_n - tgt_stress_n.view(1, -1)).pow(2).mean(dim=1)
            if tgt_nu_n is not None and w_nu_eff > 0.0:
                loss = loss + float(w_nu_eff) * (nu_pred_n - tgt_nu_n.view(1, -1)).pow(2).mean(dim=1)

            vf = volfrac_hard_from_xm11(xm11, threshold=args.binarize_threshold)
            keep = vf >= float(args.volfrac_min)

        if keep.any():
            kept_x.append(xm11[keep].cpu())
            kept_loss.append(loss[keep].cpu())
            kept_vf.append(vf[keep].cpu())
            kept_pred_stress_n.append(stress_pred_n[keep].cpu())
            kept_pred_nu_n.append(nu_pred_n[keep].cpu())

        n_kept = sum(t.shape[0] for t in kept_x)
        if (n_kept % int(args.save_every)) < int(args.num_batch):
            if n_kept > 0:
                xm = torch.cat(kept_x, dim=0)
                ls = torch.cat(kept_loss, dim=0)
                idx = torch.argsort(ls)[: min(64, xm.shape[0])]
                preview = xm[idx]
                save_grid(preview, os.path.join(args.out_dir, "preview_best.png"),
                          nrow=args.save_nrow, binarize=True, threshold=args.binarize_threshold, upscale=6)
            print(f"[Progress] kept={n_kept}/{pool_target} draws={total_draws}")

    xm_all = torch.cat(kept_x, dim=0)
    loss_all = torch.cat(kept_loss, dim=0)
    vf_all = torch.cat(kept_vf, dim=0)
    ps_all = torch.cat(kept_pred_stress_n, dim=0)
    pn_all = torch.cat(kept_pred_nu_n, dim=0)

    order = torch.argsort(loss_all)[:want]
    xm_best = xm_all[order]
    loss_best = loss_all[order]
    vf_best = vf_all[order]
    ps_best_n = ps_all[order]
    pn_best_n = pn_all[order]

    with torch.no_grad():
        ps_best_phys, pn_best_phys = ns.denormalize(ps_best_n.to(device), pn_best_n.to(device))
        ps_best_phys = ps_best_phys.cpu()
        pn_best_phys = pn_best_phys.cpu()

    save_grid(xm_best[: min(64, want)], os.path.join(args.out_dir, "grid_best.png"),
              nrow=args.save_nrow, binarize=True, threshold=args.binarize_threshold, upscale=6)
    save_grid(xm_best[: min(64, want)], os.path.join(args.out_dir, "grid_best_gray.png"),
              nrow=args.save_nrow, binarize=False, threshold=args.binarize_threshold, upscale=6)

    x01_bin = (xm_best > float(args.binarize_threshold)).float()
    out_pt = os.path.join(args.out_dir, f"guided_samples_{want}.pt")

    payload = {
        "xm11": xm_best,
        "x01_bin": x01_bin,
        "volfrac": vf_best,
        "loss_norm": loss_best,
        "pred_stress_norm": ps_best_n,
        "pred_nu_norm": pn_best_n,
        "pred_stress_phys": ps_best_phys,
        "pred_nu_phys": pn_best_phys,
        "target_mode": args.target_mode,
        "target_raw": tgt_raw,
        "target_used": {"stress": bool(use_stress), "nu": bool(use_nu)},
        "ddpm_ckpt": args.ddpm_ckpt,
        "cnn_ckpt": args.cnn_ckpt,
        "cnn_cfg": args.cnn_cfg,
        "sampling_args": vars(args),
    }
    if tgt_stress_n is not None:
        payload["target_stress_norm"] = tgt_stress_n.detach().cpu()
    if tgt_nu_n is not None:
        payload["target_nu_norm"] = tgt_nu_n.detach().cpu()

    torch.save(payload, out_pt)

    meta = {
        "ddpm_ckpt": args.ddpm_ckpt,
        "cnn_ckpt": args.cnn_ckpt,
        "cnn_cfg": args.cnn_cfg,
        "target_json": args.target_json,
        "target_used": {"stress": bool(use_stress), "nu": bool(use_nu)},
        "num": want,
        "pool_mult": int(args.pool_mult),
        "pool_size": int(pool_target),
        "timesteps": int(timesteps),
        "schedule": str(schedule),
        "volfrac_min": float(args.volfrac_min),
        "guidance_scale": float(args.guidance_scale),
        "guidance_start_frac": float(args.guidance_start_frac),
        "guidance_power": float(args.guidance_power),
        "guidance_every": int(args.guidance_every),
        "w_stress": float(w_stress_eff),
        "w_nu": float(w_nu_eff),
        "volfrac_lambda": float(args.volfrac_lambda),
        "total_draws": int(total_draws),
        "kept_before_topk": int(xm_all.shape[0]),
        "loss_norm_mean": float(loss_best.mean().item()),
        "loss_norm_min": float(loss_best.min().item()),
        "loss_norm_max": float(loss_best.max().item()),
        "vf_mean": float(vf_best.mean().item()),
        "vf_min": float(vf_best.min().item()),
        "vf_max": float(vf_best.max().item()),
        "artifacts": {
            "grid_best": "grid_best.png",
            "grid_best_gray": "grid_best_gray.png",
            "preview_best": "preview_best.png",
            "pt": os.path.basename(out_pt),
        },
    }
    with open(os.path.join(args.out_dir, "guided_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[Done] wrote {out_pt} and grids in {args.out_dir}")


if __name__ == "__main__":
    main()
