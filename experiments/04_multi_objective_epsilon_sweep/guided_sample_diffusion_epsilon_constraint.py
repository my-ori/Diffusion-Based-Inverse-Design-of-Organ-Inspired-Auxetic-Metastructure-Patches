# -*- coding: utf-8 -*-
"""guided_sample_diffusion_epsilon_constraint.py

Surrogate-guided sampling for an unconditional DDPM using an ε-constraint
formulation.

Problem solved for one run:
    minimize   f_primary(x)
    subject to f_constraint(x) <= epsilon

where x is the final returned design, and during diffusion guidance we use the
current x0 estimate as a differentiable proxy.

Current supported objectives:
  - stress MSE
  - Poisson's-ratio (nu) MSE

Choose one as the primary objective via --primary_objective. The other is
used as the ε-constrained objective.

Example (stress primary, nu constrained):
    python guided_sample_diffusion_epsilon_constraint.py \
      --ddpm_ckpt ddpm_epoch_2950.pt \
      --cnn_ckpt best.pt \
      --cnn_cfg config.json \
      --target_json y_target.json \
      --primary_objective stress \
      --epsilon 0.02 \
      --out_dir runs/ddpm/guided_eps
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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
        n = int(stress.numel())
        return (stress - self.stress_mean[:n]) / self.stress_std[:n]

    def normalize_nu(self, nu: torch.Tensor) -> torch.Tensor:
        n = int(nu.numel())
        return (nu - self.nu_mean[:n]) / self.nu_std[:n]

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
    ap.add_argument(
        "--target_json",
        type=str,
        required=True,
        help="Target property JSON with stress/nu arrays. Each array can have any length 1..30; length n means the first n default strain points.",
    )

    # Output
    ap.add_argument("--out_dir", type=str, default="runs/ddpm/guided_eps")

    # Sampling
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num", type=int, default=16, help="How many final feasible designs to save.")
    ap.add_argument("--num_batch", type=int, default=32, help="Batch size for diffusion sampling.")
    ap.add_argument("--pool_mult", type=int, default=3, help="Need pool_mult * num feasible designs before final top-k selection.")
    ap.add_argument("--max_draws", type=int, default=50000, help="Safety cap on total generated designs.")
    ap.add_argument("--volfrac_min", type=float, default=0.30, help="Hard filter after binarization.")
    ap.add_argument("--binarize_threshold", type=float, default=0.0, help="Threshold in [-1,1] space (0.0 == 0.5 in [0,1]).")

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

    # ε-constraint formulation
    ap.add_argument("--primary_objective", type=str, default="stress", choices=["stress", "nu"],
                    help="Primary objective to minimize. The other objective becomes the ε-constrained one.")
    ap.add_argument("--epsilon", type=float, required=True,
                    help="Constraint threshold on the non-primary objective, evaluated in normalized MSE space.")
    ap.add_argument("--constraint_penalty", type=float, default=10.0,
                    help="Penalty weight for max(0, f_constraint - epsilon)^2 during guidance.")
    ap.add_argument("--volfrac_lambda", type=float, default=0.0,
                    help="Optional soft penalty weight to encourage volfrac >= volfrac_min during guidance.")

    # Saving
    ap.add_argument("--save_every", type=int, default=16, help="Write a grid PNG every N feasible kept samples.")
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
        if stress_t.ndim != 1 or not (1 <= stress_t.numel() <= 30):
            raise ValueError(f"stress target must have length in [1, 30]. Got shape={tuple(stress_t.shape)}")

    if nu is not None:
        nu_t = torch.tensor(nu, dtype=torch.float32, device=device)
        if nu_t.ndim != 1 or not (1 <= nu_t.numel() <= 30):
            raise ValueError(f"nu target must have length in [1, 30]. Got shape={tuple(nu_t.shape)}")

    if stress_t is None and nu_t is None:
        raise ValueError("target_json must contain at least one of: 'stress'/'y_stress' or 'nu'/'y_nu'.")

    return stress_t, nu_t, d


# -------------------------
# Objective helpers
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


def mse_scalar(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    n = int(target.numel())
    return F.mse_loss(pred[:, :n], target.view(1, -1).expand(pred.shape[0], n), reduction="mean")


def mse_vector(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    n = int(target.numel())
    return (pred[:, :n] - target.view(1, -1)).pow(2).mean(dim=1)


def get_primary_and_constraint(primary_objective: str) -> Tuple[str, str]:
    if primary_objective == "stress":
        return "stress", "nu"
    if primary_objective == "nu":
        return "nu", "stress"
    raise ValueError(f"Unsupported primary_objective={primary_objective}")


def pick_metric(primary_objective: str, stress_metric: torch.Tensor, nu_metric: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    primary_name, constraint_name = get_primary_and_constraint(primary_objective)
    if primary_name == "stress":
        return stress_metric, nu_metric
    if primary_name == "nu":
        return nu_metric, stress_metric
    raise ValueError(primary_name)


# -------------------------
# Guidance / sampling
# -------------------------
def guided_p_sample(
    ddpm: DDPM,
    x: torch.Tensor,
    t: torch.Tensor,
    *,
    cnn: AuxeticCNNSurrogate,
    target_stress_n: torch.Tensor,
    target_nu_n: torch.Tensor,
    scale: float,
    primary_objective: str,
    epsilon: float,
    constraint_penalty: float,
    volfrac_min: float,
    volfrac_lambda: float,
    grad_clip: float,
) -> torch.Tensor:
    # ----- Standard DDPM mean (no grad through UNet) -----
    with torch.no_grad():
        eps = ddpm.model(x, t)
        betas_t = extract(ddpm.betas, t, x.shape)
        sqrt_one_minus_acp_t = extract(ddpm.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = extract(ddpm.sqrt_recip_alphas, t, x.shape)
        model_mean = sqrt_recip_alphas_t * (x - betas_t * eps / sqrt_one_minus_acp_t)

    # ----- Guidance term (grad wrt x only) -----
    if scale > 0.0:
        x_in = x.detach().requires_grad_(True)

        sqrt_acp_t = extract(ddpm.sqrt_alphas_cumprod, t, x.shape)
        sqrt_om_t = extract(ddpm.sqrt_one_minus_alphas_cumprod, t, x.shape)
        x0_hat = (x_in - sqrt_om_t * eps.detach()) / sqrt_acp_t

        x01_hat = to_zero_one(x0_hat).clamp(0.0, 1.0)
        stress_pred_n, nu_pred_n = cnn(x01_hat)

        loss_stress = mse_scalar(stress_pred_n, target_stress_n)
        loss_nu = mse_scalar(nu_pred_n, target_nu_n)
        primary_loss, constraint_loss = pick_metric(primary_objective, loss_stress, loss_nu)

        constraint_violation = F.relu(constraint_loss - float(epsilon))
        loss = primary_loss + float(constraint_penalty) * constraint_violation.pow(2)

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
    target_stress_n: torch.Tensor,
    target_nu_n: torch.Tensor,
    batch_size: int,
    device: torch.device,
    guidance_scale: float,
    guidance_start_frac: float,
    guidance_power: float,
    guidance_every: int,
    primary_objective: str,
    epsilon: float,
    constraint_penalty: float,
    volfrac_min: float,
    volfrac_lambda: float,
    grad_clip: float,
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
            primary_objective=primary_objective,
            epsilon=epsilon,
            constraint_penalty=constraint_penalty,
            volfrac_min=volfrac_min,
            volfrac_lambda=volfrac_lambda,
            grad_clip=grad_clip,
        )
    return x


def _preview_order(primary_loss: torch.Tensor, violation: torch.Tensor) -> torch.Tensor:
    # Feasible samples have zero violation and sort ahead of infeasible ones.
    score = violation * 1.0e6 + primary_loss
    return torch.argsort(score)


def _epsilon_tag(epsilon: float) -> str:
    s = f"{epsilon:.8g}"
    return s.replace("-", "m").replace("+", "p").replace(".", "p")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    cnn, ns, _cnn_cfgj = load_cnn(args.cnn_ckpt, args.cnn_cfg, device=device)

    tgt_stress, tgt_nu, tgt_raw = load_target(args.target_json, device=device)
    if tgt_stress is None or tgt_nu is None:
        raise ValueError("ε-constraint mode requires both stress and nu targets to be present in target_json.")

    if args.target_mode == "physical":
        tgt_stress_n = ns.normalize_stress(tgt_stress)
        tgt_nu_n = ns.normalize_nu(tgt_nu)
    else:
        tgt_stress_n = tgt_stress
        tgt_nu_n = tgt_nu

    ckpt_tmp = torch.load(args.ddpm_ckpt, map_location="cpu")
    ckpt_args = ckpt_tmp.get("args", {})
    timesteps = int(ckpt_args.get("timesteps", 1000)) if args.timesteps < 0 else int(args.timesteps)
    schedule = str(ckpt_args.get("schedule", "cosine")) if args.schedule == "" else str(args.schedule)
    del ckpt_tmp

    ddpm, _ = load_ddpm(args.ddpm_ckpt, device=device, timesteps=timesteps, schedule=schedule, use_ema=True)
    primary_name, constraint_name = get_primary_and_constraint(args.primary_objective)
    print(f"[DDPM] timesteps={timesteps} schedule={schedule}")
    print(
        f"[Target] len_stress={int(tgt_stress.numel())} len_nu={int(tgt_nu.numel())} "
        f"primary={primary_name} constraint={constraint_name} epsilon={float(args.epsilon):.6g}"
    )

    want = int(args.num)
    pool_target = want * max(1, int(args.pool_mult))

    feasible_x: List[torch.Tensor] = []
    feasible_vf: List[torch.Tensor] = []
    feasible_stress_mse: List[torch.Tensor] = []
    feasible_nu_mse: List[torch.Tensor] = []
    feasible_primary: List[torch.Tensor] = []
    feasible_constraint: List[torch.Tensor] = []
    feasible_violation: List[torch.Tensor] = []
    feasible_pred_stress_n: List[torch.Tensor] = []
    feasible_pred_nu_n: List[torch.Tensor] = []

    backup_x: List[torch.Tensor] = []
    backup_vf: List[torch.Tensor] = []
    backup_stress_mse: List[torch.Tensor] = []
    backup_nu_mse: List[torch.Tensor] = []
    backup_primary: List[torch.Tensor] = []
    backup_constraint: List[torch.Tensor] = []
    backup_violation: List[torch.Tensor] = []
    backup_pred_stress_n: List[torch.Tensor] = []
    backup_pred_nu_n: List[torch.Tensor] = []

    total_draws = 0
    while sum(t.shape[0] for t in feasible_x) < pool_target and total_draws < int(args.max_draws):
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
            primary_objective=args.primary_objective,
            epsilon=float(args.epsilon),
            constraint_penalty=float(args.constraint_penalty),
            volfrac_min=args.volfrac_min,
            volfrac_lambda=args.volfrac_lambda,
            grad_clip=args.grad_clip,
        )
        total_draws += int(xm11.shape[0])

        with torch.no_grad():
            x01 = to_zero_one(xm11).clamp(0.0, 1.0)
            stress_pred_n, nu_pred_n = cnn(x01)
            stress_mse = mse_vector(stress_pred_n, tgt_stress_n)
            nu_mse = mse_vector(nu_pred_n, tgt_nu_n)
            primary_loss, constraint_loss = pick_metric(args.primary_objective, stress_mse, nu_mse)
            violation = F.relu(constraint_loss - float(args.epsilon))
            vf = volfrac_hard_from_xm11(xm11, threshold=args.binarize_threshold)

            hard_valid = vf >= float(args.volfrac_min)
            feasible = hard_valid & (constraint_loss <= float(args.epsilon))

        if hard_valid.any():
            backup_x.append(xm11[hard_valid].cpu())
            backup_vf.append(vf[hard_valid].cpu())
            backup_stress_mse.append(stress_mse[hard_valid].cpu())
            backup_nu_mse.append(nu_mse[hard_valid].cpu())
            backup_primary.append(primary_loss[hard_valid].cpu())
            backup_constraint.append(constraint_loss[hard_valid].cpu())
            backup_violation.append(violation[hard_valid].cpu())
            backup_pred_stress_n.append(stress_pred_n[hard_valid].cpu())
            backup_pred_nu_n.append(nu_pred_n[hard_valid].cpu())

        if feasible.any():
            feasible_x.append(xm11[feasible].cpu())
            feasible_vf.append(vf[feasible].cpu())
            feasible_stress_mse.append(stress_mse[feasible].cpu())
            feasible_nu_mse.append(nu_mse[feasible].cpu())
            feasible_primary.append(primary_loss[feasible].cpu())
            feasible_constraint.append(constraint_loss[feasible].cpu())
            feasible_violation.append(violation[feasible].cpu())
            feasible_pred_stress_n.append(stress_pred_n[feasible].cpu())
            feasible_pred_nu_n.append(nu_pred_n[feasible].cpu())

        n_feasible = sum(t.shape[0] for t in feasible_x)
        if (n_feasible % int(args.save_every)) < int(args.num_batch):
            if n_feasible > 0:
                xm = torch.cat(feasible_x, dim=0)
                primary_all = torch.cat(feasible_primary, dim=0)
                violation_all = torch.cat(feasible_violation, dim=0)
            elif len(backup_x) > 0:
                xm = torch.cat(backup_x, dim=0)
                primary_all = torch.cat(backup_primary, dim=0)
                violation_all = torch.cat(backup_violation, dim=0)
            else:
                xm = None
                primary_all = None
                violation_all = None

            if xm is not None:
                idx = _preview_order(primary_all, violation_all)[: min(64, xm.shape[0])]
                preview = xm[idx]
                save_grid(
                    preview,
                    os.path.join(args.out_dir, "preview_best.png"),
                    nrow=args.save_nrow,
                    binarize=True,
                    threshold=args.binarize_threshold,
                    upscale=6,
                )
            print(f"[Progress] feasible_kept={n_feasible}/{pool_target} draws={total_draws}")

    if len(feasible_x) > 0:
        xm_all = torch.cat(feasible_x, dim=0)
        vf_all = torch.cat(feasible_vf, dim=0)
        stress_mse_all = torch.cat(feasible_stress_mse, dim=0)
        nu_mse_all = torch.cat(feasible_nu_mse, dim=0)
        primary_all = torch.cat(feasible_primary, dim=0)
        constraint_all = torch.cat(feasible_constraint, dim=0)
        violation_all = torch.cat(feasible_violation, dim=0)
        ps_all = torch.cat(feasible_pred_stress_n, dim=0)
        pn_all = torch.cat(feasible_pred_nu_n, dim=0)
        selection_mode = "feasible"
        order = torch.argsort(primary_all)[: min(want, primary_all.shape[0])]
        is_feasible_out = True
    elif len(backup_x) > 0:
        xm_all = torch.cat(backup_x, dim=0)
        vf_all = torch.cat(backup_vf, dim=0)
        stress_mse_all = torch.cat(backup_stress_mse, dim=0)
        nu_mse_all = torch.cat(backup_nu_mse, dim=0)
        primary_all = torch.cat(backup_primary, dim=0)
        constraint_all = torch.cat(backup_constraint, dim=0)
        violation_all = torch.cat(backup_violation, dim=0)
        ps_all = torch.cat(backup_pred_stress_n, dim=0)
        pn_all = torch.cat(backup_pred_nu_n, dim=0)
        selection_mode = "fallback_infeasible"
        order = _preview_order(primary_all, violation_all)[: min(want, primary_all.shape[0])]
        is_feasible_out = False
        print("[Warn] No final samples satisfied the epsilon constraint. Saving nearest candidates instead.")
    else:
        raise RuntimeError(
            "No samples passed the hard volume-fraction filter. Try relaxing volfrac_min, increasing max_draws, or adjusting guidance."
        )

    xm_best = xm_all[order]
    vf_best = vf_all[order]
    stress_mse_best = stress_mse_all[order]
    nu_mse_best = nu_mse_all[order]
    primary_best = primary_all[order]
    constraint_best = constraint_all[order]
    violation_best = violation_all[order]
    ps_best_n = ps_all[order]
    pn_best_n = pn_all[order]

    with torch.no_grad():
        ps_best_phys, pn_best_phys = ns.denormalize(ps_best_n.to(device), pn_best_n.to(device))
        ps_best_phys = ps_best_phys.cpu()
        pn_best_phys = pn_best_phys.cpu()

    save_grid(
        xm_best[: min(64, xm_best.shape[0])],
        os.path.join(args.out_dir, "grid_best.png"),
        nrow=args.save_nrow,
        binarize=True,
        threshold=args.binarize_threshold,
        upscale=6,
    )
    save_grid(
        xm_best[: min(64, xm_best.shape[0])],
        os.path.join(args.out_dir, "grid_best_gray.png"),
        nrow=args.save_nrow,
        binarize=False,
        threshold=args.binarize_threshold,
        upscale=6,
    )

    x01_bin = (xm_best > float(args.binarize_threshold)).float()
    eps_tag = _epsilon_tag(float(args.epsilon))
    out_pt = os.path.join(args.out_dir, f"guided_eps_{constraint_name}_{eps_tag}_{xm_best.shape[0]}.pt")

    payload = {
        "xm11": xm_best,
        "x01_bin": x01_bin,
        "volfrac": vf_best,
        "stress_mse_norm": stress_mse_best,
        "nu_mse_norm": nu_mse_best,
        "primary_loss_norm": primary_best,
        "constraint_loss_norm": constraint_best,
        "constraint_violation_norm": violation_best,
        "pred_stress_norm": ps_best_n,
        "pred_nu_norm": pn_best_n,
        "pred_stress_phys": ps_best_phys,
        "pred_nu_phys": pn_best_phys,
        "target_mode": args.target_mode,
        "target_raw": tgt_raw,
        "primary_objective": primary_name,
        "constraint_objective": constraint_name,
        "epsilon": float(args.epsilon),
        "constraint_penalty": float(args.constraint_penalty),
        "selection_mode": selection_mode,
        "selection_is_feasible": bool(is_feasible_out),
        "ddpm_ckpt": args.ddpm_ckpt,
        "cnn_ckpt": args.cnn_ckpt,
        "cnn_cfg": args.cnn_cfg,
        "sampling_args": vars(args),
    }
    payload["target_stress_norm"] = tgt_stress_n.detach().cpu()
    payload["target_nu_norm"] = tgt_nu_n.detach().cpu()

    torch.save(payload, out_pt)

    meta = {
        "ddpm_ckpt": args.ddpm_ckpt,
        "cnn_ckpt": args.cnn_ckpt,
        "cnn_cfg": args.cnn_cfg,
        "target_json": args.target_json,
        "num_requested": want,
        "num_returned": int(xm_best.shape[0]),
        "pool_mult": int(args.pool_mult),
        "pool_target": int(pool_target),
        "timesteps": int(timesteps),
        "schedule": str(schedule),
        "volfrac_min": float(args.volfrac_min),
        "guidance_scale": float(args.guidance_scale),
        "guidance_start_frac": float(args.guidance_start_frac),
        "guidance_power": float(args.guidance_power),
        "guidance_every": int(args.guidance_every),
        "grad_clip": float(args.grad_clip),
        "primary_objective": primary_name,
        "constraint_objective": constraint_name,
        "epsilon": float(args.epsilon),
        "constraint_penalty": float(args.constraint_penalty),
        "volfrac_lambda": float(args.volfrac_lambda),
        "total_draws": int(total_draws),
        "kept_feasible_before_topk": int(sum(t.shape[0] for t in feasible_x)),
        "kept_hard_valid_before_topk": int(sum(t.shape[0] for t in backup_x)),
        "selection_mode": selection_mode,
        "selection_is_feasible": bool(is_feasible_out),
        "primary_loss_mean": float(primary_best.mean().item()),
        "primary_loss_min": float(primary_best.min().item()),
        "primary_loss_max": float(primary_best.max().item()),
        "constraint_loss_mean": float(constraint_best.mean().item()),
        "constraint_loss_min": float(constraint_best.min().item()),
        "constraint_loss_max": float(constraint_best.max().item()),
        "constraint_violation_mean": float(violation_best.mean().item()),
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
