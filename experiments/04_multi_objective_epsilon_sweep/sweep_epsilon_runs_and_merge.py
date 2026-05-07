#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sweep_epsilon_runs_and_merge.py

Outer sweep driver for guided_sample_diffusion_epsilon_constraint.py.

What it does
------------
1. Runs the inner epsilon-constraint sampler over a list of epsilon values.
2. Repeats each epsilon for multiple seeds.
3. Collects all returned .pt files.
4. Merges returned candidates into one combined pool.
5. Computes global Pareto sets in (stress_mse_norm, nu_mse_norm) space.
6. Writes CSV/PT summaries for downstream selection and plotting.

By default, the Pareto comparison assumes both objectives are minimized:
    - stress_mse_norm
    - nu_mse_norm

This is independent of which objective was primary in any specific run.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch


def parse_float_list(s: str) -> List[float]:
    vals = []
    for x in s.split(","):
        x = x.strip()
        if x:
            vals.append(float(x))
    if not vals:
        raise ValueError("No epsilon values parsed.")
    return vals


def epsilon_tag(eps: float) -> str:
    s = f"{eps:.8g}"
    return s.replace("-", "m").replace("+", "p").replace(".", "p")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    # Required paths
    ap.add_argument("--sampler_py", type=str, default="guided_sample_diffusion_epsilon_constraint.py",
                    help="Path to the inner epsilon-constraint sampler script.")
    ap.add_argument("--ddpm_ckpt", type=str, required=True)
    ap.add_argument("--cnn_ckpt", type=str, required=True)
    ap.add_argument("--cnn_cfg", type=str, required=True)
    ap.add_argument("--target_json", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True,
                    help="Base output directory for the whole sweep.")

    # Sweep settings
    ap.add_argument(
        "--epsilons",
        type=str,
        default="0.03,0.05,0.0835,0.1392,0.2323,0.3875,0.6463,1.0781,1.7985,3.0",
        help="Comma-separated epsilon schedule.",
    )
    ap.add_argument("--runs_per_epsilon", type=int, default=10,
                    help="How many seeds to run for each epsilon.")
    ap.add_argument("--base_seed", type=int, default=0,
                    help="Seed offset for sweep. Actual seed = base_seed + eps_idx*runs_per_epsilon + run_idx.")
    ap.add_argument("--dry_run", action="store_true",
                    help="Print commands without running them.")
    ap.add_argument("--stop_on_error", action="store_true",
                    help="Stop entire sweep if one run fails.")
    ap.add_argument("--skip_existing", action="store_true",
                    help="Skip a run if its expected PT file already exists.")

    # Pass-through inner sampler args
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--num", type=int, default=32,
                    help="Returned designs per inner run.")
    ap.add_argument("--num_batch", type=int, default=32)
    ap.add_argument("--pool_mult", type=int, default=3)
    ap.add_argument("--max_draws", type=int, default=50000)
    ap.add_argument("--volfrac_min", type=float, default=0.30)
    ap.add_argument("--binarize_threshold", type=float, default=0.0)
    ap.add_argument("--timesteps", type=int, default=-1)
    ap.add_argument("--schedule", type=str, default="")
    ap.add_argument("--target_mode", type=str, default="physical", choices=["physical", "normalized"])
    ap.add_argument("--guidance_scale", type=float, default=2.0)
    ap.add_argument("--guidance_start_frac", type=float, default=0.30)
    ap.add_argument("--guidance_power", type=float, default=2.0)
    ap.add_argument("--guidance_every", type=int, default=1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--primary_objective", type=str, default="stress", choices=["stress", "nu"])
    ap.add_argument("--constraint_penalty", type=float, default=10.0)
    ap.add_argument("--volfrac_lambda", type=float, default=0.0)
    ap.add_argument("--save_every", type=int, default=16)
    ap.add_argument("--save_nrow", type=int, default=4)

    # Merge behavior
    ap.add_argument("--include_fallback_in_pareto", action="store_true",
                    help="Also include fallback_infeasible returned samples in the global Pareto merge.")

    return ap.parse_args()


def run_one(cmd: Sequence[str], dry_run: bool = False) -> int:
    print("[Run]", " ".join(cmd))
    if dry_run:
        return 0
    cp = subprocess.run(cmd)
    return int(cp.returncode)


def find_generated_pt(run_dir: Path) -> Path | None:
    pts = sorted(run_dir.glob("guided_eps_*.pt"))
    if not pts:
        return None
    # There should only be one payload file per run dir.
    return pts[-1]


def is_nondominated_min_2d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return boolean mask of nondominated points for minimizing x and y.

    O(N log N) after sorting by x, then sweeping smallest y seen so far.
    Points with equal x and y are all kept.
    """
    assert x.ndim == 1 and y.ndim == 1 and x.shape[0] == y.shape[0]
    n = x.shape[0]
    if n == 0:
        return torch.zeros((0,), dtype=torch.bool)

    order = torch.argsort(x, stable=True)
    xs = x[order]
    ys = y[order]

    keep_sorted = torch.zeros(n, dtype=torch.bool)
    best_y = math.inf
    i = 0
    while i < n:
        j = i + 1
        while j < n and float(xs[j].item()) == float(xs[i].item()):
            j += 1
        group_y = ys[i:j]
        group_min_y = float(group_y.min().item())
        keep_group = group_y <= best_y
        keep_sorted[i:j] = keep_group
        best_y = min(best_y, group_min_y)
        i = j

    keep = torch.zeros(n, dtype=torch.bool)
    keep[order] = keep_sorted
    return keep


def merge_payloads(pt_paths: Sequence[Path], include_fallback_in_pareto: bool) -> Dict[str, object]:
    rec_xm11: List[torch.Tensor] = []
    rec_x01_bin: List[torch.Tensor] = []
    rec_volfrac: List[torch.Tensor] = []
    rec_stress: List[torch.Tensor] = []
    rec_nu: List[torch.Tensor] = []
    rec_primary: List[torch.Tensor] = []
    rec_constraint: List[torch.Tensor] = []
    rec_violation: List[torch.Tensor] = []
    rec_pred_stress_n: List[torch.Tensor] = []
    rec_pred_nu_n: List[torch.Tensor] = []
    rec_pred_stress_phys: List[torch.Tensor] = []
    rec_pred_nu_phys: List[torch.Tensor] = []

    eps_list: List[torch.Tensor] = []
    seed_list: List[torch.Tensor] = []
    run_id_list: List[torch.Tensor] = []
    source_feasible_list: List[torch.Tensor] = []
    selection_mode_list: List[str] = []
    source_path_list: List[str] = []

    run_records: List[Dict[str, object]] = []

    for run_id, pt_path in enumerate(pt_paths):
        payload = torch.load(pt_path, map_location="cpu")

        xm11 = payload["xm11"]
        n = int(xm11.shape[0])
        if n == 0:
            continue

        x01_bin = payload.get("x01_bin", (xm11 > 0).float())
        volfrac = payload["volfrac"]
        stress = payload["stress_mse_norm"]
        nu = payload["nu_mse_norm"]
        primary = payload.get("primary_loss_norm", stress.clone())
        constraint = payload.get("constraint_loss_norm", nu.clone())
        violation = payload.get("constraint_violation_norm", torch.zeros_like(primary))
        psn = payload.get("pred_stress_norm")
        pnn = payload.get("pred_nu_norm")
        psp = payload.get("pred_stress_phys")
        pnp = payload.get("pred_nu_phys")

        eps = float(payload.get("epsilon", float("nan")))
        sampling_args = payload.get("sampling_args", {})
        seed = int(sampling_args.get("seed", -1))
        selection_mode = str(payload.get("selection_mode", "unknown"))
        selection_is_feasible = bool(payload.get("selection_is_feasible", False))

        rec_xm11.append(xm11)
        rec_x01_bin.append(x01_bin)
        rec_volfrac.append(volfrac)
        rec_stress.append(stress)
        rec_nu.append(nu)
        rec_primary.append(primary)
        rec_constraint.append(constraint)
        rec_violation.append(violation)
        if psn is not None:
            rec_pred_stress_n.append(psn)
        if pnn is not None:
            rec_pred_nu_n.append(pnn)
        if psp is not None:
            rec_pred_stress_phys.append(psp)
        if pnp is not None:
            rec_pred_nu_phys.append(pnp)

        eps_list.append(torch.full((n,), eps, dtype=torch.float32))
        seed_list.append(torch.full((n,), seed, dtype=torch.int64))
        run_id_list.append(torch.full((n,), run_id, dtype=torch.int64))
        source_feasible_list.append(torch.full((n,), selection_is_feasible, dtype=torch.bool))
        selection_mode_list.extend([selection_mode] * n)
        source_path_list.extend([str(pt_path)] * n)

        run_records.append(
            {
                "run_id": run_id,
                "pt_path": str(pt_path),
                "epsilon": eps,
                "seed": seed,
                "selection_mode": selection_mode,
                "selection_is_feasible": selection_is_feasible,
                "num_returned": n,
                "stress_mse_min": float(stress.min().item()),
                "stress_mse_mean": float(stress.mean().item()),
                "nu_mse_min": float(nu.min().item()),
                "nu_mse_mean": float(nu.mean().item()),
            }
        )

    if not rec_xm11:
        raise RuntimeError("No run payloads were available to merge.")

    xm11_all = torch.cat(rec_xm11, dim=0)
    x01_bin_all = torch.cat(rec_x01_bin, dim=0)
    volfrac_all = torch.cat(rec_volfrac, dim=0)
    stress_all = torch.cat(rec_stress, dim=0)
    nu_all = torch.cat(rec_nu, dim=0)
    primary_all = torch.cat(rec_primary, dim=0)
    constraint_all = torch.cat(rec_constraint, dim=0)
    violation_all = torch.cat(rec_violation, dim=0)
    eps_all = torch.cat(eps_list, dim=0)
    seed_all = torch.cat(seed_list, dim=0)
    run_id_all = torch.cat(run_id_list, dim=0)
    source_feasible_all = torch.cat(source_feasible_list, dim=0)

    pred_stress_n_all = torch.cat(rec_pred_stress_n, dim=0) if rec_pred_stress_n else None
    pred_nu_n_all = torch.cat(rec_pred_nu_n, dim=0) if rec_pred_nu_n else None
    pred_stress_phys_all = torch.cat(rec_pred_stress_phys, dim=0) if rec_pred_stress_phys else None
    pred_nu_phys_all = torch.cat(rec_pred_nu_phys, dim=0) if rec_pred_nu_phys else None

    pareto_all_mask = is_nondominated_min_2d(stress_all, nu_all)

    if include_fallback_in_pareto:
        pareto_filter_mask = torch.ones_like(source_feasible_all, dtype=torch.bool)
    else:
        pareto_filter_mask = source_feasible_all.clone()

    pareto_input_mask = pareto_filter_mask
    pareto_on_filtered = is_nondominated_min_2d(stress_all[pareto_input_mask], nu_all[pareto_input_mask]) if pareto_input_mask.any() else torch.zeros((0,), dtype=torch.bool)
    pareto_feasible_mask = torch.zeros_like(source_feasible_all, dtype=torch.bool)
    if pareto_input_mask.any():
        pareto_feasible_mask[pareto_input_mask] = pareto_on_filtered

    return {
        "xm11": xm11_all,
        "x01_bin": x01_bin_all,
        "volfrac": volfrac_all,
        "stress_mse_norm": stress_all,
        "nu_mse_norm": nu_all,
        "primary_loss_norm": primary_all,
        "constraint_loss_norm": constraint_all,
        "constraint_violation_norm": violation_all,
        "pred_stress_norm": pred_stress_n_all,
        "pred_nu_norm": pred_nu_n_all,
        "pred_stress_phys": pred_stress_phys_all,
        "pred_nu_phys": pred_nu_phys_all,
        "source_epsilon": eps_all,
        "source_seed": seed_all,
        "source_run_id": run_id_all,
        "source_selection_is_feasible": source_feasible_all,
        "source_selection_mode": selection_mode_list,
        "source_pt_path": source_path_list,
        "pareto_all_mask": pareto_all_mask,
        "pareto_filtered_mask": pareto_feasible_mask,
        "run_records": run_records,
    }


def subset_payload(payload: Dict[str, object], mask: torch.Tensor) -> Dict[str, object]:
    out: Dict[str, object] = {}
    n = int(mask.numel())
    for k, v in payload.items():
        if k == "run_records":
            out[k] = v
        elif isinstance(v, torch.Tensor) and v.shape[:1] == (n,):
            out[k] = v[mask]
        elif isinstance(v, list) and len(v) == n:
            out[k] = [vv for vv, m in zip(v, mask.tolist()) if m]
        else:
            out[k] = v
    return out


def write_csv(payload: Dict[str, object], csv_path: Path) -> None:
    n = int(payload["stress_mse_norm"].shape[0])
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "index",
            "source_run_id",
            "source_seed",
            "source_epsilon",
            "stress_mse_norm",
            "nu_mse_norm",
            "primary_loss_norm",
            "constraint_loss_norm",
            "constraint_violation_norm",
            "volfrac",
            "source_selection_is_feasible",
            "source_selection_mode",
            "pareto_all",
            "pareto_filtered",
            "source_pt_path",
        ])
        for i in range(n):
            writer.writerow([
                i,
                int(payload["source_run_id"][i].item()),
                int(payload["source_seed"][i].item()),
                float(payload["source_epsilon"][i].item()),
                float(payload["stress_mse_norm"][i].item()),
                float(payload["nu_mse_norm"][i].item()),
                float(payload["primary_loss_norm"][i].item()),
                float(payload["constraint_loss_norm"][i].item()),
                float(payload["constraint_violation_norm"][i].item()),
                float(payload["volfrac"][i].item()),
                bool(payload["source_selection_is_feasible"][i].item()),
                payload["source_selection_mode"][i],
                bool(payload["pareto_all_mask"][i].item()),
                bool(payload["pareto_filtered_mask"][i].item()),
                payload["source_pt_path"][i],
            ])


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    runs_dir = out_dir / "runs"
    merge_dir = out_dir / "merged"
    runs_dir.mkdir(parents=True, exist_ok=True)
    merge_dir.mkdir(parents=True, exist_ok=True)

    epsilons = parse_float_list(args.epsilons)

    launched: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []

    sampler_py = Path(args.sampler_py)
    if not sampler_py.is_absolute():
        sampler_py = Path.cwd() / sampler_py

    run_counter = 0
    for eps_idx, eps in enumerate(epsilons):
        eps_dir = runs_dir / f"eps_{eps_idx+1:02d}_{epsilon_tag(eps)}"
        eps_dir.mkdir(parents=True, exist_ok=True)

        for rep in range(args.runs_per_epsilon):
            seed = int(args.base_seed + eps_idx * args.runs_per_epsilon + rep)
            run_name = f"seed_{seed:04d}"
            run_dir = eps_dir / run_name
            run_dir.mkdir(parents=True, exist_ok=True)

            expected_pt = find_generated_pt(run_dir)
            if args.skip_existing and expected_pt is not None:
                print(f"[Skip] existing payload found: {expected_pt}")
                launched.append({
                    "epsilon": eps,
                    "seed": seed,
                    "run_dir": str(run_dir),
                    "returncode": 0,
                    "skipped": True,
                })
                run_counter += 1
                continue

            cmd = [
                sys.executable,
                str(sampler_py),
                "--ddpm_ckpt", args.ddpm_ckpt,
                "--cnn_ckpt", args.cnn_ckpt,
                "--cnn_cfg", args.cnn_cfg,
                "--target_json", args.target_json,
                "--out_dir", str(run_dir),
                "--device", args.device,
                "--seed", str(seed),
                "--num", str(args.num),
                "--num_batch", str(args.num_batch),
                "--pool_mult", str(args.pool_mult),
                "--max_draws", str(args.max_draws),
                "--volfrac_min", str(args.volfrac_min),
                "--binarize_threshold", str(args.binarize_threshold),
                "--timesteps", str(args.timesteps),
                "--target_mode", str(args.target_mode),
                "--guidance_scale", str(args.guidance_scale),
                "--guidance_start_frac", str(args.guidance_start_frac),
                "--guidance_power", str(args.guidance_power),
                "--guidance_every", str(args.guidance_every),
                "--grad_clip", str(args.grad_clip),
                "--primary_objective", str(args.primary_objective),
                "--epsilon", str(eps),
                "--constraint_penalty", str(args.constraint_penalty),
                "--volfrac_lambda", str(args.volfrac_lambda),
                "--save_every", str(args.save_every),
                "--save_nrow", str(args.save_nrow),
            ]
            if args.schedule:
                cmd.extend(["--schedule", str(args.schedule)])

            rc = run_one(cmd, dry_run=args.dry_run)
            record = {
                "epsilon": eps,
                "seed": seed,
                "run_dir": str(run_dir),
                "returncode": rc,
                "skipped": False,
            }
            launched.append(record)
            run_counter += 1

            if rc != 0:
                failures.append(record)
                if args.stop_on_error:
                    raise RuntimeError(f"Run failed: epsilon={eps} seed={seed} rc={rc}")

    with (merge_dir / "launch_records.json").open("w", encoding="utf-8") as f:
        json.dump(launched, f, indent=2)

    if args.dry_run:
        print("[Dry-run] commands printed only. Merge stage skipped.")
        return

    # Collect PT payloads from successful/available runs.
    pt_paths: List[Path] = []
    missing_runs: List[Dict[str, object]] = []
    for rec in launched:
        run_dir = Path(rec["run_dir"])
        pt_path = find_generated_pt(run_dir)
        if pt_path is None:
            missing_runs.append(rec)
            continue
        pt_paths.append(pt_path)

    with (merge_dir / "missing_runs.json").open("w", encoding="utf-8") as f:
        json.dump(missing_runs, f, indent=2)
    with (merge_dir / "failures.json").open("w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2)

    if not pt_paths:
        raise RuntimeError("No output PT files were found. Nothing to merge.")

    payload = merge_payloads(pt_paths, include_fallback_in_pareto=args.include_fallback_in_pareto)

    all_pt = merge_dir / "merged_all_candidates.pt"
    torch.save(payload, all_pt)
    write_csv(payload, merge_dir / "merged_all_candidates.csv")

    pareto_all_payload = subset_payload(payload, payload["pareto_all_mask"])
    torch.save(pareto_all_payload, merge_dir / "pareto_all_candidates.pt")
    write_csv(pareto_all_payload, merge_dir / "pareto_all_candidates.csv")

    pareto_filtered_payload = subset_payload(payload, payload["pareto_filtered_mask"])
    torch.save(pareto_filtered_payload, merge_dir / "pareto_filtered_candidates.pt")
    write_csv(pareto_filtered_payload, merge_dir / "pareto_filtered_candidates.csv")

    summary = {
        "num_epsilons": len(epsilons),
        "runs_per_epsilon": int(args.runs_per_epsilon),
        "num_launched_records": len(launched),
        "num_failures": len(failures),
        "num_missing_payloads": len(missing_runs),
        "num_payloads_merged": len(pt_paths),
        "num_candidates_all": int(payload["stress_mse_norm"].shape[0]),
        "num_pareto_all": int(payload["pareto_all_mask"].sum().item()),
        "num_pareto_filtered": int(payload["pareto_filtered_mask"].sum().item()),
        "include_fallback_in_pareto": bool(args.include_fallback_in_pareto),
        "epsilons": epsilons,
        "artifacts": {
            "launch_records": "launch_records.json",
            "missing_runs": "missing_runs.json",
            "failures": "failures.json",
            "merged_all_pt": all_pt.name,
            "merged_all_csv": "merged_all_candidates.csv",
            "pareto_all_pt": "pareto_all_candidates.pt",
            "pareto_all_csv": "pareto_all_candidates.csv",
            "pareto_filtered_pt": "pareto_filtered_candidates.pt",
            "pareto_filtered_csv": "pareto_filtered_candidates.csv",
        },
    }
    with (merge_dir / "sweep_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[Done] merged {len(pt_paths)} payloads")
    print(f"[Done] wrote: {all_pt}")
    print(f"[Done] Pareto(all) count = {summary['num_pareto_all']}")
    print(f"[Done] Pareto(filtered) count = {summary['num_pareto_filtered']}")


if __name__ == "__main__":
    main()
