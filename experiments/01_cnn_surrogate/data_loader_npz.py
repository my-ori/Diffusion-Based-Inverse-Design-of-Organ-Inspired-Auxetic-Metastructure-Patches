# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset


@dataclass
class NPZNormStats:
    """
    Store normalization stats for labels.
    We normalize per-dimension (each strain point) by default.
    """
    stress_mean: torch.Tensor  # (30,)
    stress_std: torch.Tensor   # (30,)
    nu_mean: torch.Tensor      # (30,)
    nu_std: torch.Tensor       # (30,)

    def to(self, device: torch.device) -> "NPZNormStats":
        return NPZNormStats(
            stress_mean=self.stress_mean.to(device),
            stress_std=self.stress_std.to(device),
            nu_mean=self.nu_mean.to(device),
            nu_std=self.nu_std.to(device),
        )


def _safe_std(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.clamp(x, min=eps)


def compute_norm_stats(
    y_stress: torch.Tensor,  # (N,30)
    y_nu: torch.Tensor,      # (N,30)
) -> NPZNormStats:
    """
    Compute per-dimension mean/std over the training split only.
    """
    stress_mean = y_stress.mean(dim=0)
    stress_std = _safe_std(y_stress.std(dim=0, unbiased=False))
    nu_mean = y_nu.mean(dim=0)
    nu_std = _safe_std(y_nu.std(dim=0, unbiased=False))
    return NPZNormStats(stress_mean, stress_std, nu_mean, nu_std)


def _norm_cname(s: str) -> str:
    # normalize class name for robust matching
    return str(s).strip().lower().replace(" ", "_")


def resolve_class_ids_by_name(class_names: List[str], names: List[str]) -> List[int]:
    """
    Map a list of class name strings to their integer class_ids based on class_names.
    Matching is case-insensitive and treats spaces as underscores.
    Raises ValueError if any name is unknown.
    """
    if names is None:
        return []
    name_to_id = {_norm_cname(cn): i for i, cn in enumerate(class_names)}
    out: List[int] = []
    missing: List[str] = []
    for n in names:
        key = _norm_cname(n)
        if key not in name_to_id:
            missing.append(str(n))
        else:
            out.append(int(name_to_id[key]))
    if missing:
        raise ValueError(
            "Unknown class name(s): "
            + ", ".join(missing)
            + ". Available: "
            + ", ".join([str(c) for c in class_names])
        )
    return out


def compute_volfrac_from_X(
    X: np.ndarray,
    x_as_binary: bool = True,
    x_invert: bool = False,
) -> np.ndarray:
    """
    Compute volume fraction (solid pixel ratio) for each sample.

    X: (N,H,W) array. Typically uint8 0/1 where 1 = solid.
    Returns: (N,) float32 in [0,1]. If x_invert=True, vf := 1 - vf so it matches
    the returned x when inversion is enabled.
    """
    if X.ndim != 3:
        raise ValueError(f"Expected X with shape (N,H,W), got {X.shape}")
    solid = (X > 0).astype(np.float32) if x_as_binary else X.astype(np.float32)
    vf = solid.mean(axis=(1, 2))
    if x_invert:
        vf = 1.0 - vf
    return vf.astype(np.float32)



class AuxeticNPZDataset(Dataset):
    """
    Loads one compressed .npz file with fields:
      X: (N,50,50) uint8 (0/1)
      y_stress: (N,30) float32
      y_nu: (N,30) float32
      class_ids: (N,) int64
      class_names: (C,) object
      sample_keys: (N,) object
      target_strains: (30,) float32

    Returns a dict:
      {
        "x": FloatTensor (1,50,50),
        "y_stress": FloatTensor (30,),
        "y_nu": FloatTensor (30,),
        "y": FloatTensor (60,)   # optional
        "class_id": LongTensor (),
        "sample_key": str,
        "volfrac": FloatTensor (),
      }
    """
    def __init__(
        self,
        npz_path: str,
        return_concat_y: bool = True,
        normalize_y: bool = False,
        norm_stats: Optional[NPZNormStats] = None,
        x_as_binary: bool = True,
        x_invert: bool = False,
        dtype_x: torch.dtype = torch.float32,
    ):
        super().__init__()
        if not os.path.isfile(npz_path):
            raise FileNotFoundError(f"NPZ not found: {npz_path}")

        self.npz_path = npz_path
        self.return_concat_y = return_concat_y
        self.normalize_y = normalize_y
        self.norm_stats = norm_stats
        self.x_as_binary = x_as_binary
        self.x_invert = x_invert
        self.dtype_x = dtype_x

        # Load into memory (fast & simple for a few thousand samples)
        z = np.load(npz_path, allow_pickle=True)

        self.X = z["X"]                  # (N,50,50) uint8
        self.y_stress = z["y_stress"]    # (N,30) float32
        self.y_nu = z["y_nu"]            # (N,30) float32
        self.class_ids = z["class_ids"]  # (N,) int64
        self.sample_keys = z["sample_keys"]  # (N,) object

        self.class_names = [str(x) for x in list(z["class_names"])]
        self.target_strains = z["target_strains"].astype(np.float32)

        # Basic sanity checks
        n = self.X.shape[0]
        assert self.y_stress.shape[0] == n and self.y_nu.shape[0] == n, "N mismatch"
        assert self.X.shape[1:] == (50, 50), f"Unexpected X shape: {self.X.shape}"

        # If normalizing, stats must be present (we recommend compute on train split)
        if self.normalize_y and (self.norm_stats is None):
            raise ValueError("normalize_y=True but norm_stats is None. Compute stats on train split and pass in.")
        # Pre-compute per-sample volume fraction (solid pixel ratio). This matches the returned x
        # (i.e., respects x_as_binary and x_invert).
        self.volfrac = compute_volfrac_from_X(self.X, x_as_binary=self.x_as_binary, x_invert=self.x_invert)


    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        x = self.X[idx]  # (50,50) uint8
        if self.x_as_binary:
            x = (x > 0).astype(np.float32)
        else:
            x = x.astype(np.float32)

        if self.x_invert:
            x = 1.0 - x

        # add channel dim -> (1,50,50)
        x_t = torch.from_numpy(x).to(self.dtype_x).unsqueeze(0)

        ys = torch.from_numpy(self.y_stress[idx].astype(np.float32))
        yn = torch.from_numpy(self.y_nu[idx].astype(np.float32))

        if self.normalize_y:
            ns = self.norm_stats
            ys = (ys - ns.stress_mean) / ns.stress_std
            yn = (yn - ns.nu_mean) / ns.nu_std

        out: Dict[str, Any] = {
            "x": x_t,
            "y_stress": ys,
            "y_nu": yn,
            "class_id": torch.tensor(int(self.class_ids[idx]), dtype=torch.long),
            "sample_key": str(self.sample_keys[idx]),
            "volfrac": torch.tensor(float(self.volfrac[idx]), dtype=torch.float32),
        }
        if self.return_concat_y:
            out["y"] = torch.cat([ys, yn], dim=0)  # (60,)
        return out


def stratified_split_indices(
    class_ids: np.ndarray,
    val_ratio: float = 0.1,
    seed: int = 0,
    base_indices: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stratified split by class_ids.

    If base_indices is provided, split is performed ONLY on those indices, while
    stratifying by the corresponding class_ids.
    Returns (train_idx, val_idx) as absolute indices into the original arrays.
    """
    rng = np.random.default_rng(seed)
    class_ids = np.asarray(class_ids, dtype=np.int64)

    if base_indices is None:
        base_indices = np.arange(class_ids.shape[0], dtype=np.int64)
    else:
        base_indices = np.asarray(base_indices, dtype=np.int64)

    train_idx: List[int] = []
    val_idx: List[int] = []

    base_class_ids = class_ids[base_indices]

    for c in np.unique(base_class_ids):
        idxs_local = np.where(base_class_ids == c)[0]  # positions within base_indices
        idxs = base_indices[idxs_local]                # absolute indices
        rng.shuffle(idxs)
        n_val = max(1, int(round(len(idxs) * val_ratio))) if len(idxs) > 1 else 0
        val_idx.extend(idxs[:n_val].tolist())
        train_idx.extend(idxs[n_val:].tolist())

    train_idx = np.array(train_idx, dtype=np.int64)
    val_idx = np.array(val_idx, dtype=np.int64)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def stratified_split_threeway_indices(
    class_ids: np.ndarray,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 0,
    base_indices: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified split by class_ids into (train, val, test).

    Ratios are applied *within each class* on the provided base_indices.
    Returned indices are absolute indices into the original arrays.

    Notes:
      - If a class has too few samples, some splits may get 0 items.
      - If val_ratio/test_ratio > 0 and a class has >=2 samples, we try to allocate
        at least 1 sample to that split while keeping at least 1 sample for train.
    """
    if val_ratio < 0 or test_ratio < 0:
        raise ValueError(f"val_ratio and test_ratio must be >= 0, got {val_ratio}, {test_ratio}")
    if (val_ratio + test_ratio) >= 1.0:
        raise ValueError(f"val_ratio + test_ratio must be < 1.0, got {val_ratio + test_ratio}")

    rng = np.random.default_rng(seed)
    class_ids = np.asarray(class_ids, dtype=np.int64)

    if base_indices is None:
        base_indices = np.arange(class_ids.shape[0], dtype=np.int64)
    else:
        base_indices = np.asarray(base_indices, dtype=np.int64)

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    base_class_ids = class_ids[base_indices]

    for c in np.unique(base_class_ids):
        idxs_local = np.where(base_class_ids == c)[0]  # positions within base_indices
        idxs = base_indices[idxs_local].copy()         # absolute indices
        rng.shuffle(idxs)

        m = len(idxs)
        if m <= 1:
            # Only train
            train_idx.extend(idxs.tolist())
            continue

        n_test = int(round(m * test_ratio)) if test_ratio > 0 else 0
        n_val = int(round(m * val_ratio)) if val_ratio > 0 else 0

        # Try to allocate at least 1 to each requested split, but keep >=1 for train
        if test_ratio > 0:
            n_test = max(1, n_test)
        if val_ratio > 0:
            n_val = max(1, n_val)

        max_nontrain = m - 1  # keep at least one for train
        if (n_test + n_val) > max_nontrain:
            overflow = (n_test + n_val) - max_nontrain
            # Reduce the larger bucket first
            for _ in range(overflow):
                if n_test >= n_val and n_test > 0:
                    n_test -= 1
                elif n_val > 0:
                    n_val -= 1
                elif n_test > 0:
                    n_test -= 1

        test_idx.extend(idxs[:n_test].tolist())
        val_idx.extend(idxs[n_test:n_test + n_val].tolist())
        train_idx.extend(idxs[n_test + n_val:].tolist())

    train_idx = np.array(train_idx, dtype=np.int64)
    val_idx = np.array(val_idx, dtype=np.int64)
    test_idx = np.array(test_idx, dtype=np.int64)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx



def split_indices_with_holdout(
    class_ids: np.ndarray,
    holdout_class_ids: Optional[List[int]] = None,
    val_ratio: float = 0.1,
    seed: int = 0,
    base_indices: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create 3 splits (optionally restricted to a subset of samples via base_indices):

      - train_idx / val_idx: stratified split within the *seen* classes only (within base_indices)
      - test_idx: ALL samples from the holdout classes (within base_indices)

    base_indices:
        If provided, ONLY these indices are eligible for any split.
        This is how we implement dataset filtering (e.g., by volume fraction).
    """
    class_ids = np.asarray(class_ids, dtype=np.int64)
    n = class_ids.shape[0]
    all_idx = np.arange(n, dtype=np.int64)

    # Allowed subset mask
    if base_indices is None:
        allowed_mask = np.ones((n,), dtype=bool)
    else:
        base_indices = np.asarray(base_indices, dtype=np.int64)
        allowed_mask = np.zeros((n,), dtype=bool)
        allowed_mask[base_indices] = True

    holdout_class_ids = holdout_class_ids or []
    if not holdout_class_ids:
        seen_idx = all_idx[allowed_mask]
        train_idx, val_idx = stratified_split_indices(class_ids, val_ratio=val_ratio, seed=seed, base_indices=seen_idx)
        test_idx = np.zeros((0,), dtype=np.int64)
        return train_idx, val_idx, test_idx

    hold = np.array(sorted({int(x) for x in holdout_class_ids}), dtype=np.int64)

    # Test: holdout classes, restricted to allowed subset
    test_mask = allowed_mask & np.isin(class_ids, hold)
    test_idx = all_idx[test_mask]

    # Seen: non-holdout classes, restricted to allowed subset
    seen_mask = allowed_mask & (~np.isin(class_ids, hold))
    seen_idx = all_idx[seen_mask]

    train_idx, val_idx = stratified_split_indices(class_ids, val_ratio=val_ratio, seed=seed, base_indices=seen_idx)
    return train_idx, val_idx, test_idx


def split_indices_with_holdout_and_seen_test(
    class_ids: np.ndarray,
    holdout_class_ids: Optional[List[int]] = None,
    val_ratio: float = 0.1,
    test_ratio: float = 0.0,
    seed: int = 0,
    base_indices: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create 4 splits (optionally restricted to a subset of samples via base_indices):

      - train_idx / val_idx / test_seen_idx: stratified split within the *seen* classes only
      - test_holdout_idx: ALL samples from the holdout classes

    If test_ratio <= 0, test_seen_idx will be empty and behavior reduces to
    the original split_indices_with_holdout (train/val on seen classes, holdout test).
    """
    class_ids = np.asarray(class_ids, dtype=np.int64)
    n = class_ids.shape[0]
    all_idx = np.arange(n, dtype=np.int64)

    # Allowed subset mask
    if base_indices is None:
        allowed_mask = np.ones((n,), dtype=bool)
    else:
        base_indices = np.asarray(base_indices, dtype=np.int64)
        allowed_mask = np.zeros((n,), dtype=bool)
        allowed_mask[base_indices] = True

    holdout_class_ids = holdout_class_ids or []
    hold = np.array(sorted({int(x) for x in holdout_class_ids}), dtype=np.int64) if holdout_class_ids else np.zeros((0,), dtype=np.int64)

    # Test-1: holdout classes (unseen)
    test_holdout_mask = allowed_mask & (np.isin(class_ids, hold) if len(hold) else False)
    test_holdout_idx = all_idx[test_holdout_mask]

    # Seen set: everything else in allowed subset
    seen_mask = allowed_mask & (~np.isin(class_ids, hold) if len(hold) else True)
    seen_idx = all_idx[seen_mask]

    if test_ratio is None or float(test_ratio) <= 0.0:
        train_idx, val_idx = stratified_split_indices(class_ids, val_ratio=val_ratio, seed=seed, base_indices=seen_idx)
        test_seen_idx = np.zeros((0,), dtype=np.int64)
        return train_idx, val_idx, test_holdout_idx, test_seen_idx

    train_idx, val_idx, test_seen_idx = stratified_split_threeway_indices(
        class_ids,
        val_ratio=val_ratio,
        test_ratio=float(test_ratio),
        seed=seed,
        base_indices=seen_idx,
    )
    return train_idx, val_idx, test_holdout_idx, test_seen_idx


def make_dataloaders(
    npz_path: str,
    batch_size: int = 64,
    val_ratio: float = 0.1,
    seed: int = 0,
    num_workers: int = 4,
    pin_memory: bool = True,
    normalize_y: bool = True,
    return_concat_y: bool = True,
    holdout_class_names: Optional[List[str]] = None,
    holdout_class_ids: Optional[List[int]] = None,
    return_test_loader: bool = False,
    # --- NEW: seen-class test split (testset2) ---
    test_ratio_seen: float = 0.0,
    return_seen_test_loader: bool = False,
    # --- NEW: volume fraction filter ---
    volfrac_min: Optional[float] = None,
    volfrac_max: Optional[float] = None,
    # Keep these here so volfrac matches the returned x if you ever enable inversion
    x_as_binary: bool = True,
    x_invert: bool = False,
) -> Tuple[Any, ...]:
    """
    Create train/val (and optionally test) DataLoaders.

    Key behavior:
      - Optional filtering by volume fraction (solid pixel ratio) BEFORE splitting.
      - val split is stratified within the *seen* classes only.
      - test split contains ALL samples of the holdout classes (unseen during training).
      - normalization stats are computed from the TRAIN split only.

    Volume fraction (vf) is computed as mean(X) over pixels (after x_as_binary / x_invert),
    giving a value in [0,1]. You can filter with volfrac_min/volfrac_max.

    Returns:
      If (return_test_loader=False and return_seen_test_loader=False):
          (train_loader, val_loader, stats, info)
      Else:
          (train_loader, val_loader, test_holdout_loader, test_seen_loader, stats, info)
          where test_holdout_loader corresponds to holdout classes (testset1),
          and test_seen_loader corresponds to seen classes (testset2).
    """
    # Load once to get class_ids/names (and X for optional vf filtering)
    z = np.load(npz_path, allow_pickle=True)
    X = z["X"]  # (N,50,50)
    class_ids = z["class_ids"]
    class_names = [str(x) for x in list(z["class_names"])]
    sample_keys = z["sample_keys"]
    z.close()

    # Optional vf filtering
    base_indices = None
    volfrac = None
    if (volfrac_min is not None) or (volfrac_max is not None):
        volfrac = compute_volfrac_from_X(X, x_as_binary=x_as_binary, x_invert=x_invert)  # (N,)
        mask = np.ones((len(class_ids),), dtype=bool)
        if volfrac_min is not None:
            mask &= (volfrac >= float(volfrac_min))
        if volfrac_max is not None:
            mask &= (volfrac <= float(volfrac_max))
        base_indices = np.where(mask)[0].astype(np.int64)

    # Resolve holdout class ids (either by name or explicit ids)
    if holdout_class_ids is None:
        holdout_class_ids = []
    if holdout_class_names:
        holdout_class_ids = list(set(holdout_class_ids + resolve_class_ids_by_name(class_names, holdout_class_names)))

    train_idx, val_idx, test_holdout_idx, test_seen_idx = split_indices_with_holdout_and_seen_test(
        class_ids,
        holdout_class_ids=holdout_class_ids,
        val_ratio=val_ratio,
        test_ratio=float(test_ratio_seen or 0.0),
        seed=seed,
        base_indices=base_indices,
    )

    # Build a dataset without normalization first to compute stats from train subset
    base_train = AuxeticNPZDataset(
        npz_path,
        return_concat_y=return_concat_y,
        normalize_y=False,
        norm_stats=None,
        x_as_binary=x_as_binary,
        x_invert=x_invert,
    )

    ys = torch.from_numpy(base_train.y_stress[train_idx].astype(np.float32))
    yn = torch.from_numpy(base_train.y_nu[train_idx].astype(np.float32))
    stats = compute_norm_stats(ys, yn)

    # Datasets (with optional normalization)
    train_ds = AuxeticNPZDataset(
        npz_path,
        return_concat_y=return_concat_y,
        normalize_y=normalize_y,
        norm_stats=stats if normalize_y else None,
        x_as_binary=x_as_binary,
        x_invert=x_invert,
    )
    val_ds = AuxeticNPZDataset(
        npz_path,
        return_concat_y=return_concat_y,
        normalize_y=normalize_y,
        norm_stats=stats if normalize_y else None,
        x_as_binary=x_as_binary,
        x_invert=x_invert,
    )

    train_loader = DataLoader(
        Subset(train_ds, train_idx.tolist()),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        Subset(val_ds, val_idx.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    test_holdout_loader = None
    if return_test_loader:
        test_holdout_ds = AuxeticNPZDataset(
            npz_path,
            return_concat_y=return_concat_y,
            normalize_y=normalize_y,
            norm_stats=stats if normalize_y else None,
            x_as_binary=x_as_binary,
            x_invert=x_invert,
        )
        test_holdout_loader = DataLoader(
            Subset(test_holdout_ds, test_holdout_idx.tolist()),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    test_seen_loader = None
    if return_seen_test_loader and (len(test_seen_idx) > 0):
        test_seen_ds = AuxeticNPZDataset(
            npz_path,
            return_concat_y=return_concat_y,
            normalize_y=normalize_y,
            norm_stats=stats if normalize_y else None,
            x_as_binary=x_as_binary,
            x_invert=x_invert,
        )
        test_seen_loader = DataLoader(
            Subset(test_seen_ds, test_seen_idx.tolist()),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    holdout_class_names_resolved = [class_names[i] for i in sorted(set(holdout_class_ids))] if holdout_class_ids else []

    # Info summary (useful for config.json)
    info: Dict[str, Any] = {
        "num_samples": int(len(class_ids)),
        "volfrac_filter": None if ((volfrac_min is None) and (volfrac_max is None)) else {"min": volfrac_min, "max": volfrac_max},
        "num_kept_after_volfrac_filter": int(len(base_indices)) if base_indices is not None else int(len(class_ids)),
        "volfrac_stats_kept": None if volfrac is None else {
            "min": float(volfrac[base_indices].min()) if base_indices is not None and len(base_indices) else float(volfrac.min()),
            "max": float(volfrac[base_indices].max()) if base_indices is not None and len(base_indices) else float(volfrac.max()),
            "mean": float(volfrac[base_indices].mean()) if base_indices is not None and len(base_indices) else float(volfrac.mean()),
        },
        "num_train": int(len(train_idx)),
        "num_val": int(len(val_idx)),
        # For backward compatibility: "num_test" refers to holdout test (testset1)
        "num_test": int(len(test_holdout_idx)),
        "num_test_holdout": int(len(test_holdout_idx)),
        "num_test_seen": int(len(test_seen_idx)),
        "test_ratio_seen": float(test_ratio_seen or 0.0),
        "class_names": class_names,
        "class_counts": {class_names[i]: int((class_ids == i).sum()) for i in range(len(class_names))},
        "holdout_class_ids": [int(i) for i in sorted(set(holdout_class_ids))],
        "holdout_class_names": holdout_class_names_resolved,
        "example_sample_key": str(sample_keys[0]) if len(sample_keys) else "",
    }

    if return_test_loader or return_seen_test_loader:
        return train_loader, val_loader, test_holdout_loader, test_seen_loader, stats, info
    return train_loader, val_loader, stats, info

