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


def split_indices_with_holdout(
    class_ids: np.ndarray,
    holdout_class_ids: Optional[List[int]] = None,
    val_ratio: float = 0.1,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create 3 splits:
      - train_idx / val_idx: stratified over *seen* classes only
      - test_idx: all samples whose class_id is in holdout_class_ids

    holdout_class_ids: list of integer class ids to hold out for testing
    """
    class_ids = np.asarray(class_ids, dtype=np.int64)
    n = class_ids.shape[0]

    if not holdout_class_ids:
        train_idx, val_idx = stratified_split_indices(class_ids, val_ratio=val_ratio, seed=seed)
        test_idx = np.zeros((0,), dtype=np.int64)
        return train_idx, val_idx, test_idx

    hold = set(int(x) for x in holdout_class_ids)
    all_idx = np.arange(n, dtype=np.int64)

    test_mask = np.isin(class_ids, np.array(sorted(list(hold)), dtype=np.int64))
    test_idx = all_idx[test_mask]

    seen_idx = all_idx[~test_mask]
    train_idx, val_idx = stratified_split_indices(class_ids, val_ratio=val_ratio, seed=seed, base_indices=seen_idx)
    return train_idx, val_idx, test_idx


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
) -> Tuple[Any, ...]:
    """
    Create train/val (and optionally test) DataLoaders.

    Key behavior:
      - val split is stratified within the *seen* classes.
      - test split contains ALL samples of the holdout classes.
      - normalization stats are computed from the TRAIN split only.

    Returns:
      If return_test_loader=False:
          (train_loader, val_loader, stats, info)
      If return_test_loader=True:
          (train_loader, val_loader, test_loader, stats, info)
    """
    # Load once to get class_ids and names
    z = np.load(npz_path, allow_pickle=True)
    class_ids = z["class_ids"]
    class_names = [str(x) for x in list(z["class_names"])]
    sample_keys = z["sample_keys"]
    z.close()

    # Resolve holdout class ids (either by name or explicit ids)
    if holdout_class_ids is None:
        holdout_class_ids = []
    if holdout_class_names:
        holdout_class_ids = list(set(holdout_class_ids + resolve_class_ids_by_name(class_names, holdout_class_names)))

    train_idx, val_idx, test_idx = split_indices_with_holdout(
        class_ids,
        holdout_class_ids=holdout_class_ids,
        val_ratio=val_ratio,
        seed=seed,
    )

    # Build a dataset without normalization first to compute stats from train subset
    base_train = AuxeticNPZDataset(
        npz_path,
        return_concat_y=return_concat_y,
        normalize_y=False,
        norm_stats=None,
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
    )
    val_ds = AuxeticNPZDataset(
        npz_path,
        return_concat_y=return_concat_y,
        normalize_y=normalize_y,
        norm_stats=stats if normalize_y else None,
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

    test_loader = None
    if return_test_loader:
        test_ds = AuxeticNPZDataset(
            npz_path,
            return_concat_y=return_concat_y,
            normalize_y=normalize_y,
            norm_stats=stats if normalize_y else None,
        )
        test_loader = DataLoader(
            Subset(test_ds, test_idx.tolist()),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    holdout_class_names_resolved = [class_names[i] for i in sorted(set(holdout_class_ids))] if holdout_class_ids else []

    info: Dict[str, Any] = {
        "num_samples": int(len(class_ids)),
        "num_train": int(len(train_idx)),
        "num_val": int(len(val_idx)),
        "num_test": int(len(test_idx)),
        "class_names": class_names,
        "class_counts": {class_names[i]: int((class_ids == i).sum()) for i in range(len(class_names))},
        "holdout_class_ids": [int(i) for i in sorted(set(holdout_class_ids))],
        "holdout_class_names": holdout_class_names_resolved,
        "example_sample_key": str(sample_keys[0]) if len(sample_keys) else "",
    }

    if return_test_loader:
        return train_loader, val_loader, test_loader, stats, info
    return train_loader, val_loader, stats, info
