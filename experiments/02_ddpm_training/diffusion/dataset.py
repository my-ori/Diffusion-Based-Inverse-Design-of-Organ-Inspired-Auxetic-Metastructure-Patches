# -*- coding: utf-8 -*-
"""
Dataset wrapper for diffusion training.

- Filters samples by volume fraction (solid fraction) >= threshold.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List
import numpy as np
import torch
from torch.utils.data import Dataset
from data_loader_npz import AuxeticNPZDataset, resolve_class_ids_by_name  # your existing loader
from .utils import to_neg_one_to_one

@dataclass
class DiffusionDatasetInfo:
    num_total: int
    num_kept: int
    volfrac_min: float
    volfrac_mean_kept: float
    volfrac_min_kept: float
    volfrac_max_kept: float
    holdout_class_names: List[str] = field(default_factory=list)
    holdout_class_ids: List[int] = field(default_factory=list)


class VolumeFractionFilteredDataset(Dataset):
    """
    Returns only:
      - x: FloatTensor (1,50,50) scaled to [-1,1]
      - volfrac: float (solid fraction in [0,1])
      - class_id, sample_key (optional metadata)
    """
    def __init__(
        self,
        npz_path: str,
        volfrac_min: float = 0.30,
        augment_flip: bool = True,
        return_meta: bool = True,
        holdout_class_names: Optional[List[str]] = None,
        holdout_class_ids: Optional[List[int]] = None,
    ):
        super().__init__()
        self.base = AuxeticNPZDataset(
            npz_path,
            return_concat_y=False,
            normalize_y=False,
            norm_stats=None,
            x_as_binary=True,
            x_invert=False,
            dtype_x=torch.float32,
        )
        self.volfrac_min = float(volfrac_min)
        self.augment_flip = bool(augment_flip)
        self.return_meta = bool(return_meta)

        # base.X is uint8 (0/1). Use >0 to be robust.
        X = self.base.X
        solids = (X.reshape(X.shape[0], -1) > 0).mean(axis=1)  # (N,)
        keep = np.where(solids >= self.volfrac_min)[0].astype(np.int64)

        holdout_class_names = holdout_class_names or []
        holdout_class_ids = holdout_class_ids or []

        # resolve holdout ids from names (robust matching)
        resolved_ids = resolve_class_ids_by_name(self.base.class_names, holdout_class_names) if holdout_class_names else []
        hold_ids = sorted(set([int(x) for x in holdout_class_ids] + [int(x) for x in resolved_ids]))

        if hold_ids:
            class_ids = np.asarray(self.base.class_ids, dtype=np.int64)
            hold_mask = np.isin(class_ids, np.asarray(hold_ids, dtype=np.int64))  # (N,)
            keep = keep[~hold_mask[keep]]  # drop held-out classes from the kept set

        if keep.size == 0:
            raise ValueError(
                f"No samples remain after filtering: volfrac_min={self.volfrac_min:.3f}, "
                f"holdout_class_names={holdout_class_names}, holdout_class_ids={hold_ids}."
            )

        self.keep_idx = keep
        kept_solids = solids[keep]
        self.info = DiffusionDatasetInfo(
            num_total=int(X.shape[0]),
            num_kept=int(keep.size),
            volfrac_min=self.volfrac_min,
            volfrac_mean_kept=float(kept_solids.mean()),
            volfrac_min_kept=float(kept_solids.min()),
            volfrac_max_kept=float(kept_solids.max()),
            holdout_class_names=[str(n) for n in holdout_class_names],
            holdout_class_ids=[int(i) for i in hold_ids],
        )

    def __len__(self) -> int:
        return int(self.keep_idx.size)

    def _maybe_augment(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1,50,50) in {0,1}
        if not self.augment_flip:
            return x
        if torch.rand(()) < 0.5:
            x = torch.flip(x, dims=[2])  # flip W
        if torch.rand(()) < 0.5:
            x = torch.flip(x, dims=[1])  # flip H
        return x

    def __getitem__(self, i: int) -> Dict[str, Any]:
        idx = int(self.keep_idx[i])
        item = self.base[idx]
        x01 = item["x"]  # (1,50,50) in {0,1}
        x01 = self._maybe_augment(x01)

        volfrac = float(x01.mean().item())
        x = to_neg_one_to_one(x01)

        out: Dict[str, Any] = {"x": x, "volfrac": volfrac}
        if self.return_meta:
            out["class_id"] = item["class_id"]
            out["sample_key"] = item["sample_key"]
        return out
