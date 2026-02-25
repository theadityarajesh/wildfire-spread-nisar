"""
PyTorch Dataset that loads from pipeline zarr outputs for next-day spread prediction.

Family A: fire mask (t) + weather (t) + static topo → label = burned at t+1.
Family B: Family A + SAR backscatter (t) (Sentinel-1 VV, VH).

Samples are (optionally) random patches of a fixed size so batches can mix fires and fit in memory.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Zarr layout produced by data_pipeline.py
FIRE_MASK_VAR = "fire_mask"
WEATHER_VARS = ["pr", "tmmn", "tmmx", "vs", "th", "rmax", "rmin"]
SAR_VAR = "sar_backscatter"
STATIC_VARS = ["elevation", "slope", "aspect"]

FAMILY_A_CHANNELS = 1 + len(WEATHER_VARS) + len(STATIC_VARS)  # 11
FAMILY_B_CHANNELS = FAMILY_A_CHANNELS + 2  # VV, VH -> 13


def _normalize_fire(mask: np.ndarray) -> np.ndarray:
    """Binary or count to float [0,1]."""
    out = np.asarray(mask, dtype=np.float32)
    if out.size == 0:
        return out
    out = np.where(np.isfinite(out) & (out > 0), 1.0, 0.0)
    return out


def _normalize_weather(arr: np.ndarray, var: str) -> np.ndarray:
    """Simple scaling so values are in a reasonable range for the network."""
    out = np.asarray(arr, dtype=np.float32)
    if not np.isfinite(out).any():
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    # Rough normalization: clip and scale to ~[-1,1] or [0,1] by variable
    if var in ("pr", "rmax", "rmin"):
        out = np.clip(out, 0, 500 if var == "pr" else 100)
        out = out / (500 if var == "pr" else 100)
    elif var in ("tmmn", "tmmx"):
        out = np.clip(out, 200, 320)
        out = (out - 260) / 60
    elif var == "vs":
        out = np.clip(out, 0, 30)
        out = out / 30
    elif var == "th":
        out = out % 360
        out = (out - 180) / 180
    else:
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def _normalize_static(arr: np.ndarray, var: str) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32)
    if not np.isfinite(out).any():
        return np.nan_to_num(out, nan=0.0)
    if var == "elevation":
        out = np.clip(out, 0, 4000)
        out = out / 4000
    elif var == "slope":
        out = np.clip(out, 0, 90)
        out = out / 90
    elif var == "aspect":
        out = (out % 360) / 360
    return out


def _normalize_sar(arr: np.ndarray) -> np.ndarray:
    """dB or linear to roughly [0,1]; handle NaNs."""
    out = np.asarray(arr, dtype=np.float32)
    out = np.nan_to_num(out, nan=-30.0, posinf=0.0, neginf=-30.0)
    out = np.clip(out, -30, 0)
    out = (out + 30) / 30
    return out


class WildfireSpreadDataset(Dataset):
    """
    Load (input_t, label_t+1) from a list of processed fire directories.

    Each directory must contain: fire_masks.zarr, weather.zarr, satellite.zarr, static.zarr
    (as produced by data_pipeline.WildfireDataProcessor).

    - family: "A" (no SAR) or "B" (include SAR).
    - patch_size: if set, each __getitem__ returns a random (patch_size, patch_size) crop;
      otherwise returns full grid (can be huge; use only for small grids).
    - fire_dirs: list of paths to processed fire run directories.
    """

    def __init__(
        self,
        fire_dirs: List[Path],
        family: Literal["A", "B"] = "A",
        patch_size: Optional[int] = 128,
    ) -> None:
        self.fire_dirs = [Path(d) for d in fire_dirs]
        self.family = family.upper()
        self.patch_size = patch_size

        # Build list of (fire_idx, day_idx) for which we have day_idx+1
        self.samples: List[Tuple[int, int]] = []
        self._grid_shapes: List[Tuple[int, int]] = []

        for f_idx, fdir in enumerate(self.fire_dirs):
            fire_path = fdir / "fire_masks.zarr"
            if not fire_path.exists():
                logger.warning("Skipping %s: no fire_masks.zarr", fdir)
                continue
            try:
                fire_ds = xr.open_zarr(fire_path)
            except Exception as e:
                logger.warning("Skipping %s: failed to open fire_masks.zarr: %s", fdir, e)
                continue
            fire_da = fire_ds[FIRE_MASK_VAR]
            if "time" not in fire_da.dims or fire_da.sizes["time"] < 2:
                logger.warning("Skipping %s: need at least 2 timesteps", fdir)
                continue
            n_time = int(fire_da.sizes["time"])
            h, w = int(fire_da.sizes["y"]), int(fire_da.sizes["x"])
            self._grid_shapes.append((h, w))
            for day_idx in range(n_time - 1):
                self.samples.append((f_idx, day_idx))
            fire_ds.close()

        logger.info("WildfireSpreadDataset: %d samples from %d fires (family=%s)", len(self.samples), len(self._grid_shapes), self.family)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_fire_day(
        self,
        fire_idx: int,
        day_idx: int,
        include_sar: bool,
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        """Load full-grid features (C,H,W) and label (H,W) for one fire/day. All float32."""
        fdir = self.fire_dirs[fire_idx]
        # Fire mask at t and t+1
        fire_ds = xr.open_zarr(fdir / "fire_masks.zarr")
        fire_da = fire_ds[FIRE_MASK_VAR]
        fire_times = fire_da["time"].values
        fire_t = _normalize_fire(fire_da.isel(time=day_idx).values)
        label = _normalize_fire(fire_da.isel(time=day_idx + 1).values)
        fire_ds.close()
        h, w = fire_t.shape
        channels = [fire_t]

        # Weather at t
        weather_ds = xr.open_zarr(fdir / "weather.zarr")
        for v in WEATHER_VARS:
            if v in weather_ds.data_vars:
                arr = weather_ds[v].isel(time=day_idx).values
                channels.append(_normalize_weather(arr, v))
            else:
                channels.append(np.zeros((h, w), dtype=np.float32))
        weather_ds.close()

        # Static
        static_ds = xr.open_zarr(fdir / "static.zarr")
        for v in STATIC_VARS:
            arr = static_ds[v].values
            channels.append(_normalize_static(arr, v))
        static_ds.close()

        # SAR at t (reindex to fire time if needed)
        if include_sar:
            sat_ds = xr.open_zarr(fdir / "satellite.zarr")
            sar_da = sat_ds[SAR_VAR]
            if "time" in sar_da.dims and sar_da.sizes["time"] > 0:
                # Align SAR to fire time by nearest day
                sar_da = sar_da.reindex(time=fire_times, method="nearest")
                for b in range(sar_da.sizes["band"]):
                    ch = sar_da.isel(time=day_idx, band=b).values
                    channels.append(_normalize_sar(ch))
            else:
                for _ in range(2):
                    channels.append(np.zeros((h, w), dtype=np.float32))
            sat_ds.close()

        x = np.stack(channels, axis=0).astype(np.float32)
        return x, label, (h, w)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        fire_idx, day_idx = self.samples[idx]
        x, label, (h, w) = self._load_fire_day(fire_idx, day_idx, include_sar=(self.family == "B"))

        if self.patch_size is not None:
            ph, pw = self.patch_size, self.patch_size

            # Handle each dimension independently:
            # - if larger than patch: random crop
            # - if smaller than patch: keep full extent, then pad
            top = np.random.randint(0, h - ph + 1) if h > ph else 0
            left = np.random.randint(0, w - pw + 1) if w > pw else 0
            bottom = min(top + ph, h)
            right = min(left + pw, w)

            x_crop = x[:, top:bottom, left:right]
            label_crop = label[top:bottom, left:right]

            if x_crop.shape[1] != ph or x_crop.shape[2] != pw:
                x_pad = np.zeros((x.shape[0], ph, pw), dtype=np.float32)
                label_pad = np.zeros((ph, pw), dtype=np.float32)
                ch, cw = x_crop.shape[1], x_crop.shape[2]
                x_pad[:, :ch, :cw] = x_crop
                label_pad[:ch, :cw] = label_crop
                x, label = x_pad, label_pad
            else:
                x, label = x_crop, label_crop

        return (
            torch.from_numpy(x).float(),
            torch.from_numpy(label).float().unsqueeze(0),
        )


def train_val_test_split(
    fire_dirs: List[Path],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """Split fire directories by event (not by pixel) for train/val/test."""
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    rng = np.random.default_rng(seed)
    dirs = list(fire_dirs)
    rng.shuffle(dirs)
    n = len(dirs)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    n_test = n - n_train - n_val
    train = dirs[:n_train]
    val = dirs[n_train : n_train + n_val]
    test = dirs[n_train + n_val :]
    return train, val, test
